import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import time
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
from thop import profile  # For calculating FLOPs
import math
import os
from collections import defaultdict
import gc
import argparse
from datetime import timedelta  # For distributed timeout


GLOBAL_TRAIN_BATCH_INDICES = None

def parse_args():
    """
    Parse command line arguments for training hyperparameters
    """
    parser = argparse.ArgumentParser(description='ImageNet Classification with Batch Pruning using DDP')
    
    # Training hyperparameters
    parser.add_argument('--total_epochs', type=int, default=200,
                        help='Total number of training epochs (default: 200)')
    parser.add_argument('--batch_size_per_gpu', type=int, default=256,
                        help='Batch size per GPU (default: 256)')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Base learning rate (will be scaled by world_size) (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum (default: 0.9)')
    
    # Threshold parameters
    parser.add_argument('--delta_start', type=float, default=0.000005,
                        help='Starting threshold for exponential schedule (default: 0.000005)')
    parser.add_argument('--delta_end', type=float, default=0.00005,
                        help='Ending threshold for exponential schedule (default: 0.00005)')
    
    # Model selection
    parser.add_argument('--model', type=str, default='resnet50', 
                        choices=['resnet18', 'resnet50'],
                        help='Model architecture (default: resnet50)')
    
    # Data paths
    parser.add_argument('--train_dir', type=str, default='/work/mustakin/imagenet/ILSVRC2012_img_train',
                        help='Path to ImageNet training directory')
    parser.add_argument('--synset_path', type=str, default='/work/mustakin/imagenet/synset_words.txt',
                        help='Path to synset_words.txt file')
    
    return parser.parse_args()

def configure_distributed_gpu():
    """
    Configure distributed GPU settings for PyTorch DDP
    """
    # Set NCCL timeout to avoid premature timeouts (in seconds)
    os.environ['NCCL_TIMEOUT'] = '7200'  # 2 hours
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_DEBUG'] = 'WARN'  # Reduce debug verbosity
    
    # Initialize distributed training
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Environment variables for distributed training not found")
        rank = 0
        world_size = 1
        local_rank = 0
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Initialize process group with increased timeout
    if world_size > 1:
        dist.init_process_group(backend="nccl", 
                               init_method="env://",
                               world_size=world_size, 
                               rank=rank,
                               timeout=timedelta(seconds=7200))
        print(f"Rank {rank}/{world_size} initialized on GPU {local_rank}")
    
    # Optimize GPU memory settings for A100
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        # Enable memory efficient attention if available
        torch.backends.cuda.enable_flash_sdp(True)
        # Set memory fraction to avoid OOM with large models
        torch.cuda.set_per_process_memory_fraction(0.95, device)
        
        if rank == 0:  # Only print from main process
            print(f"Total GPUs Available: {torch.cuda.device_count()}")
            print(f"Using {world_size} GPUs for distributed training")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1e9:.1f} GB")
    
    return device, rank, world_size, local_rank

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, synset_path, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            synset_path (string): Path to the synset_words.txt file.
            transform (callable, optional): Transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load synset mappings
        self.synset_to_class_idx = {}
        self.idx_to_class_name = {}
        idx = 0
        with open(synset_path, 'r') as f:
            for line in f:
                synset, class_name = line.strip().split(' ', 1)
                self.synset_to_class_idx[synset] = idx
                self.idx_to_class_name[idx] = class_name
                idx += 1
        
        # Number of classes
        self.num_classes = len(self.synset_to_class_idx)
        print(f"Loaded {self.num_classes} classes from synset file")
        
        # List all images in each synset folder
        self.image_paths = []
        self.labels = []
        self.synset_indices = defaultdict(list)  # Track indices for each class for stratified split
        
        for synset in os.listdir(root_dir):
            synset_path = os.path.join(root_dir, synset)
            if not os.path.isdir(synset_path):
                continue
                
            # Get class index for this synset
            class_idx = self.synset_to_class_idx.get(synset, -1)
            if class_idx == -1:
                continue
                
            # Add all images in this synset folder
            for img_name in os.listdir(synset_path):
                if img_name.endswith('.JPEG'):
                    img_path = os.path.join(synset_path, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
                    self.synset_indices[class_idx].append(len(self.image_paths) - 1)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Open image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            image = Image.new('RGB', (224, 224))
        
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class DistributedBatchSampler:
    """
    A distributed sampler that:
    - Maintains consistent global batch composition across epochs
    - Assigns correct subset to each GPU
    - Supports intra-batch shuffling
    - Works with standard DataLoader
    """
    def __init__(self, dataset, batch_size_per_gpu, world_size, rank, shuffle_within_batch=True):
        self.dataset = dataset
        self.batch_size_per_gpu = batch_size_per_gpu
        self.world_size = world_size
        self.rank = rank
        self.shuffle_within_batch = shuffle_within_batch
        self.epoch = 0

        # Total batch size across all GPUs
        self.global_batch_size = batch_size_per_gpu * world_size

        # Create fixed global batch assignments
        total_size = len(dataset)
        indices = np.arange(total_size)
        np.random.shuffle(indices)  # Shuffle once at initialization

        # Group into full global batches (drop last incomplete batch)
        self.global_batch_indices = [
            indices[i:i + self.global_batch_size].tolist()
            for i in range(0, total_size, self.global_batch_size)
            if i + self.global_batch_size <= total_size
        ]

        # Store for pruning logic
        global GLOBAL_TRAIN_BATCH_INDICES
        if GLOBAL_TRAIN_BATCH_INDICES is None:
            GLOBAL_TRAIN_BATCH_INDICES = self.global_batch_indices

        # Assign local batches for this GPU
        self.local_batch_indices = []
        for global_batch in self.global_batch_indices:
            start = rank * batch_size_per_gpu
            end = (rank + 1) * batch_size_per_gpu
            self.local_batch_indices.append(global_batch[start:end])

    def __iter__(self):
        # Copy local batch indices
        local_batches = [b.copy() for b in self.local_batch_indices]

        # shuffle order of batches between epochs
        # (Comment out if you want fixed batch order across epochs)
        # np.random.shuffle(local_batches)  

        for batch in local_batches:
            if self.shuffle_within_batch:
                np.random.shuffle(batch)
            yield from batch  # Yield individual indices

    def __len__(self):
        return sum(len(b) for b in self.local_batch_indices)

    def set_active_batches(self, active_batch_indices):
        
        if active_batch_indices is None:
            self.local_batch_indices = [
                global_batch[self.rank * self.batch_size_per_gpu : (self.rank + 1) * self.batch_size_per_gpu]
                for global_batch in GLOBAL_TRAIN_BATCH_INDICES
            ]
        else:
            filtered_global_batches = [
                GLOBAL_TRAIN_BATCH_INDICES[i] for i in active_batch_indices
            ]
            self.local_batch_indices = [
                global_batch[self.rank * self.batch_size_per_gpu : (self.rank + 1) * self.batch_size_per_gpu]
                for global_batch in filtered_global_batches
            ]

def create_three_way_split_indices(dataset, val_size=10000, test_ratio=0.0391):
    """
    Create indices for a three-way split of the dataset:
    1. First split: take val_size images for validation (stratified)
    2. Second split: take test_ratio for test set (stratified)
    3. Rest: training set
    
    Args:
        dataset: ImageNet dataset with synset_indices
        val_size: Number of images for validation set (default: 10000)
        test_ratio: Ratio of remaining data to use for test set (default: 0.0391)
    
    Returns:
        train_indices, val_indices, test_indices
    """
    train_indices = []
    val_indices = []
    test_indices = []
    
    # Calculate how many samples per class for validation (stratified)
    num_classes = len(dataset.synset_indices)
    val_per_class = val_size // num_classes
    remainder = val_size % num_classes
    
    # For each class, split its samples
    for class_idx, indices in dataset.synset_indices.items():
        # Shuffle indices for this class
        indices_shuffled = indices.copy()
        np.random.shuffle(indices_shuffled)
        
        # Determine validation size for this class
        current_val_size = val_per_class + (1 if class_idx < remainder else 0)
        
        # Split off validation set
        val_indices.extend(indices_shuffled[:current_val_size])
        remaining_indices = indices_shuffled[current_val_size:]
        
        # From remaining, calculate test split
        test_split_idx = int(len(remaining_indices) * (1 - test_ratio))
        
        # Split into train and test
        train_indices.extend(remaining_indices[:test_split_idx])
        test_indices.extend(remaining_indices[test_split_idx:])
    
    return train_indices, val_indices, test_indices

def load_images_ddp(batch_size_per_gpu, rank, world_size, train_dir, synset_path):
    """
    Loads the ImageNet dataset for DDP training with:
    - 10,000 images for validation
    - 0.0391 ratio for test set 
    - Rest for training
    """
    # Define transformations
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load full dataset
    full_dataset = ImageNetDataset(
        root_dir=train_dir,
        synset_path=synset_path,
        transform=None
    )

    # Three-way stratified split: 10k validation, 0.0391 test ratio from remaining, rest training
    train_indices, val_indices, test_indices = create_three_way_split_indices(
        full_dataset, val_size=10000, test_ratio=0.0391
    )
    
    if rank == 0:
        print(f"Split dataset into:")
        print(f"  Training: {len(train_indices)} samples")
        print(f"  Validation: {len(val_indices)} samples")
        print(f"  Test: {len(test_indices)} samples")

    # Create subsets
    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)

    # Apply transforms via wrapper
    class TransformSubset(Dataset):
        def __init__(self, subset, transform=None):
            self.subset = subset
            self.transform = transform
        def __getitem__(self, idx):
            x, y = self.subset[idx]
            if self.transform:
                x = self.transform(x)
            return x, y
        def __len__(self):
            return len(self.subset)

    train_dataset = TransformSubset(train_subset, transform_train)
    val_dataset = TransformSubset(val_subset, transform_val)
    test_dataset = TransformSubset(test_subset, transform_val)

    # Create distributed batch sampler (handles consistent batches)
    sampler = DistributedBatchSampler(
        dataset=train_dataset,
        batch_size_per_gpu=batch_size_per_gpu,
        world_size=world_size,
        rank=rank,
        shuffle_within_batch=True
    )

    # Training loader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_per_gpu,
        sampler=sampler,
        num_workers=4,  # Reduced from 8 to prevent resource contention
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True  # Ensure all ranks have same number of batches
    )

    # Validation sampler (standard DistributedSampler)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_per_gpu,
        sampler=val_sampler,
        num_workers=4,  # Reduced from 8
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True  # Ensure all ranks have same number of batches
    )

    # Test sampler
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size_per_gpu,
        sampler=test_sampler,
        num_workers=4,  # Reduced from 8
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
        drop_last=True  # Ensure all ranks have same number of batches
    )

    # Class names
    class_names = [full_dataset.idx_to_class_name[i] for i in range(len(full_dataset.idx_to_class_name))]

    if rank == 0:
        print(f"Created efficient batch setup with {len(GLOBAL_TRAIN_BATCH_INDICES)} global batches")
        print(f"Each batch contains the same data samples across all epochs")
        print(f"Intra-batch shuffling is enabled")

    return train_loader, val_loader, test_loader, class_names, val_sampler, test_sampler, sampler

# ResNet model definitions (keeping the same as original)
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64
        
        # List to track layer names for activation tracking
        self.layer_names = ['layer1', 'layer2', 'layer3', 'layer4']
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(0.2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        
        # List of layers to track activations
        self.conv_layers = [self.layer1, self.layer2, self.layer3, self.layer4]

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, return_activations=False):
        # Initialize list to store activations if requested
        activations = []
        
        # Initial convolution and pooling for ImageNet
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        
        # ResNet blocks with activation tracking
        out = self.layer1(out)
        if return_activations:
            activations.append(out.detach())
        
        out = self.layer2(out)
        if return_activations:
            activations.append(out.detach())
        
        out = self.layer3(out)
        if return_activations:
            activations.append(out.detach())
        
        out = self.layer4(out)
        if return_activations:
            activations.append(out.detach())
        
        # Global average pooling and final classifier
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.linear(out)
        
        if return_activations:
            return out, activations
        return out

def ResNet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def ResNet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def train_epoch_ddp(model, device, train_loader, optimizer, epoch, batch_indices=None,
                    scheduler=None, threshold=0.0001, rank=0, world_size=1, sampler=None):
    """
    Train one epoch using efficient DataLoader + DistributedBatchSampler.
    """
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    batch_size_per_gpu = train_loader.batch_size

    std_devs = {layer_name: [] for layer_name in model.module.layer_names}
    batch_indices_list = []

    start_time = time.time()
    total_images_processed = 0
    batch_processing_times = []

    # Set active batches in sampler
    if sampler is not None:
        sampler.set_active_batches(batch_indices)

    # Determine total batches to process
    global GLOBAL_TRAIN_BATCH_INDICES
    total_logical_batches = len(GLOBAL_TRAIN_BATCH_INDICES)
    active_batch_indices = batch_indices if batch_indices is not None else list(range(total_logical_batches))
    if rank == 0:
        print(f"Using {len(active_batch_indices)} out of {total_logical_batches} global batches")

    # Synchronize before starting epoch to ensure all ranks are ready
    if world_size > 1:
        dist.barrier()
    
    # Iterate directly over DataLoader - all ranks iterate together
    for batch_idx, (data, target) in enumerate(train_loader):
        # Map enumeration index to logical batch index
        if batch_indices is not None and batch_idx < len(active_batch_indices):
            logical_batch_idx = active_batch_indices[batch_idx]
        else:
            logical_batch_idx = batch_idx
        
        batch_start_time = time.time()
        batch_indices_list.append(logical_batch_idx)

        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # Mixup
        if np.random.random() > 0.5:
            lam = np.random.beta(0.2, 0.2)
            rand_idx = torch.randperm(data.size(0)).to(device)
            mixed_data = lam * data + (1 - lam) * data[rand_idx]
            target_a, target_b = target, target[rand_idx]
            optimizer.zero_grad()
            output, activations = model(mixed_data, return_activations=True)
            loss = lam * F.cross_entropy(output, target_a) + (1 - lam) * F.cross_entropy(output, target_b)
        else:
            optimizer.zero_grad()
            output, activations = model(data, return_activations=True)
            loss = F.cross_entropy(output, target)

        # Track activation std
        for layer_idx, act in enumerate(activations):
            std_dev = torch.std(act).item()
            std_devs[model.module.layer_names[layer_idx]].append(std_dev)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        _, pred = output.max(1)
        total += target.size(0)
        correct += pred.eq(target).sum().item()

        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time
        batch_processing_times.append(batch_time)
        total_images_processed += data.size(0)

        if rank == 0 and total_images_processed % (100 * batch_size_per_gpu) == 0:
            print(f'Train Epoch: {epoch} [{total_images_processed}/{len(train_loader) * batch_size_per_gpu * world_size} '
                  f'({100. * (batch_idx + 1) / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\tAccuracy: {100. * correct / total:.2f}%')

        if (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Synchronize all ranks before gathering metrics
    if world_size > 1:
        torch.cuda.synchronize()
        dist.barrier()
    
    # Gather metrics from all processes
    if world_size > 1:
        train_loss_tensor = torch.tensor(train_loss, device=device)
        correct_tensor = torch.tensor(correct, device=device)
        total_tensor = torch.tensor(total, device=device)
        dist.all_reduce(train_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        train_loss = train_loss_tensor.item() / world_size
        correct = correct_tensor.item()
        total = total_tensor.item()

    end_time = time.time()
    training_time = end_time - start_time

    # Build std DataFrame
    std_df = pd.DataFrame(std_devs)
    std_df['batch_idx'] = batch_indices_list
    std_df = std_df.set_index('batch_idx')
    std_df['mean_std'] = std_df.mean(axis=1)
    
    # Synchronize activation statistics across all ranks for consistent pruning decisions
    if world_size > 1 and len(batch_indices_list) > 0:
        # Average the activation statistics across all GPUs
        # This ensures pruning decisions are based on the full global batch, not just rank 0's portion
        for layer_name in model.module.layer_names:
            if layer_name in std_df.columns:
                layer_values = torch.tensor(std_df[layer_name].values, dtype=torch.float32, device=device)
                dist.all_reduce(layer_values, op=dist.ReduceOp.SUM)
                std_df[layer_name] = (layer_values / world_size).cpu().numpy()
        
        # Recompute mean_std with averaged values
        std_df['mean_std'] = std_df[model.module.layer_names].mean(axis=1)

    avg_batch_time = np.mean(batch_processing_times) if batch_processing_times else 0
    images_per_second = total_images_processed / training_time if training_time > 0 else 0

    epoch_loss = train_loss / len(batch_indices_list) if batch_indices_list else 0
    epoch_acc = 100. * correct / total if total > 0 else 0

    if scheduler:
        scheduler.step()

    if rank == 0:
        print(f"\nEpoch {epoch} completed in {training_time:.2f} seconds")
        print(f"Training Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        print(f"Throughput: {images_per_second * world_size:.2f} images/sec (global)")
        print(f"Current threshold: {threshold:.8f}")

    throughput_metrics = {
        'epoch_time': training_time,
        'images_per_second': images_per_second,
        'avg_batch_time': avg_batch_time,
        'total_batches': len(batch_indices_list)
    }

    return epoch_loss, epoch_acc, std_df, batch_indices_list, throughput_metrics

def validate_ddp(model, device, test_loader, rank=0, world_size=1):
    """
    Validates the model on the test set with DDP
    """
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            # Move data to device
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            
            # Forward pass
            output = model(data)
            
            # Compute loss
            val_loss += F.cross_entropy(output, target).item()
            
            # Calculate accuracy
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            
            # Store predictions and targets for confusion matrix (only on rank 0)
            if rank == 0:
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
    
    # Gather metrics from all processes
    if world_size > 1:
        # Synchronize before all_reduce to prevent hangs
        torch.cuda.synchronize()
        dist.barrier()
        
        val_loss_tensor = torch.tensor(val_loss, device=device)
        correct_tensor = torch.tensor(correct, device=device)
        total_tensor = torch.tensor(total, device=device)
        
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
        
        val_loss = val_loss_tensor.item() / world_size
        correct = correct_tensor.item()
        total = total_tensor.item()
    
    # Calculate average loss and accuracy
    val_loss /= len(test_loader)
    val_acc = 100. * correct / total
    
    if rank == 0:
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
    
    return val_loss, val_acc, all_targets, all_preds

def calculate_exponential_threshold(delta_start, delta_end, T, t):
    """
    Calculate threshold using exponential function
    """
    alpha = math.log(delta_end / delta_start) / T
    threshold = delta_start * math.exp(alpha * t)
    return threshold

def run_imagenet_classification_ddp(args):
    """
    Main function to run ImageNet classification with DDP and batch pruning
    """
    # Configure distributed training
    device, rank, world_size, local_rank = configure_distributed_gpu()
    
    # Training hyperparameters from args
    total_epochs = args.total_epochs
    batch_size_per_gpu = args.batch_size_per_gpu
    learning_rate = args.learning_rate * world_size  # Scale learning rate with number of GPUs
    weight_decay = args.weight_decay
    momentum = args.momentum
    
    # Exponential threshold parameters from args
    delta_start = args.delta_start
    delta_end = args.delta_end
    
    # Early stopping parameter
    min_batch_percentage = 0  # Set to 0 as requested
    
    # Load data with 10k validation and 0.0391 test ratio
    train_loader, val_loader, test_loader, class_names, val_sampler, test_sampler, batch_sampler = load_images_ddp(
        batch_size_per_gpu, rank, world_size, args.train_dir, args.synset_path
    )
    
    # Calculate max batches per epoch
    total_original_batches = len(train_loader)
    max_epochs = total_epochs
    
    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"Model: {args.model}")
        print(f"Total epochs: {total_epochs}")
        print(f"Batch size per GPU: {batch_size_per_gpu}")
        print(f"Total global batch size: {batch_size_per_gpu * world_size}")
        print(f"Base learning rate: {args.learning_rate}")
        print(f"Scaled learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Momentum: {momentum}")
        print(f"Delta start: {delta_start}")
        print(f"Delta end: {delta_end}")
        print(f"Total original batches per GPU: {total_original_batches}")
        print(f"Maximum number of epochs: {max_epochs}")
        print(f"Early stopping threshold: {min_batch_percentage*100}% of original batches")
        print(f"{'='*60}\n")
    
    # Create model based on args
    if args.model == 'resnet18':
        model = ResNet18(num_classes=1000).to(device)
    elif args.model == 'resnet50':
        model = ResNet50(num_classes=1000).to(device)
    else:
        raise ValueError(f"Unknown model: {args.model}")
    
    # Wrap model in DDP
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    
    # Print model summary (only from rank 0)
    if rank == 0:
        print("Model architecture:")
        print(model)
        
        # Calculate model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModel Parameters:")
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
    
    # Define optimizer and scheduler
    optimizer = optim.SGD(model.parameters(), 
                         lr=learning_rate, 
                         momentum=momentum, 
                         weight_decay=weight_decay,
                         nesterov=True)
    
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 80], gamma=0.1)
    
    # Training and validation history
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    # Metrics to track
    epoch_times = []
    batch_counts = []
    
    # Initialize tracking for batch dropping with exponential threshold
    prev_epoch_std_df = None
    batch_indices_to_use = None
    total_dropped_batches = 0
    total_remaining_batches = []
    total_batches_dropped_per_epoch = []
    
    # New tracking for exponential threshold
    threshold_history = []
    pruned_percentage_history = []
    
    # Track total training time
    total_training_start = time.time()
    
    # Variables for early stopping
    best_val_acc = 0
    early_stopped = False
    
    # Train and evaluate
    for epoch in range(1, total_epochs + 1):
        if rank == 0:
            print(f"\n{'='*50}")
            print(f"EPOCH {epoch}/{total_epochs}")
            print(f"{'='*50}")
        
        # Calculate threshold using exponential function
        current_threshold = calculate_exponential_threshold(delta_start, delta_end, total_epochs, epoch)
        
        # Record threshold history
        if rank == 0:
            threshold_history.append(current_threshold)
            print(f"Current threshold (δ({epoch})): {current_threshold:.8f}")
        
        # Synchronize all processes before training
        if world_size > 1:
            dist.barrier()
        
        # Train for one epoch using the selected batch indices
        train_loss, train_acc, current_std_df, used_batch_indices, throughput_metrics = train_epoch_ddp(
            model, device, train_loader, optimizer, epoch,
            batch_indices=batch_indices_to_use,
            scheduler=scheduler,
            threshold=current_threshold,
            rank=rank,
            world_size=world_size,
            sampler=batch_sampler
        )
        
        # Record metrics (only on rank 0)
        if rank == 0:
            epoch_times.append(throughput_metrics['epoch_time'])
            batch_counts.append(throughput_metrics['total_batches'])
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Track remaining batches for this epoch
            current_remaining_batches = len(used_batch_indices) if used_batch_indices is not None else total_original_batches
            total_remaining_batches.append(current_remaining_batches)
            
            # Calculate and record pruned percentage
            current_pruned_percentage = (1 - current_remaining_batches / total_original_batches) * 100
            pruned_percentage_history.append(current_pruned_percentage)
        
        # Synchronize before validation
        if world_size > 1:
            dist.barrier()
        
        # Set validation sampler epoch for proper shuffling
        val_sampler.set_epoch(epoch)
        
        # Evaluate on validation set
        val_loss, val_acc, all_targets, all_preds = validate_ddp(
            model, device, val_loader, rank=rank, world_size=world_size
        )
        
        # Record validation metrics (only on rank 0)
        if rank == 0:
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            
            # Save best model so far (to disk, so all ranks can load it later)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                # Save model state dict from the underlying module
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), f'best_{args.model}_imagenet_ddp.pth')
                print(f"✓ New best validation accuracy: {best_val_acc:.2f}% (saved to disk)")


        
        # Initialize variables for batch dropping
        batches_dropped_this_epoch = 0
        
        # After the first epoch, we start the batch dropping mechanism
        # Note: All ranks now have synchronized std_df, so any rank could make the decision
        # But we keep it on rank 0 for consistency and broadcast the result
        if rank == 0 and epoch > 1 and prev_epoch_std_df is not None:
            # Find matching batch indices between previous and current epoch
            common_indices = set(prev_epoch_std_df.index).intersection(set(current_std_df.index))
            
            # Initialize list to track which batches to drop
            batches_to_drop = []
            
            # Compare mean standard deviations for each batch using exponential threshold
            for idx in common_indices:
                prev_mean_std = prev_epoch_std_df.loc[idx, 'mean_std']
                curr_mean_std = current_std_df.loc[idx, 'mean_std']
                
                # If the absolute difference is below the exponential threshold, mark this batch for dropping
                if abs(prev_mean_std - curr_mean_std) <= current_threshold:
                    batches_to_drop.append(idx)
            
            # Print information about dropped batches
            if batches_to_drop:
                print(f"\nDropping {len(batches_to_drop)} batches for the next epoch:")
                print(f"Batch indices: {batches_to_drop[:10]}{'...' if len(batches_to_drop) > 10 else ''}")
                total_dropped_batches += len(batches_to_drop)
                batches_dropped_this_epoch = len(batches_to_drop)
            else:
                print("\nNo batches to drop for the next epoch")
            
            # Update batch indices for the next epoch (exclude the ones to drop)
            if batch_indices_to_use is None:
                # If we were using all batches, create a list of all batch indices except those to drop
                batch_indices_to_use = [i for i in range(total_original_batches) if i not in batches_to_drop]
            else:
                # Otherwise, filter the current indices
                batch_indices_to_use = [i for i in batch_indices_to_use if i not in batches_to_drop]
            
            # Calculate percent of original batches remaining
            percent_remaining = len(batch_indices_to_use) / total_original_batches * 100
            print(f"\nBatches remaining: {len(batch_indices_to_use)}/{total_original_batches} ({percent_remaining:.2f}%)")
            print(f"Total batches dropped so far: {total_dropped_batches}/{total_original_batches} ({total_dropped_batches/total_original_batches*100:.2f}%)")
            
            # Early stopping if we've dropped too many batches
            if batch_indices_to_use is not None and len(batch_indices_to_use) <= total_original_batches * min_batch_percentage:
                print(f"\nStopping early as fewer than {min_batch_percentage*100}% of original batches remain")
                early_stopped = True
        
        # Broadcast batch indices and early stopping flag to all processes
        if world_size > 1:
            # Broadcast early stopping flag
            early_stop_tensor = torch.tensor(1 if early_stopped else 0, dtype=torch.long, device=device)
            dist.broadcast(early_stop_tensor, src=0)
            if rank != 0:
                early_stopped = bool(early_stop_tensor.item())
            
            if rank == 0:
                # Prepare data to broadcast
                if batch_indices_to_use is not None:
                    batch_indices_tensor = torch.tensor(batch_indices_to_use, dtype=torch.long, device=device)
                    size_tensor = torch.tensor(len(batch_indices_tensor), dtype=torch.long, device=device)
                else:
                    # Use -1 as a sentinel value to indicate None
                    batch_indices_tensor = torch.tensor([-1], dtype=torch.long, device=device)
                    size_tensor = torch.tensor(1, dtype=torch.long, device=device)
            else:
                # Initialize dummy values for non-rank-0 processes
                size_tensor = torch.tensor(0, dtype=torch.long, device=device)
                batch_indices_tensor = torch.tensor([], dtype=torch.long, device=device)
            
            # Broadcast size first
            dist.broadcast(size_tensor, src=0)
            
            # Prepare tensor on non-rank-0 processes
            if rank != 0:
                batch_indices_tensor = torch.zeros(size_tensor.item(), dtype=torch.long, device=device)
            
            # Broadcast batch indices
            dist.broadcast(batch_indices_tensor, src=0)
            
            # Convert back to list (or None if placeholder)
            if len(batch_indices_tensor) == 1 and batch_indices_tensor[0].item() == -1:
                batch_indices_to_use = None
            else:
                batch_indices_to_use = batch_indices_tensor.cpu().tolist()
        
        # Record batches dropped this epoch (only on rank 0)
        if rank == 0:
            total_batches_dropped_per_epoch.append(batches_dropped_this_epoch)
        
        # All ranks must store the synchronized std_df for consistent next-epoch comparisons
        prev_epoch_std_df = current_std_df
        
        # Check for early stopping
        if early_stopped:
            if rank == 0:
                print(f"Training stopped early at epoch {epoch}")
            break
        
        # Synchronize before next epoch
        if world_size > 1:
            dist.barrier()
        
        
        # Clear cache periodically
        if epoch % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()
    
    # === POST-TRAINING EVALUATION AND SUMMARY (Always executes) ===
    
    # Synchronize before loading best model (ensure rank 0 finished saving)
    if world_size > 1:
        dist.barrier()
    
    # ALL RANKS load the best model from disk
    best_model_path = f'best_{args.model}_imagenet_ddp.pth'
    if os.path.exists(best_model_path):
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(torch.load(best_model_path, map_location=device))
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"All ranks loaded best model (val_acc: {best_val_acc:.2f}%)")
            print(f"{'='*60}")
    else:
        if rank == 0:
            print(f"\n{'='*60}")
            print(f"Warning: No best model checkpoint found, using last trained model")
            print(f"{'='*60}")
    
    # Synchronize after loading (ensure all ranks have the same model)
    if world_size > 1:
        dist.barrier()
    
    # Calculate total training time
    total_training_end = time.time()
    total_training_time = total_training_end - total_training_start
    
    if rank == 0:
        # Calculate Data Utilization Index (DUI)
        total_epochs_completed = len(epoch_times)
        sum_remaining_batches = sum(total_remaining_batches) if total_remaining_batches else 0
        max_possible_batches = max_epochs * total_original_batches
        data_utilization_index = sum_remaining_batches / max_possible_batches if max_possible_batches > 0 else 0
        data_savings_index = 1 - data_utilization_index
        
        # Final statistics
        percent_dropped = total_dropped_batches / total_original_batches * 100 if total_original_batches > 0 else 0
        percent_remaining = 100 - percent_dropped
        total_remaining_batch = total_original_batches - total_dropped_batches
        
        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"\nData Utilization Index (DUI): {data_utilization_index:.4f}")
        print(f"Data Savings Index (DSI): {data_savings_index:.4f}")
        print(f"Sum of remaining batches across all epochs: {sum_remaining_batches}")
        print(f"Maximum possible batches (max_epochs * total_batches): {max_possible_batches}")
        
        print(f"\nFinal Statistics:")
        print(f"Training stopped {'early' if early_stopped else 'normally'} after {total_epochs_completed} epochs")
        print(f"Total batches dropped: {total_dropped_batches}/{total_original_batches} ({percent_dropped:.2f}%)")
        print(f"Total batches remaining: {total_remaining_batch}/{total_original_batches} ({percent_remaining:.2f}%)")
        print(f"Final threshold: {threshold_history[-1] if threshold_history else delta_start:.8f}")
        
        print(f"\nTraining Time Statistics:")
        print(f"Total training time: {total_training_time:.2f} seconds ({total_training_time/60:.2f} minutes)")
        print(f"Average epoch time: {sum(epoch_times)/len(epoch_times) if epoch_times else 0:.2f} seconds")
        print(f"Number of epochs completed: {len(epoch_times)}")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    # Perform final evaluation on test set (always, even if training stopped early)
    if world_size > 1:
        dist.barrier()
    
    test_sampler.set_epoch(0)  # Reset for final evaluation
    
    if rank == 0:
        print("\n" + "="*60)
        print("Final Test Set Evaluation")
        print("="*60)
    
    final_test_loss, final_test_acc, final_targets, final_preds = validate_ddp(
        model, device, test_loader, rank=rank, world_size=world_size
    )
    
    if rank == 0:
        print(f"Test Loss: {final_test_loss:.4f}, Test Accuracy: {final_test_acc:.2f}%")
        
        # Create and save final summary DataFrame (always, even if training stopped early)
        summary_data = {
            'total_training_time': [total_training_time],
            'average_epoch_time': [sum(epoch_times)/len(epoch_times) if epoch_times else 0],
            'epochs_completed': [total_epochs_completed],
            'early_stopped': [early_stopped],
            'best_validation_accuracy': [best_val_acc],
            'final_test_accuracy': [final_test_acc],
            'data_utilization_index': [data_utilization_index],
            'data_savings_index': [data_savings_index],
            'total_batches_dropped': [total_dropped_batches],
            'total_remaining_batches': [total_remaining_batch],
            'batch_drop_percentage': [percent_dropped],
            'final_threshold': [threshold_history[-1] if threshold_history else delta_start],
            'min_batch_percentage': [min_batch_percentage],
            'learning_rate': [learning_rate],
            'weight_decay': [weight_decay],
            'momentum': [momentum],
            'world_size': [world_size],
            'batch_size_per_gpu': [batch_size_per_gpu],
            'total_batch_size': [batch_size_per_gpu * world_size],
            'model': [args.model],
            'delta_start': [delta_start],
            'delta_end': [delta_end]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(f'imagenet_ddp_{args.model}_batch_dropping_summary.csv', index=False)
        print(f"\n✓ Summary saved to imagenet_ddp_{args.model}_batch_dropping_summary.csv")
        
        # Save threshold history for further analysis (handle empty case)
        if threshold_history:
            threshold_df = pd.DataFrame({
                'epoch': range(1, len(threshold_history) + 1),
                'threshold': threshold_history,
                'remaining_batches': total_remaining_batches,
                'batches_dropped': total_batches_dropped_per_epoch,
                'pruned_percentage': pruned_percentage_history
            })
            threshold_df.to_csv(f'imagenet_ddp_{args.model}_threshold_history.csv', index=False)
            print(f"✓ Threshold history saved to imagenet_ddp_{args.model}_threshold_history.csv")
        else:
            print("Note: No threshold history to save (training stopped before epoch 1)")
        
        print(f"\n{'='*60}")
        print(f"ALL RESULTS SAVED SUCCESSFULLY")
        print(f"{'='*60}\n")
    
    # Cleanup distributed training
    if world_size > 1:
        dist.destroy_process_group()
    
    if rank == 0:
        return model, summary_df if 'summary_df' in locals() else None
    else:
        return None, None

if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Run the entire classification process with DDP
    model, summary_df = run_imagenet_classification_ddp(args)