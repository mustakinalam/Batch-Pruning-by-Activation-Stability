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
import numpy as np
import time
import pandas as pd
from torch.optim.lr_scheduler import CosineAnnealingLR
import math
import os
from collections import defaultdict
from einops import rearrange
from einops.layers.torch import Rearrange
import gc
import argparse
from datetime import timedelta


GLOBAL_TRAIN_BATCH_INDICES = None


def parse_args():
    """
    Parse command line arguments for training hyperparameters.
    """
    parser = argparse.ArgumentParser(description='ImageNet Classification with CvT-13 + Batch Pruning (DDP)')

    # Training hyperparameters
    parser.add_argument('--total_epochs', type=int, default=200,
                        help='Total number of training epochs (default: 200)')
    parser.add_argument('--batch_size_per_gpu', type=int, default=128,
                        help='Batch size per GPU (default: 128)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                        help='Base learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='Weight decay (default: 0.05)')

    # Threshold parameters
    parser.add_argument('--delta_start', type=float, default=0.00001,
                        help='Starting threshold for exponential schedule (default: 0.00001)')
    parser.add_argument('--delta_end', type=float, default=0.0005,
                        help='Ending threshold for exponential schedule (default: 0.0005)')

    # Model selection (only cvt13 in this script)
    parser.add_argument('--model', type=str, default='cvt13',
                        choices=['cvt13'],
                        help='Model architecture (default: cvt13)')

    # Data paths
    parser.add_argument('--train_dir', type=str, default='/ILSVRC2012_img_train',
                        help='Path to ImageNet training directory')
    parser.add_argument('--synset_path', type=str, default='/synset_words.txt',
                        help='Path to synset_words.txt file')

    return parser.parse_args()


def configure_distributed_gpu():
    """
    Configure distributed GPU settings for PyTorch DDP.
    """
    os.environ['NCCL_TIMEOUT'] = '7200'
    os.environ['NCCL_BLOCKING_WAIT'] = '1'
    os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'
    os.environ['NCCL_DEBUG'] = 'WARN'

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("Environment variables for distributed training not found")
        rank = 0
        world_size = 1
        local_rank = 0

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if world_size > 1:
        dist.init_process_group(backend="nccl",
                                init_method="env://",
                                world_size=world_size,
                                rank=rank,
                                timeout=timedelta(seconds=7200))
        print(f"Rank {rank}/{world_size} initialized on GPU {local_rank}")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.backends.cuda.enable_flash_sdp(True)
        torch.cuda.set_per_process_memory_fraction(0.95, device)

        if rank == 0:
            print(f"Total GPUs Available: {torch.cuda.device_count()}")
            print(f"Using {world_size} GPUs for distributed training")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"GPU {i}: {props.name}, Memory: {props.total_memory / 1e9:.1f} GB")

    return device, rank, world_size, local_rank


# ---------------------------------------------------------------------------
# Dataset & data-loading helpers  (identical to the ResNet version)
# ---------------------------------------------------------------------------

class ImageNetDataset(Dataset):
    def __init__(self, root_dir, synset_path, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        self.synset_to_class_idx = {}
        self.idx_to_class_name = {}
        idx = 0
        with open(synset_path, 'r') as f:
            for line in f:
                synset, class_name = line.strip().split(' ', 1)
                self.synset_to_class_idx[synset] = idx
                self.idx_to_class_name[idx] = class_name
                idx += 1

        self.num_classes = len(self.synset_to_class_idx)
        print(f"Loaded {self.num_classes} classes from synset file")

        self.image_paths = []
        self.labels = []
        self.synset_indices = defaultdict(list)

        for synset in os.listdir(root_dir):
            synset_dir = os.path.join(root_dir, synset)
            if not os.path.isdir(synset_dir):
                continue
            class_idx = self.synset_to_class_idx.get(synset, -1)
            if class_idx == -1:
                continue
            for img_name in os.listdir(synset_dir):
                if img_name.endswith('.JPEG'):
                    img_path = os.path.join(synset_dir, img_name)
                    self.image_paths.append(img_path)
                    self.labels.append(class_idx)
                    self.synset_indices[class_idx].append(len(self.image_paths) - 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            image = Image.new('RGB', (224, 224))
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


class DistributedBatchSampler:
    """
    Distributed sampler that maintains consistent global batch composition
    across epochs and supports batch pruning.
    """
    def __init__(self, dataset, batch_size_per_gpu, world_size, rank, shuffle_within_batch=True):
        self.dataset = dataset
        self.batch_size_per_gpu = batch_size_per_gpu
        self.world_size = world_size
        self.rank = rank
        self.shuffle_within_batch = shuffle_within_batch
        self.epoch = 0

        self.global_batch_size = batch_size_per_gpu * world_size

        total_size = len(dataset)
        indices = np.arange(total_size)
        np.random.shuffle(indices)

        self.global_batch_indices = [
            indices[i:i + self.global_batch_size].tolist()
            for i in range(0, total_size, self.global_batch_size)
            if i + self.global_batch_size <= total_size
        ]

        global GLOBAL_TRAIN_BATCH_INDICES
        if GLOBAL_TRAIN_BATCH_INDICES is None:
            GLOBAL_TRAIN_BATCH_INDICES = self.global_batch_indices

        self.local_batch_indices = []
        for global_batch in self.global_batch_indices:
            start = rank * batch_size_per_gpu
            end = (rank + 1) * batch_size_per_gpu
            self.local_batch_indices.append(global_batch[start:end])

    def __iter__(self):
        local_batches = [b.copy() for b in self.local_batch_indices]
        for batch in local_batches:
            if self.shuffle_within_batch:
                np.random.shuffle(batch)
            yield from batch

    def __len__(self):
        return sum(len(b) for b in self.local_batch_indices)

    def set_active_batches(self, active_batch_indices):
        if active_batch_indices is None:
            self.local_batch_indices = [
                gb[self.rank * self.batch_size_per_gpu:(self.rank + 1) * self.batch_size_per_gpu]
                for gb in GLOBAL_TRAIN_BATCH_INDICES
            ]
        else:
            filtered = [GLOBAL_TRAIN_BATCH_INDICES[i] for i in active_batch_indices]
            self.local_batch_indices = [
                gb[self.rank * self.batch_size_per_gpu:(self.rank + 1) * self.batch_size_per_gpu]
                for gb in filtered
            ]


def create_three_way_split_indices(dataset, val_size=10000, test_ratio=0.0391):
    train_indices, val_indices, test_indices = [], [], []
    num_classes = len(dataset.synset_indices)
    val_per_class = val_size // num_classes
    remainder = val_size % num_classes

    for class_idx, indices in dataset.synset_indices.items():
        indices_shuffled = indices.copy()
        np.random.shuffle(indices_shuffled)
        current_val_size = val_per_class + (1 if class_idx < remainder else 0)
        val_indices.extend(indices_shuffled[:current_val_size])
        remaining = indices_shuffled[current_val_size:]
        test_split_idx = int(len(remaining) * (1 - test_ratio))
        train_indices.extend(remaining[:test_split_idx])
        test_indices.extend(remaining[test_split_idx:])

    return train_indices, val_indices, test_indices


def load_images_ddp(batch_size_per_gpu, rank, world_size, train_dir, synset_path):
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

    full_dataset = ImageNetDataset(root_dir=train_dir, synset_path=synset_path, transform=None)

    train_indices, val_indices, test_indices = create_three_way_split_indices(
        full_dataset, val_size=10000, test_ratio=0.0391)

    if rank == 0:
        print(f"Split dataset into:")
        print(f"  Training: {len(train_indices)} samples")
        print(f"  Validation: {len(val_indices)} samples")
        print(f"  Test: {len(test_indices)} samples")

    train_subset = Subset(full_dataset, train_indices)
    val_subset = Subset(full_dataset, val_indices)
    test_subset = Subset(full_dataset, test_indices)

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

    sampler = DistributedBatchSampler(
        dataset=train_dataset,
        batch_size_per_gpu=batch_size_per_gpu,
        world_size=world_size,
        rank=rank,
        shuffle_within_batch=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_per_gpu, sampler=sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
        prefetch_factor=2, drop_last=True)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size_per_gpu, sampler=val_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
        prefetch_factor=2, drop_last=True)

    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=True)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size_per_gpu, sampler=test_sampler,
        num_workers=4, pin_memory=True, persistent_workers=True,
        prefetch_factor=2, drop_last=True)

    class_names = [full_dataset.idx_to_class_name[i] for i in range(len(full_dataset.idx_to_class_name))]

    if rank == 0:
        print(f"Created efficient batch setup with {len(GLOBAL_TRAIN_BATCH_INDICES)} global batches")
        print(f"Each batch contains the same data samples across all epochs")
        print(f"Intra-batch shuffling is enabled")

    return train_loader, val_loader, test_loader, class_names, val_sampler, test_sampler, sampler


# ---------------------------------------------------------------------------
#  CvT-13 Architecture
# ---------------------------------------------------------------------------

class ConvEmbed(nn.Module):
    """Convolutional Token Embedding used at the start of each CvT stage."""
    def __init__(self, in_channels, embed_dim, patch_size, stride, padding):
        super().__init__()
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=stride, padding=padding
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)                           # (B, embed_dim, H', W')
        B, C, H, W = x.shape
        x = rearrange(x, 'b c h w -> b (h w) c')  # (B, N, C)
        x = self.norm(x)
        return x, H, W


class DepthwiseSeparableConv(nn.Module):
    """Depth-wise separable convolution used for Q, K, V projections in CvT."""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        x = self.dw(x)
        x = self.bn(x)
        x = self.pw(x)
        return x


class ConvProjection(nn.Module):
    """Convolutional projection for Q, K, V with optional strided convolution for K/V."""
    def __init__(self, dim, num_heads, kernel_size=3, stride=1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        padding = kernel_size // 2
        self.conv = DepthwiseSeparableConv(dim, dim, kernel_size=kernel_size,
                                           stride=stride, padding=padding)

    def forward(self, x, H, W):
        # x: (B, N, C) -> reshape to spatial -> conv -> reshape back
        B, N, C = x.shape
        x = rearrange(x, 'b (h w) c -> b c h w', h=H, w=W)
        x = self.conv(x)
        Hp, Wp = x.shape[2], x.shape[3]
        x = rearrange(x, 'b c h w -> b (h w) c')
        # Split into heads: (B, N', num_heads, head_dim) -> (B, num_heads, N', head_dim)
        x = rearrange(x, 'b n (nh hd) -> b nh n hd', nh=self.num_heads, hd=self.head_dim)
        return x, Hp, Wp


class ConvAttention(nn.Module):
    """
    Multi-head self-attention with convolutional projections (CvT style).
    K and V use strided convolutions to reduce spatial resolution.
    """
    def __init__(self, dim, num_heads, qkv_kernel_size=3, kv_stride=1, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.proj_q = ConvProjection(dim, num_heads, kernel_size=qkv_kernel_size, stride=1)
        self.proj_k = ConvProjection(dim, num_heads, kernel_size=qkv_kernel_size, stride=kv_stride)
        self.proj_v = ConvProjection(dim, num_heads, kernel_size=qkv_kernel_size, stride=kv_stride)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, H, W):
        q, _, _ = self.proj_q(x, H, W)       # (B, heads, Nq, head_dim)
        k, _, _ = self.proj_k(x, H, W)       # (B, heads, Nk, head_dim)
        v, _, _ = self.proj_v(x, H, W)       # (B, heads, Nv, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, heads, Nq, Nk)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v)                                  # (B, heads, Nq, head_dim)
        x = rearrange(x, 'b nh n hd -> b n (nh hd)')   # (B, Nq, dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.drop(self.act(self.fc1(x)))
        x = self.drop(self.fc2(x))
        return x


class CvTBlock(nn.Module):
    """Single CvT transformer block: LayerNorm -> ConvAttention -> LayerNorm -> MLP."""
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_kernel_size=3,
                 kv_stride=1, attn_drop=0., drop=0., drop_path=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ConvAttention(dim, num_heads, qkv_kernel_size=qkv_kernel_size,
                                  kv_stride=kv_stride, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)

        # Stochastic depth (drop path)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth per sample."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor = torch.floor(random_tensor + keep_prob)
        return x / keep_prob * random_tensor


class CvTStage(nn.Module):
    """
    One stage of CvT: ConvEmbed -> N × CvTBlock.
    Returns the token sequence AND the spatial dims so the next stage
    (or the pruning tracker) can reshape back to spatial format.
    """
    def __init__(self, in_channels, embed_dim, depth, num_heads,
                 patch_size, stride, padding, mlp_ratio=4.,
                 qkv_kernel_size=3, kv_stride=2,
                 attn_drop=0., drop=0., drop_path_rates=None):
        super().__init__()
        self.embed = ConvEmbed(in_channels, embed_dim, patch_size, stride, padding)

        if drop_path_rates is None:
            drop_path_rates = [0.] * depth

        self.blocks = nn.ModuleList([
            CvTBlock(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_kernel_size=qkv_kernel_size, kv_stride=kv_stride,
                attn_drop=attn_drop, drop=drop, drop_path=drop_path_rates[i]
            )
            for i in range(depth)
        ])

    def forward(self, x):
        # x: (B, C, H, W)  coming from the previous stage (or input image)
        x, H, W = self.embed(x)           # (B, N, embed_dim)
        for blk in self.blocks:
            x = blk(x, H, W)
        return x, H, W


class CvT13(nn.Module):
    """
    CvT-13  (Convolutional Vision Transformer, 13-block variant).

    Stage  | Depth | Embed/Channels | Heads | Patch | Stride | KV-stride
    -------+-------+----------------+-------+-------+--------+----------
      1    |   1   |       64       |   1   |  7    |   4    |    2
      2    |   2   |      192       |   3   |  3    |   2    |    2
      3    |  10   |      384       |   6   |  3    |   2    |    2

    Stage-wise activations are projected back to spatial format after each
    stage so the batch-pruning mechanism can compute variance on the
    multi-scale feature maps.
    """

    def __init__(self, num_classes=1000, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.1):
        super().__init__()

        depths = [1, 2, 10]
        embed_dims = [64, 192, 384]
        num_heads_list = [1, 3, 6]
        # Stage-wise conv-embed params  (patch_size, stride, padding)
        stage_params = [
            (7, 4, 2),   # Stage 1: 224 -> 56
            (3, 2, 1),   # Stage 2: 56  -> 28
            (3, 2, 1),   # Stage 3: 28  -> 14
        ]

        # For activation tracking / pruning
        self.stage_names = ['stage1', 'stage2', 'stage3']
        # Alias expected by training loop (matches ResNet code path)
        self.layer_names = self.stage_names

        total_depth = sum(depths)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]

        in_channels = 3
        self.stages = nn.ModuleList()
        cur = 0
        for i, (depth, dim, nh) in enumerate(zip(depths, embed_dims, num_heads_list)):
            ps, st, pad = stage_params[i]
            stage = CvTStage(
                in_channels=in_channels,
                embed_dim=dim,
                depth=depth,
                num_heads=nh,
                patch_size=ps,
                stride=st,
                padding=pad,
                mlp_ratio=4.,
                qkv_kernel_size=3,
                kv_stride=2,
                attn_drop=attn_drop_rate,
                drop=drop_rate,
                drop_path_rates=dpr[cur:cur + depth]
            )
            self.stages.append(stage)
            in_channels = dim
            cur += depth

        self.norm = nn.LayerNorm(embed_dims[-1])
        self.head = nn.Linear(embed_dims[-1], num_classes)

        # Layers list for generic iteration (used in activation tracking)
        self.conv_layers = self.stages

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, return_activations=False):
        """
        Forward pass.

        When *return_activations=True* the method also returns a list of
        spatial activation tensors (B, C, H, W) — one per stage — so that
        the batch-pruning code can compute variance on the multi-scale
        feature maps exactly as requested.
        """
        activations = []

        for stage in self.stages:
            tokens, H, W = stage(x)  # tokens: (B, H*W, C)
            # Reshape tokens back to spatial format for:
            #   1. feeding into the next stage's ConvEmbed (needs B, C, H, W)
            #   2. computing activation variance for pruning
            x = rearrange(tokens, 'b (h w) c -> b c h w', h=H, w=W)  # (B, C, H, W)
            if return_activations:
                activations.append(x.detach())

        # Classification head — use global average pooling on the final spatial map
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.norm(x)
        # CLS-token-free: global average pool over token dimension
        x = x.mean(dim=1)  # (B, C)
        x = self.head(x)

        if return_activations:
            return x, activations
        return x


# ---------------------------------------------------------------------------
#  Training / validation / pruning helpers
# ---------------------------------------------------------------------------

def train_epoch_ddp(model, device, train_loader, optimizer, epoch, batch_indices=None,
                    scheduler=None, threshold=0.0001, rank=0, world_size=1, sampler=None):
    """Train one epoch with activation-based batch pruning."""
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    batch_size_per_gpu = train_loader.batch_size

    # Use stage_names via the module handle (DDP wraps model)
    base_model = model.module if hasattr(model, 'module') else model
    std_devs = {name: [] for name in base_model.layer_names}
    batch_indices_list = []

    start_time = time.time()
    total_images_processed = 0
    batch_processing_times = []

    if sampler is not None:
        sampler.set_active_batches(batch_indices)

    global GLOBAL_TRAIN_BATCH_INDICES
    total_logical_batches = len(GLOBAL_TRAIN_BATCH_INDICES)
    active_batch_indices = batch_indices if batch_indices is not None else list(range(total_logical_batches))
    if rank == 0:
        print(f"Using {len(active_batch_indices)} out of {total_logical_batches} global batches")

    if world_size > 1:
        dist.barrier()

    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_indices is not None and batch_idx < len(active_batch_indices):
            logical_batch_idx = active_batch_indices[batch_idx]
        else:
            logical_batch_idx = batch_idx

        batch_start_time = time.time()
        batch_indices_list.append(logical_batch_idx)

        data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)

        # Mixup augmentation
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

        # Track activation std — activations are spatial (B, C, H, W) from each stage
        for layer_idx, act in enumerate(activations):
            std_dev = torch.std(act).item()
            std_devs[base_model.layer_names[layer_idx]].append(std_dev)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        _, pred = output.max(1)
        total += target.size(0)
        correct += pred.eq(target).sum().item()

        batch_end_time = time.time()
        batch_processing_times.append(batch_end_time - batch_start_time)
        total_images_processed += data.size(0)

        if rank == 0 and total_images_processed % (100 * batch_size_per_gpu) == 0:
            print(f'Train Epoch: {epoch} [{total_images_processed}/{len(train_loader) * batch_size_per_gpu * world_size} '
                  f'({100. * (batch_idx + 1) / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}\tAccuracy: {100. * correct / total:.2f}%')

        if (batch_idx + 1) % 50 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # Synchronize metrics
    if world_size > 1:
        torch.cuda.synchronize()
        dist.barrier()

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

    # Synchronize activation statistics across all ranks for consistent pruning
    if world_size > 1 and len(batch_indices_list) > 0:
        for layer_name in base_model.layer_names:
            if layer_name in std_df.columns:
                layer_values = torch.tensor(std_df[layer_name].values, dtype=torch.float32, device=device)
                dist.all_reduce(layer_values, op=dist.ReduceOp.SUM)
                std_df[layer_name] = (layer_values / world_size).cpu().numpy()
        std_df['mean_std'] = std_df[base_model.layer_names].mean(axis=1)

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
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device, non_blocking=True), target.to(device, non_blocking=True)
            output = model(data)
            val_loss += F.cross_entropy(output, target).item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
            if rank == 0:
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

    if world_size > 1:
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

    val_loss /= len(test_loader)
    val_acc = 100. * correct / total

    if rank == 0:
        print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

    return val_loss, val_acc, all_targets, all_preds


def calculate_exponential_threshold(delta_start, delta_end, T, t):
    alpha = math.log(delta_end / delta_start) / T
    return delta_start * math.exp(alpha * t)


# ---------------------------------------------------------------------------
#  Main training loop
# ---------------------------------------------------------------------------

def run_imagenet_classification_ddp(args):
    device, rank, world_size, local_rank = configure_distributed_gpu()

    total_epochs = args.total_epochs
    batch_size_per_gpu = args.batch_size_per_gpu
    learning_rate = args.learning_rate  # No linear scaling — AdamW is used
    weight_decay = args.weight_decay

    delta_start = args.delta_start
    delta_end = args.delta_end
    min_batch_percentage = 0

    train_loader, val_loader, test_loader, class_names, val_sampler, test_sampler, batch_sampler = \
        load_images_ddp(batch_size_per_gpu, rank, world_size, args.train_dir, args.synset_path)

    total_original_batches = len(train_loader)

    if rank == 0:
        print(f"\n{'='*60}")
        print(f"Training Configuration")
        print(f"{'='*60}")
        print(f"Model: CvT-13")
        print(f"Total epochs: {total_epochs}")
        print(f"Batch size per GPU: {batch_size_per_gpu}")
        print(f"Total global batch size: {batch_size_per_gpu * world_size}")
        print(f"Learning rate: {learning_rate}")
        print(f"Weight decay: {weight_decay}")
        print(f"Optimizer: AdamW")
        print(f"Scheduler: CosineAnnealingLR")
        print(f"Delta start: {delta_start}")
        print(f"Delta end: {delta_end}")
        print(f"Total original batches per GPU: {total_original_batches}")
        print(f"{'='*60}\n")

    # Create CvT-13
    model = CvT13(num_classes=1000, drop_rate=0.0, attn_drop_rate=0.0,
                   drop_path_rate=0.1).to(device)

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model Parameters: Total={total_params:,}  Trainable={trainable_params:,}")

    # AdamW optimizer + CosineAnnealing
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs, eta_min=1e-6)

    # History
    train_losses, train_accs, val_losses, val_accs = [], [], [], []
    epoch_times, batch_counts = [], []
    prev_epoch_std_df = None
    batch_indices_to_use = None
    total_dropped_batches = 0
    total_remaining_batches = []
    total_batches_dropped_per_epoch = []
    threshold_history = []
    pruned_percentage_history = []

    total_training_start = time.time()
    best_val_acc = 0
    early_stopped = False

    for epoch in range(1, total_epochs + 1):
        if rank == 0:
            print(f"\n{'='*50}")
            print(f"EPOCH {epoch}/{total_epochs}")
            print(f"{'='*50}")

        current_threshold = calculate_exponential_threshold(delta_start, delta_end, total_epochs, epoch)

        if rank == 0:
            threshold_history.append(current_threshold)
            print(f"Current threshold (δ({epoch})): {current_threshold:.8f}")

        if world_size > 1:
            dist.barrier()

        train_loss, train_acc, current_std_df, used_batch_indices, throughput_metrics = train_epoch_ddp(
            model, device, train_loader, optimizer, epoch,
            batch_indices=batch_indices_to_use,
            scheduler=scheduler,
            threshold=current_threshold,
            rank=rank,
            world_size=world_size,
            sampler=batch_sampler
        )

        if rank == 0:
            epoch_times.append(throughput_metrics['epoch_time'])
            batch_counts.append(throughput_metrics['total_batches'])
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            current_remaining_batches = len(used_batch_indices) if used_batch_indices else total_original_batches
            total_remaining_batches.append(current_remaining_batches)
            pruned_percentage_history.append((1 - current_remaining_batches / total_original_batches) * 100)

        if world_size > 1:
            dist.barrier()

        val_sampler.set_epoch(epoch)
        val_loss, val_acc, all_targets, all_preds = validate_ddp(
            model, device, val_loader, rank=rank, world_size=world_size)

        if rank == 0:
            val_losses.append(val_loss)
            val_accs.append(val_acc)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), f'best_{args.model}_imagenet_ddp.pth')
                print(f"✓ New best validation accuracy: {best_val_acc:.2f}% (saved)")

        # ------ Batch pruning decision (rank 0) ------
        batches_dropped_this_epoch = 0

        if rank == 0 and epoch > 1 and prev_epoch_std_df is not None:
            common_indices = set(prev_epoch_std_df.index).intersection(set(current_std_df.index))
            batches_to_drop = []

            for idx in common_indices:
                prev_mean_std = prev_epoch_std_df.loc[idx, 'mean_std']
                curr_mean_std = current_std_df.loc[idx, 'mean_std']
                if abs(prev_mean_std - curr_mean_std) <= current_threshold:
                    batches_to_drop.append(idx)

            if batches_to_drop:
                print(f"\nDropping {len(batches_to_drop)} batches for the next epoch:")
                print(f"Batch indices: {batches_to_drop[:10]}{'...' if len(batches_to_drop) > 10 else ''}")
                total_dropped_batches += len(batches_to_drop)
                batches_dropped_this_epoch = len(batches_to_drop)
            else:
                print("\nNo batches to drop for the next epoch")

            if batch_indices_to_use is None:
                batch_indices_to_use = [i for i in range(total_original_batches) if i not in batches_to_drop]
            else:
                batch_indices_to_use = [i for i in batch_indices_to_use if i not in batches_to_drop]

            percent_remaining = len(batch_indices_to_use) / total_original_batches * 100
            print(f"\nBatches remaining: {len(batch_indices_to_use)}/{total_original_batches} ({percent_remaining:.2f}%)")
            print(f"Total batches dropped so far: {total_dropped_batches}/{total_original_batches} "
                  f"({total_dropped_batches / total_original_batches * 100:.2f}%)")

            if batch_indices_to_use is not None and len(batch_indices_to_use) <= total_original_batches * min_batch_percentage:
                print(f"\nStopping early — fewer than {min_batch_percentage * 100}% of original batches remain")
                early_stopped = True

        # ------ Broadcast pruning decisions ------
        if world_size > 1:
            early_stop_tensor = torch.tensor(1 if early_stopped else 0, dtype=torch.long, device=device)
            dist.broadcast(early_stop_tensor, src=0)
            if rank != 0:
                early_stopped = bool(early_stop_tensor.item())

            if rank == 0:
                if batch_indices_to_use is not None:
                    batch_indices_tensor = torch.tensor(batch_indices_to_use, dtype=torch.long, device=device)
                    size_tensor = torch.tensor(len(batch_indices_tensor), dtype=torch.long, device=device)
                else:
                    batch_indices_tensor = torch.tensor([-1], dtype=torch.long, device=device)
                    size_tensor = torch.tensor(1, dtype=torch.long, device=device)
            else:
                size_tensor = torch.tensor(0, dtype=torch.long, device=device)
                batch_indices_tensor = torch.tensor([], dtype=torch.long, device=device)

            dist.broadcast(size_tensor, src=0)
            if rank != 0:
                batch_indices_tensor = torch.zeros(size_tensor.item(), dtype=torch.long, device=device)
            dist.broadcast(batch_indices_tensor, src=0)

            if len(batch_indices_tensor) == 1 and batch_indices_tensor[0].item() == -1:
                batch_indices_to_use = None
            else:
                batch_indices_to_use = batch_indices_tensor.cpu().tolist()

        if rank == 0:
            total_batches_dropped_per_epoch.append(batches_dropped_this_epoch)

        prev_epoch_std_df = current_std_df

        if early_stopped:
            if rank == 0:
                print(f"Training stopped early at epoch {epoch}")
            break

        if world_size > 1:
            dist.barrier()

        if epoch % 5 == 0:
            torch.cuda.empty_cache()
            gc.collect()

    # === POST-TRAINING ===
    if world_size > 1:
        dist.barrier()

    best_model_path = f'best_{args.model}_imagenet_ddp.pth'
    if os.path.exists(best_model_path):
        model_to_load = model.module if hasattr(model, 'module') else model
        model_to_load.load_state_dict(torch.load(best_model_path, map_location=device))
        if rank == 0:
            print(f"\nAll ranks loaded best model (val_acc: {best_val_acc:.2f}%)")

    if world_size > 1:
        dist.barrier()

    total_training_time = time.time() - total_training_start

    if rank == 0:
        total_epochs_completed = len(epoch_times)
        sum_remaining = sum(total_remaining_batches) if total_remaining_batches else 0
        max_possible = total_epochs * total_original_batches
        dui = sum_remaining / max_possible if max_possible > 0 else 0
        dsi = 1 - dui
        percent_dropped = total_dropped_batches / total_original_batches * 100 if total_original_batches > 0 else 0

        print(f"\n{'='*60}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"DUI: {dui:.4f}   DSI: {dsi:.4f}")
        print(f"Epochs completed: {total_epochs_completed}")
        print(f"Total batches dropped: {total_dropped_batches}/{total_original_batches} ({percent_dropped:.2f}%)")
        print(f"Total training time: {total_training_time:.2f}s ({total_training_time / 60:.2f}min)")
        print(f"Best validation accuracy: {best_val_acc:.2f}%")

    if world_size > 1:
        dist.barrier()

    test_sampler.set_epoch(0)
    if rank == 0:
        print("\n" + "=" * 60)
        print("Final Test Set Evaluation")
        print("=" * 60)

    final_test_loss, final_test_acc, final_targets, final_preds = validate_ddp(
        model, device, test_loader, rank=rank, world_size=world_size)

    if rank == 0:
        print(f"Test Loss: {final_test_loss:.4f}, Test Accuracy: {final_test_acc:.2f}%")

        total_epochs_completed = len(epoch_times)
        sum_remaining = sum(total_remaining_batches) if total_remaining_batches else 0
        max_possible = total_epochs * total_original_batches
        dui = sum_remaining / max_possible if max_possible > 0 else 0
        dsi = 1 - dui
        percent_dropped = total_dropped_batches / total_original_batches * 100 if total_original_batches > 0 else 0
        total_remaining_batch = total_original_batches - total_dropped_batches

        summary_data = {
            'total_training_time': [total_training_time],
            'average_epoch_time': [sum(epoch_times) / len(epoch_times) if epoch_times else 0],
            'epochs_completed': [total_epochs_completed],
            'early_stopped': [early_stopped],
            'best_validation_accuracy': [best_val_acc],
            'final_test_accuracy': [final_test_acc],
            'data_utilization_index': [dui],
            'data_savings_index': [dsi],
            'total_batches_dropped': [total_dropped_batches],
            'total_remaining_batches': [total_remaining_batch],
            'batch_drop_percentage': [percent_dropped],
            'final_threshold': [threshold_history[-1] if threshold_history else delta_start],
            'min_batch_percentage': [min_batch_percentage],
            'learning_rate': [learning_rate],
            'weight_decay': [weight_decay],
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

        print(f"\n{'='*60}")
        print(f"ALL RESULTS SAVED SUCCESSFULLY")
        print(f"{'='*60}\n")

    if world_size > 1:
        dist.destroy_process_group()

    if rank == 0:
        return model, summary_df if 'summary_df' in locals() else None
    return None, None


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    model, summary_df = run_imagenet_classification_ddp(args)
