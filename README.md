# Batch Pruning by Activation Stability

Official implementation for the paper *"Batch Pruning by Activation Stability"* (ICLR 2026). [[Paper]](https://openreview.net/pdf?id=TUADW7db5n)


## Abstract

Training deep neural networks remains costly in terms of data, time, and energy, limiting their deployment in large-scale and resource-constrained settings. To address this, we propose Batch Pruning by Activation Stability (*B-PAS*), a dynamic plug-in strategy that accelerates training by adaptively removing batches that contribute less to learning. *B-PAS* monitors the stability of activation feature maps across epochs and prunes batches whose activation variance exhibits minimal change, indicating diminishing learning utility. Applied to ResNet-18, ResNet-50, and the Convolutional Vision Transformer (CvT) on CIFAR-10, CIFAR-100, SVHN, and ImageNet-1K, *B-PAS* reduces training batch usage by up to 57% with no loss in accuracy, and by 47% while slightly improving accuracy. Moreover, it achieves up to 61% savings in GPU node-hours, outperforming prior state-of-the-art pruning methods with up to 29% higher data savings and 21% greater GPU node-hour savings. We further demonstrate the generalization of *B-PAS* by extending it to GPT-2 fine-tuning, showing that activation stability can serve as an effective pruning signal beyond vision models. These results highlight activation stability as a powerful internal signal for efficient training, offering a practical and sustainable path toward data and energy-efficient deep learning.

Training deep neural networks remains costly in terms of data, time, and energy, limiting their deployment in large-scale and resource-constrained settings. To address this, we propose Batch Pruning by Activation Stability (*B-PAS*), a dynamic plug-in strategy that accelerates training by removing batches that contribute less to learning. *B-PAS* monitors the stability of activation representations across epochs and prunes batches whose activation variance exhibits minimal change, indicating diminishing learning utility. Applied to ResNet-18, ResNet-50, and the Convolutional vision Transformer (CvT) on CIFAR-10, CIFAR-100, SVHN, and ImageNet-1K, *B-PAS* reduces training batch usage by up to 57\% with no loss in accuracy, and by 47\% while slightly improving accuracy. Moreover, it achieves up to 61\% savings in GPU node-hours, outperforming prior state-of-the-art pruning methods with up to 29\% higher data savings and 21\% greater GPU node-hour savings. We further demonstrate the generalization of *B-PAS* by extending it to GPT-2 fine-tuning, showing that activation stability can serve as an effective pruning signal beyond vision models. These results highlight activation stability as a powerful internal signal for efficient training, offering a practical and sustainable path toward data and energy-efficient deep learning.

## Repository Structure

```
.
├── bpas_imagenet_resnets.py      # ImageNet training with ResNet-18/50 + B-PAS (DDP)
├── bpas_imagenet_cvt.py          # ImageNet training with CvT-13 + B-PAS (DDP)
├── example_notebook/
│   └── bpas_notebook.ipynb       # Interactive notebook for CIFAR and SVHN experiments
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.9.23

```bash
pip install -r requirements.txt
```

The `requirements.txt` specifies the following dependencies:

```
datasets==4.4.1
einops==0.8.2
matplotlib==3.9.4
numpy==1.23.5
pandas==2.0.3
Pillow==10.4.0
scikit-learn==1.3.2
seaborn==0.13.2
thop==0.1.1-2209072238
torch==2.5.1
torchvision==0.20.1
transformers==4.57.1
```

## Datasets

**CIFAR-10, CIFAR-100, and SVHN** are downloaded automatically by torchvision when running the notebook. No manual setup is required.

**ImageNet (ILSVRC2012)** must be downloaded manually from [https://www.image-net.org/](https://www.image-net.org/). Download the `ILSVRC2012_img_train` archive and extract it so that the directory contains one subfolder per class (standard ImageNet folder structure). The path to this directory is passed via the `--train_dir` argument.

### Data Splits

All experiments split data into training, validation, and test sets as follows:

| Dataset   | Training  | Validation | Test   |
|-----------|-----------|------------|--------|
| CIFAR-10  | 45,000    | 5,000      | 10,000 |
| CIFAR-100  | 45,000    | 5,000      | 10,000 |
| SVHN      | 68,257    | 5,000      | 26,032 |
| ImageNet  | 1,220,909 | 10,000     | 50,258 |

## Usage

### CIFAR-10, CIFAR-100, and SVHN (Notebook)

The interactive notebook at `example_notebook/bpas_notebook.ipynb` contains two cells: one for CIFAR-10 and one for SVHN. Users can experiment freely by adjusting the pruning thresholds (`delta_start`, `delta_end`) and other hyperparameters directly in the notebook.

To run experiments on CIFAR-100, replace the CIFAR-10 dataset with CIFAR-100 in the corresponding cell and update the normalization values to match CIFAR-100 statistics.

### ImageNet with ResNets

```bash
torchrun --nproc_per_node=4 bpas_imagenet_resnets.py \
    --model resnet50 \
    --total_epochs 200 \
    --batch_size_per_gpu 256 \
    --learning_rate 0.1 \
    --delta_start 0.000005 \
    --delta_end 0.00005 \
    --train_dir /path/to/ILSVRC2012_img_train \
    --synset_path /path/to/synset_words.txt
```

To train with ResNet-18, change `--model resnet18`. Adjust `--nproc_per_node` according to the number of available GPUs.

### ImageNet with CvT-13

```bash
torchrun --nproc_per_node=4 bpas_imagenet_cvt.py \
    --model cvt13 \
    --total_epochs 200 \
    --batch_size_per_gpu 128 \
    --learning_rate 1e-3 \
    --weight_decay 0.05 \
    --delta_start 0.00001 \
    --delta_end 0.0005 \
    --train_dir /path/to/ILSVRC2012_img_train \
    --synset_path /path/to/synset_words.txt
```

## Note on Reported Results

The results reported in the paper use validation accuracy across all experiments. The code in this repository has been updated to include test set evaluation as well. With the updated test configuration, test accuracy results are within ±1–2% of the validation accuracy reported in the paper for most configurations. The Data Savings Index (DSI) and GPU node-hour savings percentage also remain in a similar range.

The code reports two complementary metrics: the Data Utilization Index (DUI), which measures the fraction of total training data used during training, and the Data Savings Index (DSI), which measures the fraction of data pruned away. The two are related as DSI = 1 − DUI, and both range between 0 and 1.

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{
  alam2026batch,
  title={Batch Pruning by Activation Stability},
  author={Md Mustakin Alam and Shaker Islam and Aminul Islam},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=TUADW7db5n}
}
```
