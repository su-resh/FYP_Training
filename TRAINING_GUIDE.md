# Skin Cancer Model Training Guide

## Overview

This guide walks you through training a **ResNet50-based U-Net** with **CBAM attention** and **ASPP** for skin cancer classification + lesion segmentation. The training pipeline is optimized for **high-end GPUs** (A100, RTX 4090, etc.).

## Quick Start

### 1. Setup Environment

```bash
# Install PyTorch with CUDA 12.6
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Training

```bash
# Recommended for A100 40GB:
python train_seg.py --batch_size 32 --target_size 512 --epochs 50 --grad_accum 2

# For A100 80GB (larger batch):
python train_seg.py --batch_size 64 --target_size 512 --epochs 50 --grad_accum 1

# For RTX 4090 (24GB):
python train_seg.py --batch_size 16 --target_size 384 --epochs 50 --grad_accum 4

# Maximum quality (A100 80GB, 512px):
python train_seg.py --batch_size 64 --target_size 512 --epochs 80 --grad_accum 2 --lr 2e-5
```

### 3. System Specs Detection

On startup, the script automatically detects and displays your hardware:

```
======================================================================
    SKIN CANCER CLASSIFICATION & SEGMENTATION TRAINING
======================================================================
  SYSTEM SPECS
  GPU:                 NVIDIA RTX 4090
  VRAM:                24.0 GB
  GPU Memory Now:      1024MB / 24576MB (4.2%)
  GPU Compute Cap:     8.9
  GPU SMs:             128
  GPU Driver:          565.90
  CPU:                 13th Gen Intel Core i9-13900K
  CPU Cores:           8 physical / 24 logical
  RAM:                 64.0 GB
  OS:                  Windows 10 (10.0.19045)
  Python:              3.11.5
  PyTorch:             2.10.0
  CUDA Available:      Yes
======================================================================
  GPU CLASS: RTX 4090  | Batch 16 | Res 384 | Effective batch 64
======================================================================
```

The script classifies your GPU by VRAM and suggests optimal batch settings.

### 4. Monitor Training

Watch the **live progress bars** in your terminal:

```
Epoch 5 [Train]: 100%|████████████| 200/200 [02:30<00:00, 1.33it/s, loss=0.2345, iou=0.7234, dice=0.8456, acc=0.9123]
Epoch 5 [Val]:   100%|████████████| 50/50  [00:30<00:00, 1.67it/s, loss=0.2567, iou=0.7012, dice=0.8234, acc=0.8901]
```

After each epoch, a **resource usage summary** is printed:

```
  Train - Loss: 0.2345 | IoU: 0.7234 | Dice: 0.8456 | Acc: 0.9123
  Val   - Loss: 0.2567 | IoU: 0.7012 | Dice: 0.8234 | Acc: 0.8901
  LR: 3.00e-05 | VRAM: 14336MB/24576MB (58%) | Util: 97%  Temp: 72°C
  Time: 4.2min/epoch | Total: 0.35h | ETA: 3.15h
```

**Progress bar metrics explained:**

| Metric | What it measures | Target |
|--------|-----------------|--------|
| `loss` | Combined segmentation + classification loss | Lower is better |
| `iou` | Intersection-over-Union for mask prediction | >0.75 is excellent |
| `dice` | Dice coefficient for mask prediction | >0.85 is excellent |
| `acc` | Binary cancer classification accuracy | >0.92 is excellent |

**Resource usage (printed after each epoch):**

| Field | What it shows |
|-------|---------------|
| `VRAM` | Current GPU memory usage (used/total + %) |
| `Util` | GPU compute utilization % |
| `Temp` | GPU temperature in Celsius |
| `Time/epoch` | Time per epoch in minutes |
| `Total` | Total elapsed training time |
| `ETA` | Estimated remaining time |

### 4. Visualize Progress

Training plots are automatically saved every 5 epochs:

![Training Progress](training_plots/training_history_epoch50.png)

**6-panel plot shows:**
- Combined Loss (train vs val)
- Segmentation IoU
- Segmentation Dice
- Classification Accuracy
- Segmentation Loss (separate)
- Classification Loss (separate)

### 5. Evaluate

```bash
# Full evaluation on held-out test set:
python inference.py --model_path skin_cancer_model.pth

# Evaluate using EMA-smoothed weights (better accuracy):
python inference.py --model_path skin_cancer_model_ema.pth

# Predict a single image:
python inference.py --model_path skin_cancer_model.pth --image_path path/to/image.jpg
```

## What's Happening During Training

### Phase 1: Warm-up (Epochs 1-5)
- Model learns basic features (edges, colors, textures)
- Classification accuracy climbs from ~50% to ~80%
- Segmentation IoU starts emerging from noise

### Phase 2: Rapid Learning (Epochs 5-15)
- Classification accuracy reaches >88%
- Segmentation masks become recognizable
- Cosine LR schedule restarts, preventing plateaus

### Phase 3: Fine-tuning (Epochs 15-30)
- Small improvements in both tasks
- EMA weights (after epoch 15) stabilize predictions
- Model focuses on hard cases

### Phase 4: Convergence (Epochs 30-50)
- Marginal gains in all metrics
- Early stopping may trigger if no improvement for 15 epochs

## How to Know Training is Working

### Real-time indicators in the terminal:

```
# GOOD - healthy training:
Epoch 8 [Train] - Loss: 0.3124, IoU: 0.7123, Acc: 0.8912
Epoch 8 [Val]   - Loss: 0.3345, IoU: 0.6901, Acc: 0.8734
```
- Train loss steadily decreasing
- Val loss following closely (no overfitting)
- IoU and Dice consistently improving

```
# WARNING - potential overfitting:
Epoch 8 [Train] - Loss: 0.1800, IoU: 0.8500, Acc: 0.9600
Epoch 8 [Val]   - Loss: 0.4500, IoU: 0.5500, Acc: 0.7900
```
- Val metrics stagnating while train keeps improving
- System will auto-trigger early stopping after 15 epochs

### Saved outputs:

| File | When Created | What It Shows |
|------|-------------|---------------|
| `training_plots/training_history_epoch*.png` | Every 5 epochs | Learning curves |
| `checkpoints/checkpoint_epoch*.pth` | When model improves | Model snapshots |
| `training_history.json` | End of training | Raw numeric history |
| `skin_cancer_model.pth` | Best model | Final model weights |
| `skin_cancer_model_ema.pth` | Best model | EMA-smoothed weights |

## Configuration Guide

### For High-End GPUs & Time Estimation

**Estimated training time for 50 epochs** (based on 8000 training samples at 512×512):

| GPU | VRAM | Batch | Res | Eff Batch | Min/Epoch | Total 50 Epochs |
|-----|------|-------|-----|-----------|-----------|-----------------|
| A100 80GB | 80 GB | 64 | 512 | 128 | ~2.5 min | **~2 hours** |
| A100 40GB | 40 GB | 32 | 512 | 64 | ~3.0 min | **~2.5 hours** |
| A100 40GB ×2 | 80 GB | 64 | 512 | 128 | ~1.5 min | **~1.25 hours** |
| RTX 4090 | 24 GB | 16 | 512 | 64 | ~5.0 min | **~4 hours** |
| RTX 4090 | 24 GB | 16 | 384 | 64 | ~3.0 min | **~2.5 hours** |
| RTX 3090 | 24 GB | 12 | 384 | 48 | ~3.5 min | **~3 hours** |
| RTX 4080 | 16 GB | 8 | 384 | 32 | ~4.5 min | **~4 hours** |

On startup, the script will auto-classify your GPU and show the recommended config.

### GPU Memory Usage During Training

You can monitor real-time GPU usage in the epoch summary:

```
VRAM: 14336MB/24576MB (58%) | Util: 97% | Temp: 72°C
```

**Normal ranges:**
- **VRAM**: 50-95% is healthy. Near 100% may cause OOM errors.
- **Util**: 90-100% means GPU is fully utilized (good).
- **Temp**: <80°C is normal. >85°C may indicate throttling.

### Command Line Arguments

```
--batch_size     Samples per GPU step (default: 32)
--target_size    Image resolution in pixels (384 or 512, default: 512)
--epochs         Number of training epochs (default: 50)
--lr             Learning rate (default: 3e-5)
--grad_accum     Gradient accumulation steps (default: 2)
--no_ema         Disable Exponential Moving Average
--save_path      Model output path (default: skin_cancer_model.pth)
```

### Effective Batch Size

`Effective Batch Size = batch_size × grad_accum`

Example: `--batch_size 32 --grad_accum 2` = effective batch size of 64.

## Architecture Summary

```
Input (512x512x3)
  │
  ├── ResNet50 Encoder ───→ Classification Head ──→ Cancer/Benign
  │     │
  │     └──→ ASPP (multi-scale features)
  │             │
  │             └──→ Decoder with CBAM Attention
  │                       │
  │                       └──→ Segmentation Mask (512x512x1)
  │
  └── Combo Loss: Focal(0.3) + Dice(0.5) + BCE(0.2)  → segmentation
  └── Label Smoothing CE + Class Weights              → classification
```

- **Encoder**: ResNet50 (ImageNet pretrained, 25.6M params)
- **ASPP**: Multi-scale feature extraction (dilations 6, 12, 18)
- **CBAM**: Channel + Spatial attention in every decoder block
- **EMA**: Exponential Moving Average after epoch 15
- **MixUp**: Augmentation for robustness
- **CosineAnnealingWarmRestarts**: LR schedule with warm restarts

## Output Files

After training completes, you'll have:

```
skin_cancer_model.pth          # Best model weights (regular)
skin_cancer_model_ema.pth      # Best model weights (EMA-smoothed)
training_history.json          # Full training metrics as JSON
training_plots/                # Learning curve images
  ├── training_history_epoch5.png
  ├── training_history_epoch10.png
  └── ...
checkpoints/                   # Model checkpoints at each improvement
  ├── checkpoint_epoch3.pth
  ├── checkpoint_epoch7.pth
  └── ...
```

## Troubleshooting

### "CUDA out of memory"
Reduce batch size or target size:
```bash
python train_seg.py --batch_size 16 --target_size 384 --grad_accum 4
```

### "No improvement" messages
Normal in later epochs. Model needs more time to converge. Early stopping
triggers after 15 epochs without improvement.

### Validation accuracy not improving
Check class balance in the output. The cancer ratio is printed at startup.
If severely imbalanced, the model may need more epochs or adjustments.
