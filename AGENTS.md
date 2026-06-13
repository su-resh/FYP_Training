# Skin Cancer Classification & Segmentation

## Repository Structure
- `images/` - 10,017 JPG skin lesion images
- `masks/` - 10,015 PNG segmentation masks (lesion region)
- `GroundTruth.csv` - One-hot encoded labels (7 classes)

## Classes (from GroundTruth.csv)
MEL (Melanoma), NV (Melanocytic nevus), BCC (Basal cell carcinoma), AKIEC (Actinic keratosis), BKL (Benign keratosis), DF (Dermatofibroma), VASC (Vascular lesion)

For **binary cancer detection**: MEL + BCC + AKIEC = cancer; NV + BKL + DF + VASC = benign

## Model (Enhanced)
U-Net with **ResNet50 encoder** + **CBAM attention** + **ASPP multi-scale**:
1. **Classification**: Determines if lesion is cancerous (binary)
2. **Segmentation**: Highlights cancer area with a mask overlay

## Data Split
- **80% training / 20% testing** (train_test_split with stratify)
- Filtered to images with valid masks before splitting (fixes index mismatch bug)

## Training Configuration (High-End)
- Model: U-Net with ResNet50 encoder + CBAM + ASPP (71.8M params, pretrained ImageNet)
- Image size: 512x512 (can use 384)
- Batch size: 32 (A100 40GB) / 64 (A100 80GB) / 16 (RTX 4090)
- Optimizer: AdamW lr=3e-5, weight_decay=1e-4
- Scheduler: CosineAnnealingWarmRestarts (T_0=5, T_mult=2, eta_min=1e-7)
- Segmentation Loss: 0.3×Focal + 0.5×Dice + 0.2×BCE
- Classification Loss: LabelSmoothCE (smoothing=0.1, class weights [1.0, 3.0])
- Epochs: 50 with early stopping (patience=15)
- Mixed precision: Enabled automatically on GPU
- EMA: Exponential Moving Average (decay=0.999, active after epoch 15)
- Gradient accumulation: 2 steps (effective batch = batch_size * 2)
- MixUp augmentation: alpha=0.2

## Files
1. `dataset_seg.py` - BinarySegmentationDataset (accepts DataFrame), MixupAugmentation
2. `model_seg.py` - ResNet50 U-Net + CBAM + ASPP + EMA + FocalLoss/DiceLoss/ComboSegLoss/LabelSmoothCELoss
3. `train_seg.py` - Full training pipeline, 6-panel live plots, CosineAnnealingWarmRestarts
4. `inference.py` - Evaluation (confusion matrix, F1, sensitivity, specificity) + single image prediction
5. `TRAINING_GUIDE.md` - Detailed training guide with visualization explanations
6. `requirements.txt` - All Python dependencies
7. `setup.bat` - One-click setup script

## How to Use
```bash
# On the GPU machine:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Train (A100 40GB recommended)
python train_seg.py --batch_size 32 --target_size 512 --epochs 50

# Evaluate the trained model
python inference.py --model_path skin_cancer_model.pth

# Use EMA-smoothed weights (often better accuracy)
python inference.py --model_path skin_cancer_model_ema.pth
```

## Workflow
1. Copy entire repo to teacher's GPU machine
2. Run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
3. Run `pip install -r requirements.txt`
4. Train: `python train_seg.py --batch_size 32 --target_size 512 --epochs 50`
5. Copy `skin_cancer_model.pth` (and `skin_cancer_model_ema.pth`) back to this machine
6. Run inference: `python inference.py --model_path skin_cancer_model.pth`