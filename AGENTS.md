# Skin Cancer Classification & Segmentation

## Repository Structure
- `images/` - 10,017 JPG skin lesion images
- `masks/` - 10,015 PNG segmentation masks (lesion region)
- `GroundTruth.csv` - One-hot encoded labels (7 classes)

## Classes (from GroundTruth.csv)
MEL (Melanoma), NV (Melanocytic nevus), BCC (Basal cell carcinoma), AKIEC (Actinic keratosis), BKL (Benign keratosis), DF (Dermatofibroma), VASC (Vascular lesion)

For **binary cancer detection**: MEL + BCC + AKIEC = cancer; NV + BKL + DF + VASC = benign

## Model
U-Net with ResNet34 encoder that performs:
1. **Classification**: Determines if lesion is cancerous (binary)
2. **Segmentation**: Highlights cancer area with a mask overlay

## Data Split
- **80% training / 20% testing** (train_test_split with stratify)
- Split on image IDs from GroundTruth.csv before loading images

## Training Configuration
- Model: U-Net with ResNet34 encoder (pretrained, ImageNet)
- Image size: 384x384
- Batch size: 16 (GPU) / 4 (CPU) - auto-detected
- Optimizer: AdamW lr=5e-5, weight_decay=1e-4
- Scheduler: ReduceLROnPlateau(mode='min', factor=0.5, patience=7)
- Loss: 0.5×BCE + 0.5×Dice (segmentation) + CrossEntropyLoss (classification)
- Epochs: 30 with early stopping (patience=12)
- Mixed precision: Enabled automatically on GPU

## Files
1. `dataset_seg.py` - BinarySegmentationDataset for image/mask pairs
2. `model_seg.py` - U-Net with ResNet34 encoder + classification head + DiceBCELoss
3. `train_seg.py` - Training with combined segmentation + classification loss, auto GPU/CPU
4. `inference.py` - Evaluation with confusion matrix, F1, sensitivity, specificity + single image prediction
5. `requirements.txt` - All Python dependencies
6. `setup.bat` - One-click setup script

## How to Use
```bash
# On the GPU machine:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# Train (GPU auto-detected)
python train_seg.py --batch_size 16 --epochs 30

# Evaluate the trained model
python inference.py --model_path skin_cancer_model.pth
```

## Workflow
1. Copy entire repo to teacher's GPU machine
2. Run `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126`
3. Run `pip install -r requirements.txt`
4. Train: `python train_seg.py --batch_size 16 --epochs 30`
5. Copy `skin_cancer_model.pth` back to this machine
6. Run inference: `python inference.py --model_path skin_cancer_model.pth`