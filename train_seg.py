import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import time

from dataset_seg import BinarySegmentationDataset
from model_seg import UNetWithClassification, DiceBCELoss, count_parameters


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.9, 1.1)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


def compute_iou(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    if union == 0:
        return 1.0
    return (intersection / union).item()


def compute_dice(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    if pred_binary.sum() + target.sum() == 0:
        return 1.0
    return (2. * intersection / (pred_binary.sum() + target.sum())).item()


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, scaler=None):
    model.train()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_cls_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    running_cls_acc = 0.0
    total = 0
    use_amp = scaler is not None
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        labels = batch['is_cancer'].to(device).long()
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_masks, pred_classes = model(images)
            seg_loss = 0.5 * nn.BCELoss()(pred_masks, masks) + 0.5 * DiceBCELoss()(pred_masks, masks)
            cls_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(device))(pred_classes, labels)
            loss = 0.7 * seg_loss + 0.3 * cls_loss
        
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        running_seg_loss += seg_loss.item() * images.size(0)
        running_cls_loss += cls_loss.item() * images.size(0)
        running_iou += compute_iou(pred_masks, masks) * images.size(0)
        running_dice += compute_dice(pred_masks, masks) * images.size(0)
        running_cls_acc += (pred_classes.argmax(1) == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'iou': f'{running_iou/total:.4f}',
            'dice': f'{running_dice/total:.4f}',
            'cls_acc': f'{running_cls_acc/total:.4f}'
        })
    
    return {
        'loss': running_loss / total,
        'seg_loss': running_seg_loss / total,
        'cls_loss': running_cls_loss / total,
        'iou': running_iou / total,
        'dice': running_dice / total,
        'cls_acc': running_cls_acc / total
    }


def validate_epoch(model, dataloader, criterion, device, epoch, scaler=None):
    model.eval()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_cls_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    running_cls_acc = 0.0
    total = 0
    use_amp = scaler is not None
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['is_cancer'].to(device).long()
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_masks, pred_classes = model(images)
                seg_loss = 0.5 * nn.BCELoss()(pred_masks, masks) + 0.5 * DiceBCELoss()(pred_masks, masks)
                cls_loss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 3.0]).to(device))(pred_classes, labels)
                loss = 0.7 * seg_loss + 0.3 * cls_loss
            
            running_loss += loss.item() * images.size(0)
            running_seg_loss += seg_loss.item() * images.size(0)
            running_cls_loss += cls_loss.item() * images.size(0)
            running_iou += compute_iou(pred_masks, masks) * images.size(0)
            running_dice += compute_dice(pred_masks, masks) * images.size(0)
            running_cls_acc += (pred_classes.argmax(1) == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'iou': f'{running_iou/total:.4f}',
                'dice': f'{running_dice/total:.4f}',
                'cls_acc': f'{running_cls_acc/total:.4f}'
            })
    
    return {
        'loss': running_loss / total,
        'seg_loss': running_seg_loss / total,
        'cls_loss': running_cls_loss / total,
        'iou': running_iou / total,
        'dice': running_dice / total,
        'cls_acc': running_cls_acc / total
    }


def plot_history(history, save_path='training_history.png'):
    epochs = range(1, len(history['train_loss']) + 1)
    
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 4, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Combined Loss')
    plt.grid(True)
    
    plt.subplot(1, 4, 2)
    plt.plot(epochs, history['train_iou'], 'b-', label='Train IoU')
    plt.plot(epochs, history['val_iou'], 'r-', label='Val IoU')
    plt.xlabel('Epoch')
    plt.ylabel('IoU')
    plt.legend()
    plt.title('Segmentation IoU')
    plt.grid(True)
    
    plt.subplot(1, 4, 3)
    plt.plot(epochs, history['train_dice'], 'b-', label='Train Dice')
    plt.plot(epochs, history['val_dice'], 'r-', label='Val Dice')
    plt.xlabel('Epoch')
    plt.ylabel('Dice')
    plt.legend()
    plt.title('Segmentation Dice')
    plt.grid(True)
    
    plt.subplot(1, 4, 4)
    plt.plot(epochs, history['train_cls_acc'], 'b-', label='Train Cls Acc')
    plt.plot(epochs, history['val_cls_acc'], 'r-', label='Val Cls Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Classification Accuracy')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()


def train_model(
    image_dir,
    mask_dir,
    ground_truth_path,
    num_classes=2,
    batch_size=16,
    epochs=30,
    lr=5e-5,
    weight_decay=1e-4,
    save_path='skin_cancer_model.pth'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_gpu = device.type == 'cuda'
    
    if is_gpu:
        num_workers = 4
        pin_memory = True
    else:
        num_workers = 0
        pin_memory = False
        if batch_size > 4:
            print(f"Warning: CPU detected, reducing batch_size from {batch_size} to 4")
            batch_size = 4
    
    print(f"{'='*60}")
    print(f"SKIN CANCER CLASSIFICATION & SEGMENTATION TRAINING")
    print(f"{'='*60}")
    print(f"Using device: {device}")
    if is_gpu:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Image size: 384x384")
    print(f"Batch size: {batch_size}")
    print(f"Model: U-Net with ResNet34 encoder")
    print(f"Epochs: {epochs}")
    print(f"{'='*60}")
    
    df = pd.read_csv(ground_truth_path)
    
    cancer_labels = (df[['MEL', 'BCC', 'AKIEC']].sum(axis=1) > 0).astype(int).values
    train_idx, val_idx = train_test_split(
        np.arange(len(df)),
        test_size=0.2,
        stratify=cancer_labels,
        random_state=42
    )
    
    cancer_count_train = cancer_labels[train_idx].sum()
    cancer_count_val = cancer_labels[val_idx].sum()
    print(f"\nDataset split:")
    print(f"  Train samples: {len(train_idx)} (Cancer: {cancer_count_train}, Benign: {len(train_idx) - cancer_count_train})")
    print(f"  Val samples: {len(val_idx)} (Cancer: {cancer_count_val}, Benign: {len(val_idx) - cancer_count_val})")
    
    train_dataset = BinarySegmentationDataset(
        image_dir, mask_dir, ground_truth_path,
        transform=get_transforms(train=True)
    )
    val_dataset = BinarySegmentationDataset(
        image_dir, mask_dir, ground_truth_path,
        transform=get_transforms(train=False)
    )
    
    train_subset = Subset(train_dataset, train_idx)
    val_subset = Subset(val_dataset, val_idx)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    
    print(f"\nLoading model...")
    model = UNetWithClassification(num_classes=num_classes, pretrained=True).to(device)
    print(f"Model parameters: {count_parameters(model):,}")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=7)
    
    scaler = torch.cuda.amp.GradScaler() if is_gpu else None
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': [],
        'train_cls_acc': [], 'val_cls_acc': []
    }
    
    best_val_loss = float('inf')
    best_val_iou = 0.0
    patience_counter = 0
    early_stop_patience = 12
    
    start_time = time.time()
    
    print(f"\nStarting training...")
    print(f"{'='*60}")
    
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        
        print(f"\nEpoch {epoch}/{epochs}")
        print(f"{'-'*40}")
        
        train_metrics = train_epoch(model, train_loader, None, optimizer, device, epoch, scaler)
        val_metrics = validate_epoch(model, val_loader, None, device, epoch, scaler)
        
        scheduler.step(val_metrics['loss'])
        
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['train_cls_acc'].append(train_metrics['cls_acc'])
        history['val_cls_acc'].append(val_metrics['cls_acc'])
        
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        print(f"\nTrain - Loss: {train_metrics['loss']:.4f}, IoU: {train_metrics['iou']:.4f}, Dice: {train_metrics['dice']:.4f}, Cls Acc: {train_metrics['cls_acc']:.4f}")
        print(f"Val   - Loss: {val_metrics['loss']:.4f}, IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}, Cls Acc: {val_metrics['cls_acc']:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"Epoch time: {epoch_time/60:.1f}min, Total: {total_time/3600:.2f}h")
        
        save_flag = False
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_flag = True
        
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            save_flag = True
        
        if save_flag:
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_iou': val_metrics['iou'],
                'val_dice': val_metrics['dice'],
                'val_cls_acc': val_metrics['cls_acc'],
                'train_idx': train_idx,
                'val_idx': val_idx
            }, save_path)
            print(f"*** Saved best model (IoU: {val_metrics['iou']:.4f}, Dice: {val_metrics['dice']:.4f}) ***")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{early_stop_patience}")
        
        if patience_counter >= early_stop_patience:
            print(f"\n{'='*60}")
            print(f"Early stopping triggered after {epoch} epochs")
            break
        
        if epoch % 5 == 0:
            plot_history(history, save_path=f'training_history_epoch{epoch}.png')
    
    print(f"\n{'='*60}")
    print(f"Training completed!")
    print(f"Total time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"Best Val IoU: {best_val_iou:.4f}")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {save_path}")
    print(f"{'='*60}")
    
    plot_history(history)
    
    return model, history


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Skin Cancer Segmentation Model (Big)')
    parser.add_argument('--image_dir', type=str, default='images', help='Path to images directory')
    parser.add_argument('--mask_dir', type=str, default='masks', help='Path to masks directory')
    parser.add_argument('--ground_truth', type=str, default='GroundTruth.csv', help='Path to GroundTruth.csv')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classification classes')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (16 for GPU, auto-reduces to 4 on CPU)')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--save_path', type=str, default='skin_cancer_model.pth', help='Model save path')
    
    args = parser.parse_args()
    
    train_model(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        ground_truth_path=args.ground_truth,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path
    )