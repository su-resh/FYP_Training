import os
import re
import platform
import subprocess
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from torchvision import transforms
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import json
import warnings
warnings.filterwarnings('ignore')

from dataset_seg import BinarySegmentationDataset, MixupAugmentation
from model_seg import UNetWithClassification, ComboSegLoss, LabelSmoothCELoss, count_parameters, EMA


def get_gpu_memory_usage():
    if not torch.cuda.is_available():
        return {}
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True, timeout=5
        )
        parts = [p.strip() for p in result.stdout.strip().split(', ')]
        if len(parts) >= 5:
            return {
                'mem_total_mb': int(parts[0]),
                'mem_used_mb': int(parts[1]),
                'mem_free_mb': int(parts[2]),
                'gpu_util_pct': int(parts[3]),
                'gpu_temp_c': int(parts[4])
            }
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    try:
        props = torch.cuda.get_device_properties(0)
        allocated = torch.cuda.memory_allocated(0) / 1024**2
        reserved = torch.cuda.memory_reserved(0) / 1024**2
        return {
            'mem_total_mb': int(props.total_memory / 1024**2),
            'mem_used_mb': int(allocated),
            'mem_free_mb': int(props.total_memory / 1024**2 - allocated),
            'gpu_util_pct': -1,
            'gpu_temp_c': -1
        }
    except:
        return {}


def get_system_specs():
    specs = {}

    try:
        result = subprocess.run(['wmic', 'cpu', 'get', 'name'],
                                capture_output=True, text=True, check=True, timeout=5)
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        specs['cpu'] = lines[-1] if len(lines) > 1 else platform.processor()
    except:
        specs['cpu'] = platform.processor() or 'Unknown'

    specs['cpu_physical_cores'] = os.cpu_count() or 0
    try:
        result = subprocess.run(['wmic', 'cpu', 'get', 'NumberOfCores'],
                                capture_output=True, text=True, check=True, timeout=5)
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        if len(lines) > 1:
            specs['cpu_physical_cores'] = int(lines[-1])
    except:
        pass

    try:
        result = subprocess.run(['wmic', 'computersystem', 'get', 'TotalPhysicalMemory'],
                                capture_output=True, text=True, check=True, timeout=5)
        lines = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
        if len(lines) > 1:
            specs['ram_gb'] = int(lines[-1]) / (1024**3)
    except:
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong),
                           ("dwMemoryLoad", ctypes.c_ulong),
                           ("ullTotalPhys", ctypes.c_ulonglong),
                           ("ullAvailPhys", ctypes.c_ulonglong),
                           ("ullTotalPageFile", ctypes.c_ulonglong),
                           ("ullAvailPageFile", ctypes.c_ulonglong),
                           ("ullTotalVirtual", ctypes.c_ulonglong),
                           ("ullAvailVirtual", ctypes.c_ulonglong),
                           ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            memoryStatus = MEMORYSTATUSEX()
            memoryStatus.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if kernel32.GlobalMemoryStatusEx(ctypes.byref(memoryStatus)):
                specs['ram_gb'] = memoryStatus.ullTotalPhys / (1024**3)
        except:
            try:
                import psutil
                specs['ram_gb'] = psutil.virtual_memory().total / (1024**3)
            except:
                specs['ram_gb'] = 0

    specs['os'] = f'{platform.system()} {platform.release()} ({platform.version()})'
    specs['python'] = platform.python_version()
    specs['torch'] = torch.__version__

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        specs['gpu_name'] = torch.cuda.get_device_name(0)
        specs['gpu_vram_gb'] = props.total_memory / 1024**3
        specs['gpu_sms'] = props.multi_processor_count
        specs['gpu_cc'] = f'{props.major}.{props.minor}'
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=driver_version,pcie.link.gen.current,pcie.link.width.current',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, check=True, timeout=5
            )
            parts = [p.strip() for p in result.stdout.strip().split(', ')]
            if len(parts) >= 3:
                specs['gpu_driver'] = parts[0]
                specs['gpu_pcie_gen'] = parts[1]
                specs['gpu_pcie_width'] = parts[2]
        except:
            pass
    else:
        specs['gpu_name'] = 'None'
        specs['gpu_vram_gb'] = 0

    return specs


def get_transforms(train=True, target_size=384):
    if train:
        return transforms.Compose([
            transforms.Resize((target_size, target_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),
            transforms.RandomAffine(degrees=0, translate=(0.15, 0.15), scale=(0.85, 1.15)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.15), ratio=(0.3, 3.3))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((target_size, target_size)),
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


def train_epoch(model, dataloader, seg_criterion, cls_criterion, optimizer, device, epoch,
                scaler=None, ema=None, mixup_fn=None, grad_accum_steps=1):
    model.train()
    running_loss = 0.0
    running_seg_loss = 0.0
    running_cls_loss = 0.0
    running_iou = 0.0
    running_dice = 0.0
    running_cls_acc = 0.0
    total = 0
    use_amp = scaler is not None

    optimizer.zero_grad()

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", ncols=120)
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device, non_blocking=True)
        masks = batch['mask'].to(device, non_blocking=True)
        labels = batch['is_cancer'].to(device, non_blocking=True).long()

        if mixup_fn and epoch > 2:
            batch = mixup_fn({
                'image': images, 'mask': masks, 'is_cancer': labels.float(),
                'image_id': batch['image_id']
            })
            images = batch['image']
            masks = batch['mask']
            labels = batch['is_cancer'].long()

        with torch.cuda.amp.autocast(enabled=use_amp):
            pred_masks, pred_classes = model(images)
            seg_loss = seg_criterion(pred_masks, masks)
            cls_loss = cls_criterion(pred_classes, labels)
            loss = 0.7 * seg_loss + 0.3 * cls_loss
            loss = loss / grad_accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

            if ema is not None:
                ema.update()

        with torch.no_grad():
            running_loss += loss.item() * grad_accum_steps * images.size(0)
            running_seg_loss += seg_loss.item() * images.size(0)
            running_cls_loss += cls_loss.item() * images.size(0)
            running_iou += compute_iou(pred_masks, masks) * images.size(0)
            running_dice += compute_dice(pred_masks, masks) * images.size(0)
            running_cls_acc += (pred_classes.argmax(1) == labels).sum().item()
            total += labels.size(0)

        pbar.set_postfix({
            'loss': f'{running_loss/max(total,1):.4f}',
            'iou': f'{running_iou/max(total,1):.4f}',
            'dice': f'{running_dice/max(total,1):.4f}',
            'acc': f'{running_cls_acc/max(total,1):.4f}'
        })

    if (batch_idx + 1) % grad_accum_steps != 0:
        if scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        optimizer.zero_grad()
        if ema is not None:
            ema.update()

    return {
        'loss': running_loss / total,
        'seg_loss': running_seg_loss / total,
        'cls_loss': running_cls_loss / total,
        'iou': running_iou / total,
        'dice': running_dice / total,
        'cls_acc': running_cls_acc / total
    }


def validate_epoch(model, dataloader, seg_criterion, cls_criterion, device, epoch, scaler=None):
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
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", ncols=120)
        for batch in pbar:
            images = batch['image'].to(device, non_blocking=True)
            masks = batch['mask'].to(device, non_blocking=True)
            labels = batch['is_cancer'].to(device, non_blocking=True).long()

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred_masks, pred_classes = model(images)
                seg_loss = seg_criterion(pred_masks, masks)
                cls_loss = cls_criterion(pred_classes, labels)
                loss = 0.7 * seg_loss + 0.3 * cls_loss

            running_loss += loss.item() * images.size(0)
            running_seg_loss += seg_loss.item() * images.size(0)
            running_cls_loss += cls_loss.item() * images.size(0)
            running_iou += compute_iou(pred_masks, masks) * images.size(0)
            running_dice += compute_dice(pred_masks, masks) * images.size(0)
            running_cls_acc += (pred_classes.argmax(1) == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{running_loss/total:.4f}',
                'iou': f'{running_iou/total:.4f}',
                'dice': f'{running_dice/total:.4f}',
                'acc': f'{running_cls_acc/total:.4f}'
            })

    return {
        'loss': running_loss / total,
        'seg_loss': running_seg_loss / total,
        'cls_loss': running_cls_loss / total,
        'iou': running_iou / total,
        'dice': running_dice / total,
        'cls_acc': running_cls_acc / total
    }


def plot_training_progress(history, save_dir='training_plots'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train')
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', linewidth=2, label='Val')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Combined Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['train_iou'], 'b-', linewidth=2, label='Train')
    axes[0, 1].plot(epochs, history['val_iou'], 'r-', linewidth=2, label='Val')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('IoU')
    axes[0, 1].set_title('Segmentation IoU')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, history['train_dice'], 'b-', linewidth=2, label='Train')
    axes[0, 2].plot(epochs, history['val_dice'], 'r-', linewidth=2, label='Val')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Dice')
    axes[0, 2].set_title('Segmentation Dice')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['train_cls_acc'], 'b-', linewidth=2, label='Train')
    axes[1, 0].plot(epochs, history['val_cls_acc'], 'r-', linewidth=2, label='Val')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Classification Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(epochs, history['train_seg_loss'], 'b-', linewidth=2, label='Train')
    axes[1, 1].plot(epochs, history['val_seg_loss'], 'r-', linewidth=2, label='Val')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Segmentation Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(epochs, history['train_cls_loss'], 'b-', linewidth=2, label='Train')
    axes[1, 2].plot(epochs, history['val_cls_loss'], 'r-', linewidth=2, label='Val')
    axes[1, 2].set_xlabel('Epoch')
    axes[1, 2].set_ylabel('Loss')
    axes[1, 2].set_title('Classification Loss')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.suptitle(f'Training Progress (Best Val IoU: {max(history["val_iou"]):.4f}, Best Val Acc: {max(history["val_cls_acc"]):.4f})',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'training_history_epoch{len(epochs)}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return save_path


def train_model(
    image_dir,
    mask_dir,
    ground_truth_path,
    num_classes=2,
    batch_size=32,
    epochs=50,
    lr=3e-5,
    weight_decay=1e-4,
    save_path='skin_cancer_model.pth',
    target_size=512,
    grad_accum_steps=2,
    use_ema=True,
    ema_decay=0.999
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    is_gpu = device.type == 'cuda'

    if not is_gpu:
        print("ERROR: This enhanced training requires a GPU.")
        print("Falling back to basic training settings...")
        target_size = 384
        batch_size = 4
        grad_accum_steps = 1
        use_ema = False

    num_workers = min(8, os.cpu_count() or 4)
    if is_gpu:
        torch.backends.cudnn.benchmark = True

    specs = get_system_specs()
    gpu_mem = get_gpu_memory_usage()

    print(f"{'='*70}")
    print(f"    SKIN CANCER CLASSIFICATION & SEGMENTATION TRAINING")
    print(f"{'='*70}")
    print(f"  SYSTEM SPECS")
    print(f"  GPU:                 {specs.get('gpu_name', 'N/A')}")
    print(f"  VRAM:                {specs.get('gpu_vram_gb', 0):.1f} GB")
    if gpu_mem:
        pct = gpu_mem['mem_used_mb'] / gpu_mem['mem_total_mb'] * 100
        print(f"  GPU Memory Now:      {gpu_mem['mem_used_mb']}MB / {gpu_mem['mem_total_mb']}MB ({pct:.1f}%)")
    print(f"  GPU Compute Cap:     {specs.get('gpu_cc', 'N/A')}")
    print(f"  GPU SMs:             {specs.get('gpu_sms', 'N/A')}")
    if 'gpu_driver' in specs:
        print(f"  GPU Driver:          {specs['gpu_driver']}")
    print(f"  CPU:                 {specs.get('cpu', 'N/A')}")
    print(f"  CPU Cores:           {specs.get('cpu_physical_cores', 'N/A')} physical / {os.cpu_count()} logical")
    print(f"  RAM:                 {specs.get('ram_gb', 0):.1f} GB")
    print(f"  OS:                  {specs.get('os', 'N/A')}")
    print(f"  Python:              {specs.get('python', 'N/A')}")
    print(f"  PyTorch:             {specs.get('torch', 'N/A')}")
    print(f"  CUDA Available:      {'Yes' if is_gpu else 'No'}")
    print(f"{'='*70}")
    print(f"  TRAINING CONFIG")
    print(f"  Image size:          {target_size}x{target_size}")
    print(f"  Batch size:          {batch_size}")
    print(f"  Gradient accum:      {grad_accum_steps}")
    print(f"  Effective batch:     {batch_size * grad_accum_steps}")
    print(f"  Learning rate:       {lr}")
    print(f"  Epochs:              {epochs}")
    print(f"  Workers:             {num_workers}")
    print(f"  EMA:                 {'Yes (decay=' + str(ema_decay) + ')' if use_ema else 'No'}")
    print(f"  Model:               U-Net with ResNet50 + CBAM + ASPP")

    vram_gb = specs.get('gpu_vram_gb', 0)
    print(f"{'='*70}")
    if vram_gb >= 72:
        print(f"  GPU CLASS: A100 80GB | Batch 64 | Res 512 | Effective batch 128")
    elif vram_gb >= 36:
        print(f"  GPU CLASS: A100 40GB | Batch 32 | Res 512 | Effective batch 64")
    elif vram_gb >= 20:
        print(f"  GPU CLASS: RTX 4090  | Batch 16 | Res 384 | Effective batch 64")
    elif vram_gb >= 14:
        print(f"  GPU CLASS: RTX 3090  | Batch 12 | Res 384 | Effective batch 48")
    else:
        print(f"  GPU CLASS: Unknown   | Batch {batch_size} | Res {target_size}")
    print(f"{'='*70}")

    print("\n[1/5] Loading and filtering dataset...")
    df = pd.read_csv(ground_truth_path)
    valid_indices = []
    for idx in range(len(df)):
        image_id = df.iloc[idx]['image']
        if os.path.exists(os.path.join(mask_dir, f"{image_id}_segmentation.png")):
            valid_indices.append(idx)
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)
    print(f"  Total samples: {len(df)} -> Filtered with masks: {len(df_filtered)}")

    print("\n[2/5] Splitting dataset...")
    cancer_labels = (df_filtered[['MEL', 'BCC', 'AKIEC']].sum(axis=1) > 0).astype(int).values
    train_idx, val_idx = train_test_split(
        np.arange(len(df_filtered)),
        test_size=0.2,
        stratify=cancer_labels,
        random_state=42
    )

    train_count = len(train_idx)
    val_count = len(val_idx)
    train_cancer = cancer_labels[train_idx].sum()
    val_cancer = cancer_labels[val_idx].sum()

    print(f"  Train: {train_count} (Cancer: {train_cancer}, Benign: {train_count - train_cancer})")
    print(f"  Val:   {val_count} (Cancer: {val_cancer}, Benign: {val_count - val_cancer})")
    print(f"  Cancer ratio - Train: {train_cancer/train_count:.1%}, Val: {val_cancer/val_count:.1%}")

    print("\n[3/5] Creating datasets and dataloaders...")
    df_train = df_filtered.iloc[train_idx].reset_index(drop=True)
    df_val = df_filtered.iloc[val_idx].reset_index(drop=True)

    train_dataset = BinarySegmentationDataset(
        df_train, image_dir, mask_dir,
        transform=get_transforms(train=True, target_size=target_size),
        target_size=(target_size, target_size),
        is_train=True
    )
    val_dataset = BinarySegmentationDataset(
        df_val, image_dir, mask_dir,
        transform=get_transforms(train=False, target_size=target_size),
        target_size=(target_size, target_size),
        is_train=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=True if num_workers > 0 else False
    )

    print(f"  Train batches/epoch: {len(train_loader)}")
    print(f"  Val batches/epoch:   {len(val_loader)}")

    print("\n[4/5] Building model...")
    model = UNetWithClassification(num_classes=num_classes, pretrained=True).to(device)
    total_params = count_parameters(model)
    print(f"  Model parameters: {total_params:,}")
    if is_gpu:
        mem_usage = total_params * 4 / 1024**3
        print(f"  Approx model memory: {mem_usage:.2f} GB")

    cls_weight = torch.tensor([1.0, 3.0]).to(device)
    seg_criterion = ComboSegLoss(focal_weight=0.3, dice_weight=0.5, bce_weight=0.2)
    cls_criterion = LabelSmoothCELoss(smoothing=0.1, weight=cls_weight)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    T_0 = 5
    T_mult = 2
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=T_0, T_mult=T_mult, eta_min=1e-7)

    scaler = torch.cuda.amp.GradScaler() if is_gpu else None
    ema = EMA(model, decay=ema_decay) if use_ema else None
    mixup_fn = MixupAugmentation(alpha=0.2) if target_size >= 384 else None

    history = {
        'train_loss': [], 'val_loss': [],
        'train_seg_loss': [], 'val_seg_loss': [],
        'train_cls_loss': [], 'val_cls_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_dice': [], 'val_dice': [],
        'train_cls_acc': [], 'val_cls_acc': [],
        'lr': []
    }

    best_val_loss = float('inf')
    best_val_iou = 0.0
    best_val_acc = 0.0
    patience_counter = 0
    early_stop_patience = 15

    os.makedirs('checkpoints', exist_ok=True)
    start_time = time.time()

    print(f"\n[5/5] Starting training ({epochs} epochs)...")
    print(f"{'='*70}")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        current_lr = optimizer.param_groups[0]['lr']

        if epoch > 15 and use_ema:
            ema.apply_shadow()
            train_metrics = train_epoch(
                model, train_loader, seg_criterion, cls_criterion,
                optimizer, device, epoch, scaler, ema, mixup_fn, grad_accum_steps
            )
            ema.restore()

            val_metrics = validate_epoch(
                model, val_loader, seg_criterion, cls_criterion, device, epoch, scaler
            )
        else:
            train_metrics = train_epoch(
                model, train_loader, seg_criterion, cls_criterion,
                optimizer, device, epoch, scaler, ema, mixup_fn, grad_accum_steps
            )
            val_metrics = validate_epoch(
                model, val_loader, seg_criterion, cls_criterion, device, epoch, scaler
            )

        scheduler.step(epoch - 1)

        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        history['train_seg_loss'].append(train_metrics['seg_loss'])
        history['val_seg_loss'].append(val_metrics['seg_loss'])
        history['train_cls_loss'].append(train_metrics['cls_loss'])
        history['val_cls_loss'].append(val_metrics['cls_loss'])
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])
        history['train_cls_acc'].append(train_metrics['cls_acc'])
        history['val_cls_acc'].append(val_metrics['cls_acc'])
        history['lr'].append(current_lr)

        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        remaining_epochs = epochs - epoch
        eta = (total_time / epoch) * remaining_epochs if epoch > 0 else 0

        gpu_mem = {}
        vram_pct = 0
        if is_gpu:
            gpu_mem = get_gpu_memory_usage()
            if gpu_mem and gpu_mem['mem_total_mb'] > 0:
                vram_pct = gpu_mem['mem_used_mb'] / gpu_mem['mem_total_mb'] * 100

        print()
        print(f"  Train - Loss: {train_metrics['loss']:.4f} | IoU: {train_metrics['iou']:.4f} | Dice: {train_metrics['dice']:.4f} | Acc: {train_metrics['cls_acc']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f} | IoU: {val_metrics['iou']:.4f} | Dice: {val_metrics['dice']:.4f} | Acc: {val_metrics['cls_acc']:.4f}")
        mem_line = f"VRAM: {gpu_mem.get('mem_used_mb', 0)}MB/{gpu_mem.get('mem_total_mb', 0)}MB ({vram_pct:.0f}%)" if gpu_mem else ""
        util_line = f"Util: {gpu_mem.get('gpu_util_pct', -1)}%  Temp: {gpu_mem.get('gpu_temp_c', -1)}°C" if gpu_mem and gpu_mem.get('gpu_util_pct', -1) >= 0 else ""
        print(f"  LR: {current_lr:.2e} | {mem_line} | {util_line}")
        print(f"  Time: {epoch_time/60:.1f}min/epoch | Total: {total_time/3600:.2f}h | ETA: {eta/3600:.1f}h")

        save_flag = False
        reason = []
        if val_metrics['iou'] > best_val_iou:
            best_val_iou = val_metrics['iou']
            save_flag = True
            reason.append(f'IoU={best_val_iou:.4f}')
        if val_metrics['cls_acc'] > best_val_acc:
            best_val_acc = val_metrics['cls_acc']
            save_flag = True
            reason.append(f'Acc={best_val_acc:.4f}')
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_flag = True
            reason.append(f'Loss={best_val_loss:.4f}')

        if save_flag:
            patience_counter = 0
            save_dict = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_metrics['loss'],
                'val_iou': val_metrics['iou'],
                'val_dice': val_metrics['dice'],
                'val_cls_acc': val_metrics['cls_acc'],
                'train_idx': train_idx,
                'val_idx': val_idx,
                'history': history,
                'config': {
                    'target_size': target_size,
                    'batch_size': batch_size,
                    'grad_accum_steps': grad_accum_steps,
                    'lr': lr,
                    'epochs': epochs,
                }
            }
            if ema is not None:
                ema.apply_shadow()
                save_dict['ema_state_dict'] = model.state_dict()
                ema.restore()
                torch.save(save_dict, save_path)
                ema.apply_shadow()
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'ema': True,
                    'config': save_dict['config'],
                    'history': history
                }, save_path.replace('.pth', '_ema.pth'))
                ema.restore()
            else:
                torch.save(save_dict, save_path)

            torch.save(save_dict, os.path.join('checkpoints', f'checkpoint_epoch{epoch}.pth'))
            print(f"  >>> SAVED best model ({', '.join(reason)}) <<<")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{early_stop_patience})")

        if epoch % 5 == 0 or epoch == 1:
            plot_path = plot_training_progress(history)
            print(f"  Training plot saved: {plot_path}")

        if patience_counter >= early_stop_patience:
            print(f"\n{'!'*70}")
            print(f"  EARLY STOPPING triggered after {epoch} epochs")
            print(f"{'!'*70}")
            break

        print(f"{'='*70}")

    print(f"\n{'='*70}")
    print(f"  TRAINING COMPLETED!")
    print(f"  Total time: {(time.time() - start_time)/3600:.2f} hours")
    print(f"  Best Val IoU:      {best_val_iou:.4f}")
    print(f"  Best Val Acc:      {best_val_acc:.4f}")
    print(f"  Best Val Loss:     {best_val_loss:.4f}")
    print(f"  Model saved to:    {save_path}")
    if use_ema:
        print(f"  EMA model saved:   {save_path.replace('.pth', '_ema.pth')}")
    print(f"{'='*70}")

    plot_training_progress(history)

    with open('training_history.json', 'w') as f:
        serializable = {k: [float(v) if isinstance(v, np.floating) else v for v in vals]
                       for k, vals in history.items()}
        json.dump(serializable, f, indent=2)
    print("  Training history saved: training_history.json")

    return model, history


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train Skin Cancer Model (High-End)')
    parser.add_argument('--image_dir', type=str, default='images')
    parser.add_argument('--mask_dir', type=str, default='masks')
    parser.add_argument('--ground_truth', type=str, default='GroundTruth.csv')
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (32 for A100 40GB, 64 for A100 80GB)')
    parser.add_argument('--target_size', type=int, default=512,
                        help='Image resolution (384 or 512)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-5)
    parser.add_argument('--grad_accum', type=int, default=2,
                        help='Gradient accumulation steps')
    parser.add_argument('--save_path', type=str, default='skin_cancer_model.pth')
    parser.add_argument('--no_ema', action='store_true',
                        help='Disable EMA')

    args = parser.parse_args()

    train_model(
        image_dir=args.image_dir,
        mask_dir=args.mask_dir,
        ground_truth_path=args.ground_truth,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        save_path=args.save_path,
        target_size=args.target_size,
        grad_accum_steps=args.grad_accum,
        use_ema=not args.no_ema
    )
