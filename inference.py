import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

from dataset_seg import BinarySegmentationDataset
from model_seg import UNetWithClassification


def get_transforms():
    return transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def overlay_mask(image, mask, label, prob, class_names=['Benign', 'Cancer']):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    img_display = image.cpu().permute(1, 2, 0).numpy()
    img_display = np.clip(img_display, 0, 1)
    
    mask_display = mask.cpu().squeeze().numpy()
    
    axes[0].imshow(img_display)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(img_display)
    mask_overlay = np.zeros_like(img_display)
    mask_overlay[:, :, 1] = mask_display
    mask_overlay[:, :, 0] = mask_display * 0.3
    axes[1].imshow(mask_overlay, alpha=0.5)
    axes[1].set_title(f'Lesion Segmentation\n(Green = Lesion Area)')
    axes[1].axis('off')
    
    axes[2].imshow(img_display)
    if label == 1:
        highlight = np.zeros_like(img_display)
        highlight[:, :, 0] = mask_display
        axes[2].imshow(highlight, alpha=mask_display * 0.7)
        axes[2].set_title(f'** CANCER DETECTED **\nProbability: {prob:.1%}')
    else:
        axes[2].set_title(f'Benign Lesion\nConfidence: {prob:.1%}')
    axes[2].axis('off')
    
    result_text = f"Classification: {class_names[label]} ({prob:.1%} probability)"
    plt.suptitle(result_text, fontsize=14, fontweight='bold', 
                 color='red' if label == 1 else 'green')
    
    plt.tight_layout()
    plt.savefig('inference_result.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig


def predict_single(model, image_path, device):
    model.eval()
    
    image = Image.open(image_path).convert('RGB')
    original_size = image.size
    
    image_tensor = get_transforms()(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred_mask, pred_class = model(image_tensor)
    
    pred_mask = pred_mask.cpu().squeeze()
    pred_mask_resized = F.interpolate(
        pred_mask.unsqueeze(0).unsqueeze(0),
        size=(original_size[1], original_size[0]),
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu()
    
    pred_class = pred_class.cpu().squeeze()
    pred_prob = torch.softmax(pred_class, dim=0)
    pred_label = pred_class.argmax().item()
    
    return pred_mask_resized, pred_label, pred_prob[pred_label].item(), image, original_size


def evaluate_model(model_path, image_dir, mask_dir, ground_truth_path, num_classes=2, batch_size=8):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = UNetWithClassification(num_classes=num_classes, pretrained=False).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {model_path}")
    print(f"Trained for {checkpoint['epoch']} epochs")
    print(f"Validation IoU: {checkpoint['val_iou']:.4f}")
    print(f"Validation Dice: {checkpoint['val_dice']:.4f}")
    print(f"Validation Cls Acc: {checkpoint['val_cls_acc']:.4f}")
    
    val_idx = checkpoint['val_idx']
    
    dataset = BinarySegmentationDataset(
        image_dir, mask_dir, ground_truth_path,
        transform=get_transforms()
    )
    
    from torch.utils.data import Subset
    val_subset = Subset(dataset, val_idx)
    
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_masks = []
    all_images = []
    
    print("\nRunning evaluation on validation set...")
    with torch.no_grad():
        for batch in val_loader:
            images = batch['image'].to(device)
            labels = batch['is_cancer'].numpy()
            
            pred_masks, pred_classes = model(images)
            
            probs = torch.softmax(pred_classes, dim=1)
            preds = pred_classes.argmax(dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_labels.extend(labels)
            all_probs.extend(probs[:, 1].cpu().numpy())
            all_masks.extend(pred_masks.cpu().numpy())
            all_images.extend(images.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    print(f"\n{'='*50}")
    print("EVALUATION RESULTS")
    print(f"{'='*50}")
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='binary')
    f1_macro = f1_score(all_labels, all_preds, average='macro')
    
    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"F1 Score (Binary): {f1:.4f}")
    print(f"F1 Score (Macro): {f1_macro:.4f}")
    
    print(f"\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Benign', 'Cancer']))
    
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Confusion Matrix:")
    print(cm)
    
    tn, fp, fn, tp = cm.ravel()
    print(f"\nTrue Negatives: {tn}, False Positives: {fp}")
    print(f"False Negatives: {fn}, True Positives: {tp}")
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"\nSensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    
    plt.figure(figsize=(8, 6))
    import seaborn as sns
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Benign', 'Cancer'], 
                yticklabels=['Benign', 'Cancer'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png', dpi=150)
    plt.show()
    
    print(f"\n{'='*50}")
    print("SAMPLE PREDICTIONS")
    print(f"{'='*50}")
    
    n_samples = min(6, len(all_preds))
    indices = np.random.choice(len(all_preds), n_samples, replace=False)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(indices):
        img = all_images[idx]
        mask = all_masks[idx]
        pred = all_preds[idx]
        true = all_labels[idx]
        prob = all_probs[idx]
        
        img_display = np.transpose(img, (1, 2, 0))
        img_display = np.clip(img_display, 0, 1)
        
        mask_display = mask.squeeze()
        
        axes[i].imshow(img_display)
        axes[i].imshow(np.zeros((*mask_display.shape, 3)), alpha=0)
        
        highlight = np.zeros_like(img_display)
        highlight[:, :, 0] = mask_display * 0.8
        highlight[:, :, 1] = mask_display * 0.4
        axes[i].imshow(highlight, alpha=mask_display * 0.6)
        
        color = 'red' if pred == 1 else 'green'
        title = f"Pred: {'Cancer' if pred == 1 else 'Benign'} ({prob:.1%})\n"
        title += f"True: {'Cancer' if true == 1 else 'Benign'}"
        title += " ✓" if pred == true else " ✗"
        
        axes[i].set_title(title, color=color, fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('sample_predictions.png', dpi=150)
    plt.show()
    
    return {
        'accuracy': accuracy,
        'f1_binary': f1,
        'f1_macro': f1_macro,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'confusion_matrix': cm
    }


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Inference for Skin Cancer Model')
    parser.add_argument('--model_path', type=str, default='skin_cancer_model.pth', help='Path to trained model')
    parser.add_argument('--image_dir', type=str, default='images', help='Path to images directory')
    parser.add_argument('--mask_dir', type=str, default='masks', help='Path to masks directory')
    parser.add_argument('--ground_truth', type=str, default='GroundTruth.csv', help='Path to GroundTruth.csv')
    parser.add_argument('--image_path', type=str, default=None, help='Path to single image for prediction')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classification classes')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNetWithClassification(num_classes=args.num_classes, pretrained=False).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded model: {args.model_path}")
    
    if args.image_path:
        print(f"\nPredicting for: {args.image_path}")
        pred_mask, pred_label, pred_prob, image, _ = predict_single(
            model, args.image_path, device
        )
        overlay_mask(
            transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])(image.convert('RGB')),
            pred_mask,
            pred_label,
            pred_prob
        )
        print(f"\nResult: {'CANCER' if pred_label == 1 else 'BENIGN'} ({pred_prob:.1%} probability)")
    else:
        print("\nRunning full evaluation...")
        results = evaluate_model(
            args.model_path,
            args.image_dir,
            args.mask_dir,
            args.ground_truth,
            args.num_classes
        )
        print(f"\nFinal Results saved to:")
        print(f"  - confusion_matrix.png")
        print(f"  - sample_predictions.png")
        print(f"  - inference_result.png")