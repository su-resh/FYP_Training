import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class BinarySegmentationDataset(Dataset):
    def __init__(self, df, image_dir, mask_dir, transform=None, target_size=(384, 384), is_train=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size
        self.is_train = is_train
        self.cancer_classes = ['MEL', 'BCC', 'AKIEC']

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_id = row['image']

        img_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(img_path).convert('RGB')

        mask_path = os.path.join(self.mask_dir, f"{image_id}_segmentation.png")
        mask = Image.open(mask_path).convert('L')
        mask = mask.resize(self.target_size, Image.NEAREST)

        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(np.array(mask)).float()
        if mask.max() > 1:
            mask = mask / 255.0
        mask = mask.unsqueeze(0)

        cancer_score = row[self.cancer_classes].sum()
        is_cancer = 1.0 if cancer_score > 0 else 0.0

        return {
            'image': image,
            'mask': mask,
            'is_cancer': torch.tensor(is_cancer, dtype=torch.float32),
            'image_id': image_id
        }


class MixupAugmentation:
    def __init__(self, alpha=0.2):
        self.alpha = alpha

    def __call__(self, batch):
        if self.alpha <= 0 or not self._should_apply():
            return batch

        images = batch['image']
        masks = batch['mask']
        labels = batch['is_cancer']

        batch_size = images.size(0)
        index = torch.randperm(batch_size).to(images.device)

        lam = np.random.beta(self.alpha, self.alpha)
        if lam < 0.1 or lam > 0.9:
            lam = 0.5

        mixed_images = lam * images + (1 - lam) * images[index]
        mixed_masks = lam * masks + (1 - lam) * masks[index]
        mixed_labels = lam * labels + (1 - lam) * labels[index]

        return {
            'image': mixed_images,
            'mask': mixed_masks,
            'is_cancer': mixed_labels,
            'image_id': batch['image_id'],
            'lam': lam
        }

    def _should_apply(self):
        return np.random.random() < 0.5
