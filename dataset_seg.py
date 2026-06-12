import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class BinarySegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, ground_truth_path, transform=None, target_size=(384, 384)):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.target_size = target_size

        self.df = pd.read_csv(ground_truth_path)
        self.cancer_classes = ['MEL', 'BCC', 'AKIEC']

        self.valid_indices = []
        for idx in range(len(self.df)):
            image_id = self.df.iloc[idx]['image']
            mask_path = os.path.join(self.mask_dir, f"{image_id}_segmentation.png")
            if os.path.exists(mask_path):
                self.valid_indices.append(idx)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.df.iloc[actual_idx]
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
