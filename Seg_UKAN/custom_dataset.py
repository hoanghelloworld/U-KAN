import os
import cv2
import numpy as np
import torch
import torch.utils.data
from PIL import Image


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, mask_dir=None, transform=None, is_test=False):
        """
        Custom dataset for your data structure
        Args:
            img_dir: Directory containing images (.jpg)
            mask_dir: Directory containing masks (.png), None for test data
            transform: Albumentations transforms
            is_test: Whether this is test data (no masks)
        """
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.is_test = is_test
        
        # Get all image files
        self.img_files = [f for f in os.listdir(img_dir) if f.lower().endswith('.jpg')]
        self.img_files.sort()
        
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img_name = self.img_files[idx]
        img_id = os.path.splitext(img_name)[0]
        
        # Load image
        img_path = os.path.join(self.img_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.is_test:
            # For test data, create dummy mask
            mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
        else:
            # Load mask
            mask_name = img_id + '.png'
            mask_path = os.path.join(self.mask_dir, mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = mask[..., None]  # Add channel dimension
        
        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # Normalize image
        img = img.astype('float32') / 255.0
        img = img.transpose(2, 0, 1)  # HWC to CHW
        
        # Normalize mask
        mask = mask.astype('float32') / 255.0
        mask = mask.transpose(2, 0, 1)  # HWC to CHW
        
        # Ensure binary mask
        if not self.is_test:
            mask[mask > 0.5] = 1.0
            mask[mask <= 0.5] = 0.0
        
        return img, mask, {'img_id': img_id}
