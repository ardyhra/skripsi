import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from config import CHAR2IDX

# UKURAN STANDAR CROP PLAT (Pipih memanjang)
CROP_HEIGHT = 64
CROP_WIDTH = 256

class CropDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file, dtype={'labels': str})
        self.transform = transform
        self.max_len = 7 

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        text_label = str(row['label']).strip().upper()
        
        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        
        if image is None: 
            return self.__getitem__((idx + 1) % len(self))
            
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Encode Teks
        label_indices = [CHAR2IDX[c] for c in text_label if c in CHAR2IDX]
        if len(label_indices) < self.max_len:
            label_indices += [0] * (self.max_len - len(label_indices))
        else:
            label_indices = label_indices[:self.max_len]
            
        label_tensor = torch.tensor(label_indices, dtype=torch.long)

        # Transform & Resize
        if self.transform:
            image_tensor = self.transform(image=image)['image']
        else:
            # Fallback
            image = cv2.resize(image, (CROP_WIDTH, CROP_HEIGHT))
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image_tensor, label_tensor

# Transformasi Training (Lebih kuat karena fokus di plat saja)
train_crop_transform = A.Compose([
    A.Resize(CROP_HEIGHT, CROP_WIDTH),
    A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.5),
    A.GaussianBlur(p=0.2), # Latih model agar tahan gambar blur (sering terjadi di video)
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_crop_transform = A.Compose([
    A.Resize(CROP_HEIGHT, CROP_WIDTH),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])