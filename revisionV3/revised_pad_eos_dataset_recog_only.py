import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from revised_pad_eos_config import encode_plate_text, sanitize_plate_text

CROP_HEIGHT = 64
CROP_WIDTH = 256


class CropDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file, dtype={"label": str})
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = str(row["filename"])
        text_label = sanitize_plate_text(row["label"])

        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_tensor = torch.tensor(encode_plate_text(text_label), dtype=torch.long)

        if self.transform:
            image_tensor = self.transform(image=image)["image"]
        else:
            image = cv2.resize(image, (CROP_WIDTH, CROP_HEIGHT))
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image_tensor, label_tensor


train_crop_transform = A.Compose([
    A.Resize(CROP_HEIGHT, CROP_WIDTH),
    A.OneOf([
        A.MotionBlur(blur_limit=5, p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
    ], p=0.35),
    A.Perspective(scale=(0.02, 0.06), keep_size=True, p=0.25),
    A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), rotate=(-2, 2), shear=(-3, 3), p=0.25),
    A.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.05, p=0.35),
    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.25),
    A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(8, 8), p=0.15),
    A.ImageCompression(quality_range=(60, 100), p=0.25),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_crop_transform = A.Compose([
    A.Resize(CROP_HEIGHT, CROP_WIDTH),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
