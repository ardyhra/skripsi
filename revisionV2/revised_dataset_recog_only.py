import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from revised_config import (
    CHAR2IDX,
    PAD_IDX,
    MAX_PLATE_LEN,
    CROP_HEIGHT,
    CROP_WIDTH,
    sanitize_plate_text,
)


class CropDataset(Dataset):
    def __init__(self, root_dir, csv_file, transform=None, return_text=False):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file, dtype={"label": str})
        self.transform = transform
        self.max_len = MAX_PLATE_LEN
        self.return_text = return_text

        # Bersihkan label kosong / tidak valid sejak awal
        self.df["label"] = self.df["label"].apply(sanitize_plate_text)
        self.df = self.df[self.df["label"].str.len() > 0].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def encode_text(self, text_label: str) -> torch.Tensor:
        label_indices = [CHAR2IDX[c] for c in text_label[: self.max_len] if c in CHAR2IDX]
        padded = label_indices + [PAD_IDX] * (self.max_len - len(label_indices))
        return torch.tensor(padded, dtype=torch.long)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row["filename"]
        text_label = sanitize_plate_text(row["label"])

        img_path = os.path.join(self.root_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            # Fallback aman jika file korup / hilang
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_tensor = self.encode_text(text_label)

        if self.transform:
            image_tensor = self.transform(image=image)["image"]
        else:
            image = cv2.resize(image, (CROP_WIDTH, CROP_HEIGHT))
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.return_text:
            return image_tensor, label_tensor, text_label
        return image_tensor, label_tensor


# Augmentasi dibuat lebih dekat dengan kondisi video lalu lintas.
train_crop_transform = A.Compose([
    A.Resize(CROP_HEIGHT, CROP_WIDTH),
    A.OneOf([
        A.MotionBlur(blur_limit=5, p=1.0),
        A.GaussianBlur(blur_limit=(3, 5), p=1.0),
        A.MedianBlur(blur_limit=3, p=1.0),
    ], p=0.30),
    A.Perspective(scale=(0.02, 0.06), p=0.25),
    A.Affine(scale=(0.95, 1.05), translate_percent=(0.02, 0.04), rotate=(-3, 3), shear=(-2, 2), p=0.25),
    A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=0.40),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.15),
    A.ImageCompression(quality_range=(55, 95), p=0.25),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

val_crop_transform = A.Compose([
    A.Resize(CROP_HEIGHT, CROP_WIDTH),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])
