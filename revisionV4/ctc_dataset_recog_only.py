import os
from typing import List, Tuple

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from ctc_config import CROP_HEIGHT, CROP_WIDTH, encode_plate_text, letterbox_plate_image, sanitize_plate_text


_post_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

train_aug = A.Compose([
    A.RandomBrightnessContrast(p=0.35),
    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.15),
    A.OneOf([
        A.GaussianBlur(blur_limit=(3, 5)),
        A.MotionBlur(blur_limit=(3, 7)),
        A.MedianBlur(blur_limit=3),
    ], p=0.25),
    A.Perspective(scale=(0.02, 0.06), p=0.15),
    A.Affine(scale=(0.95, 1.05), translate_percent=(-0.03, 0.03), rotate=(-3, 3), shear=(-5, 5), p=0.2),
    A.ImageCompression(quality_range=(55, 95), p=0.2),
    A.GaussNoise(std_range=(0.01, 0.03), p=0.12),
])

val_aug = A.Compose([])


class CTCCropDataset(Dataset):
    def __init__(self, root_dir: str, csv_file: str, augment: A.Compose | None = None):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file, dtype={"label": str})
        self.augment = augment or val_aug

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int, str, str]:
        row = self.df.iloc[idx]
        img_name = str(row["filename"])
        text_label = sanitize_plate_text(row["label"])
        img_path = os.path.join(self.root_dir, img_name)

        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((idx + 1) % len(self))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.augment is not None:
            image = self.augment(image=image)["image"]

        image = letterbox_plate_image(image, CROP_HEIGHT, CROP_WIDTH)
        tensor_img = _post_transform(image=image)["image"]

        target_ids: List[int] = encode_plate_text(text_label)
        if len(target_ids) == 0:
            return self.__getitem__((idx + 1) % len(self))

        return tensor_img, torch.tensor(target_ids, dtype=torch.long), len(target_ids), text_label, img_name


def ctc_collate_fn(batch):
    images, targets, target_lengths, texts, filenames = zip(*batch)
    images = torch.stack(images, dim=0)
    flat_targets = torch.cat(targets, dim=0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    return images, flat_targets, target_lengths, list(texts), list(filenames)
