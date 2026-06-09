import os
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from revised_dataset_recog_only import CropDataset, train_crop_transform, val_crop_transform
from revised_model_recog_only import PlateRecognizer
from revised_config import NUM_CLASSES, DEVICE, PAD_IDX


DATA_DIR = "data_prepared/dataset_unified_recog"
OUT_DIR = "rfdetr/checkpoints_pure_recog_merge_v2"
BATCH_SIZE = 32
EPOCHS = 35
LR = 3e-4
WEIGHT_DECAY = 1e-4


def decode_batch_indices(indices: torch.Tensor) -> List[str]:
    results = []
    for row in indices:
        chars = [str(int(v.item())) for v in row]  # debug-safe placeholder if needed
        results.append(chars)
    return results


def full_plate_accuracy(pred_idx: torch.Tensor, gt_labels: torch.Tensor, pad_idx: int) -> float:
    matches = []
    for pred_row, gt_row in zip(pred_idx, gt_labels):
        valid_mask = gt_row != pad_idx
        pred_valid = pred_row[valid_mask]
        gt_valid = gt_row[valid_mask]
        matches.append(bool(torch.equal(pred_valid, gt_valid)))
    return (sum(matches) / max(1, len(matches))) * 100.0



def masked_char_accuracy(pred_idx: torch.Tensor, gt_labels: torch.Tensor, pad_idx: int) -> float:
    valid_mask = gt_labels != pad_idx
    correct = ((pred_idx == gt_labels) & valid_mask).sum().item()
    total = valid_mask.sum().item()
    return (correct / max(1, total)) * 100.0



def main():
    train_ds = CropDataset(
        root_dir=os.path.join(DATA_DIR, "images"),
        csv_file=os.path.join(DATA_DIR, "train.csv"),
        transform=train_crop_transform,
    )
    val_ds = CropDataset(
        root_dir=os.path.join(DATA_DIR, "images"),
        csv_file=os.path.join(DATA_DIR, "valid.csv"),
        transform=val_crop_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=(DEVICE == "cuda"),
        persistent_workers=True,
    )

    model = PlateRecognizer(NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.05)

    os.makedirs(OUT_DIR, exist_ok=True)
    best_score = -1.0

    print(f"Training recognizer on {DEVICE} | train={len(train_ds)} val={len(val_ds)}")

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")

        for crop_imgs, gt_labels in loop:
            crop_imgs = crop_imgs.to(DEVICE, non_blocking=True)
            gt_labels = gt_labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_chars = model(crop_imgs)
            loss = criterion(pred_chars.reshape(-1, NUM_CLASSES), gt_labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss = 0.0
        all_char_acc = []
        all_plate_acc = []
        with torch.no_grad():
            for crop_imgs, gt_labels in val_loader:
                crop_imgs = crop_imgs.to(DEVICE, non_blocking=True)
                gt_labels = gt_labels.to(DEVICE, non_blocking=True)

                pred_chars = model(crop_imgs)
                loss = criterion(pred_chars.reshape(-1, NUM_CLASSES), gt_labels.reshape(-1))
                val_loss += loss.item()

                pred_idx = torch.argmax(pred_chars, dim=2)
                all_char_acc.append(masked_char_accuracy(pred_idx, gt_labels, PAD_IDX))
                all_plate_acc.append(full_plate_accuracy(pred_idx, gt_labels, PAD_IDX))

        avg_train_loss = train_loss / max(1, len(train_loader))
        avg_val_loss = val_loss / max(1, len(val_loader))
        avg_char_acc = sum(all_char_acc) / max(1, len(all_char_acc))
        avg_plate_acc = sum(all_plate_acc) / max(1, len(all_plate_acc))
        scheduler.step(avg_val_loss)

        print(
            f"--> Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Char Acc: {avg_char_acc:.2f}% | Val Full Plate Acc: {avg_plate_acc:.2f}%"
        )

        score = avg_plate_acc - avg_val_loss
        if score > best_score:
            best_score = score
            ckpt_path = os.path.join(OUT_DIR, "best_recognizer.pth")
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "epoch": epoch + 1,
                    "val_loss": avg_val_loss,
                    "val_char_acc": avg_char_acc,
                    "val_plate_acc": avg_plate_acc,
                },
                ckpt_path,
            )
            print(f"    [!] Best Model Saved -> {ckpt_path}")


if __name__ == "__main__":
    main()
