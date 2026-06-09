import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from revised_pad_eos_config import DEVICE, NUM_CLASSES, PAD_IDX, MAX_SEQ_LEN, decode_token_ids
from revised_pad_eos_dataset_recog_only import CropDataset, train_crop_transform, val_crop_transform
from revised_pad_eos_model_recog_only import PlateRecognizer


def masked_char_accuracy(pred_idx, gt_idx):
    valid_mask = gt_idx != PAD_IDX
    if valid_mask.sum().item() == 0:
        return 0.0
    correct = ((pred_idx == gt_idx) & valid_mask).sum().item()
    total = valid_mask.sum().item()
    return correct, total


def full_plate_accuracy(pred_idx, gt_idx):
    correct = 0
    total = gt_idx.size(0)
    for pred_seq, target_seq in zip(pred_idx, gt_idx):
        pred_text = decode_token_ids(pred_seq.tolist())
        target_text = decode_token_ids(target_seq.tolist())
        correct += int(pred_text == target_text)
    return correct, total


def main():
    data_dir = "data_prepared/dataset_unified_recog"
    ckpt_dir = "rfdetr/checkpoints_pure_recog_merge_pad_eos"
    os.makedirs(ckpt_dir, exist_ok=True)

    train_ds = CropDataset(
        root_dir=os.path.join(data_dir, "images"),
        csv_file=os.path.join(data_dir, "train.csv"),
        transform=train_crop_transform,
    )
    val_ds = CropDataset(
        root_dir=os.path.join(data_dir, "images"),
        csv_file=os.path.join(data_dir, "valid.csv"),
        transform=val_crop_transform,
    )

    pin_memory = DEVICE.startswith("cuda")
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=pin_memory)

    model = PlateRecognizer(num_classes=NUM_CLASSES, max_seq_len=MAX_SEQ_LEN).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX, label_smoothing=0.05)

    epochs = 30
    best_val_loss = float("inf")

    print("Mulai training recognizer PAD+EOS...")
    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs} [TRAIN]")
        for crop_imgs, gt_labels in train_loop:
            crop_imgs = crop_imgs.to(DEVICE, non_blocking=True)
            gt_labels = gt_labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred_chars = model(crop_imgs)
            loss = criterion(pred_chars.reshape(-1, NUM_CLASSES), gt_labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_loop.set_postfix(loss=f"{loss.item():.4f}")

        model.eval()
        val_loss_sum = 0.0
        char_correct = 0
        char_total = 0
        plate_correct = 0
        plate_total = 0

        with torch.no_grad():
            for crop_imgs, gt_labels in val_loader:
                crop_imgs = crop_imgs.to(DEVICE, non_blocking=True)
                gt_labels = gt_labels.to(DEVICE, non_blocking=True)

                pred_chars = model(crop_imgs)
                loss = criterion(pred_chars.reshape(-1, NUM_CLASSES), gt_labels.reshape(-1))
                val_loss_sum += loss.item()

                pred_idx = torch.argmax(pred_chars, dim=2)
                c_corr, c_tot = masked_char_accuracy(pred_idx, gt_labels)
                p_corr, p_tot = full_plate_accuracy(pred_idx, gt_labels)
                char_correct += c_corr
                char_total += c_tot
                plate_correct += p_corr
                plate_total += p_tot

        avg_train_loss = train_loss_sum / max(len(train_loader), 1)
        avg_val_loss = val_loss_sum / max(len(val_loader), 1)
        char_acc = 100.0 * char_correct / max(char_total, 1)
        plate_acc = 100.0 * plate_correct / max(plate_total, 1)
        scheduler.step(avg_val_loss)

        print(
            f"--> Train Loss: {avg_train_loss:.4f} | Valid Loss: {avg_val_loss:.4f} | "
            f"Char Acc: {char_acc:.2f}% | Full Plate Acc: {plate_acc:.2f}%"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint = {
                "state_dict": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "max_seq_len": MAX_SEQ_LEN,
                "epoch": epoch + 1,
            }
            torch.save(checkpoint, os.path.join(ckpt_dir, "best_recognizer_pad_eos.pth"))
            print("    [!] Best model saved")


if __name__ == "__main__":
    main()
