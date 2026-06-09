import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from ctc_config import BLANK_IDX, DEVICE, NUM_CLASSES, ctc_greedy_decode, levenshtein
from ctc_dataset_recog_only import CTCCropDataset, ctc_collate_fn, train_aug, val_aug
from ctc_model_recog_only import PlateRecognizerCTC


def recognition_metrics(pred_texts, gt_texts):
    total = len(gt_texts)
    exact = 0
    total_chars = 0
    total_correct_chars = 0
    total_cer = 0.0

    for pred, gt in zip(pred_texts, gt_texts):
        if pred == gt:
            exact += 1
        dist = levenshtein(pred, gt)
        total_cer += dist / max(len(gt), 1)
        total_chars += len(gt)
        total_correct_chars += max(len(gt) - dist, 0)

    return {
        "exact": exact,
        "total": total,
        "char_correct": total_correct_chars,
        "char_total": total_chars,
        "mean_cer": total_cer / max(total, 1),
    }


def run_epoch(model, loader, criterion, optimizer=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()

    loss_sum = 0.0
    stats = {"exact": 0, "total": 0, "char_correct": 0, "char_total": 0, "mean_cer_sum": 0.0}

    loop = tqdm(loader, desc="TRAIN" if train_mode else "VALID")
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for images, flat_targets, target_lengths, gt_texts, _ in loop:
            images = images.to(DEVICE, non_blocking=True)
            flat_targets = flat_targets.to(DEVICE, non_blocking=True)
            target_lengths = target_lengths.to(DEVICE, non_blocking=True)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            logits, input_lengths = model(images)
            log_probs = logits.log_softmax(dim=-1).permute(1, 0, 2)
            loss = criterion(log_probs, flat_targets, input_lengths, target_lengths)

            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            loss_sum += float(loss.item())
            pred_texts, _ = ctc_greedy_decode(logits)
            batch_stats = recognition_metrics(pred_texts, gt_texts)
            stats["exact"] += batch_stats["exact"]
            stats["total"] += batch_stats["total"]
            stats["char_correct"] += batch_stats["char_correct"]
            stats["char_total"] += batch_stats["char_total"]
            stats["mean_cer_sum"] += batch_stats["mean_cer"] * batch_stats["total"]

            exact_acc = 100.0 * stats["exact"] / max(stats["total"], 1)
            loop.set_postfix(loss=f"{loss.item():.4f}", exact=f"{exact_acc:.2f}%")

    avg_loss = loss_sum / max(len(loader), 1)
    exact_acc = 100.0 * stats["exact"] / max(stats["total"], 1)
    char_acc = 100.0 * stats["char_correct"] / max(stats["char_total"], 1)
    mean_cer = stats["mean_cer_sum"] / max(stats["total"], 1)
    return avg_loss, exact_acc, char_acc, mean_cer


def main():
    parser = argparse.ArgumentParser(description="Train recognizer CTC + transformer encoder")
    parser.add_argument("--data-dir", type=str, default="data_prepared/dataset_unified_recog")
    parser.add_argument("--out-dir", type=str, default="rfdetr/checkpoints_recog_ctc")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = CTCCropDataset(
        root_dir=os.path.join(args.data_dir, "images"),
        csv_file=os.path.join(args.data_dir, "train.csv"),
        augment=train_aug,
    )
    val_ds = CTCCropDataset(
        root_dir=os.path.join(args.data_dir, "images"),
        csv_file=os.path.join(args.data_dir, "valid.csv"),
        augment=val_aug,
    )

    pin_memory = DEVICE.startswith("cuda")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=ctc_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        collate_fn=ctc_collate_fn,
    )

    model = PlateRecognizerCTC(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CTCLoss(blank=BLANK_IDX, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_loss = float("inf")
    print("Mulai training recognizer CTC...")
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        train_loss, train_exact, train_char, train_cer = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_exact, val_char, val_cer = run_epoch(model, val_loader, criterion, optimizer=None)
        scheduler.step(val_loss)

        print(
            f"--> Train Loss: {train_loss:.4f} | Valid Loss: {val_loss:.4f} | "
            f"Valid Char Acc: {val_char:.2f}% | Valid Full Plate Acc: {val_exact:.2f}% | Valid CER: {val_cer * 100:.2f}%"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "state_dict": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "blank_idx": BLANK_IDX,
                "epoch": epoch + 1,
                "arch": "transformer_ctc",
            }, os.path.join(args.out_dir, "best_recognizer_ctc.pth"))
            print("    [!] Best CTC model saved")


if __name__ == "__main__":
    main()
