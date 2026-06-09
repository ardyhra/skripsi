import argparse
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ctc_config import DEVICE, NUM_CLASSES, ctc_greedy_decode, levenshtein
from ctc_dataset_recog_only import CTCCropDataset, ctc_collate_fn, val_aug
from ctc_model_recog_only import PlateRecognizerCTC


def main():
    parser = argparse.ArgumentParser(description="Evaluate recognizer CTC on cropped plate dataset")
    parser.add_argument("--data-dir", type=str, default="data_prepared/dataset_unified_recog")
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--save-csv", type=str, default=None)
    args = parser.parse_args()

    ds = CTCCropDataset(
        root_dir=os.path.join(args.data_dir, "images"),
        csv_file=os.path.join(args.data_dir, f"{args.split}.csv"),
        augment=val_aug,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=DEVICE.startswith("cuda"),
        collate_fn=ctc_collate_fn,
    )

    ckpt = torch.load(args.weights, map_location=DEVICE, weights_only=False)
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    num_classes = ckpt.get("num_classes", NUM_CLASSES) if isinstance(ckpt, dict) else NUM_CLASSES

    model = PlateRecognizerCTC(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    total = 0
    exact = 0
    total_gt_chars = 0
    total_correct_chars = 0
    total_cer = 0.0
    rows = []

    with torch.no_grad():
        for images, _, _, gt_texts, filenames in tqdm(loader, desc=f"Eval {args.split}"):
            images = images.to(DEVICE, non_blocking=True)
            logits, _ = model(images)
            pred_texts, confs = ctc_greedy_decode(logits)

            for fname, gt, pred, conf in zip(filenames, gt_texts, pred_texts, confs):
                dist = levenshtein(pred, gt)
                total += 1
                exact += int(pred == gt)
                total_gt_chars += len(gt)
                total_correct_chars += max(len(gt) - dist, 0)
                total_cer += dist / max(len(gt), 1)
                rows.append({
                    "filename": fname,
                    "gt_text": gt,
                    "pred_text": pred,
                    "confidence": conf,
                    "edit_distance": dist,
                    "cer": dist / max(len(gt), 1),
                    "exact": int(pred == gt),
                })

    exact_acc = 100.0 * exact / max(total, 1)
    char_acc = 100.0 * total_correct_chars / max(total_gt_chars, 1)
    mean_cer = 100.0 * total_cer / max(total, 1)

    print("===== HASIL EVALUASI RECOGNIZER CTC =====")
    print(f"Samples evaluated      : {total}")
    print(f"Full Plate Accuracy    : {exact_acc:.2f}%")
    print(f"Aligned Char Accuracy  : {char_acc:.2f}%")
    print(f"Mean CER               : {mean_cer:.2f}%")

    if args.save_csv:
        pd.DataFrame(rows).to_csv(args.save_csv, index=False)
        print(f"Detail prediksi disimpan ke {args.save_csv}")


if __name__ == "__main__":
    main()
