import argparse
import csv
import json
import os
from dataclasses import dataclass
from typing import Dict, List

import cv2
from tqdm import tqdm

from rfdetr import RFDETRNano


VALID_IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]
COCO_SPLITS = ["test", "valid", "train"]


@dataclass
class GTBox:
    bbox: List[int]
    label: str


@dataclass
class PredBox:
    bbox: List[int]
    score: float


@dataclass
class MatchResult:
    gt_idx: int
    pred_idx: int
    iou: float


class RunningMetrics:
    def __init__(self):
        self.images = 0
        self.gt_total = 0
        self.pred_total = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.iou_sum = 0.0
        self.small_gt = 0
        self.medium_gt = 0
        self.large_gt = 0
        self.small_tp = 0
        self.medium_tp = 0
        self.large_tp = 0

    def update(self, gts: List[GTBox], preds: List[PredBox], matches: List[MatchResult]):
        self.images += 1
        self.gt_total += len(gts)
        self.pred_total += len(preds)
        self.tp += len(matches)
        self.fp += max(0, len(preds) - len(matches))
        self.fn += max(0, len(gts) - len(matches))

        matched_gt = {m.gt_idx for m in matches}
        for m in matches:
            self.iou_sum += m.iou

        for i, gt in enumerate(gts):
            x1, y1, x2, y2 = gt.bbox
            area = max(0, x2 - x1) * max(0, y2 - y1)
            # COCO-like area buckets, still useful for LP size analysis
            if area < 32 * 32:
                self.small_gt += 1
                if i in matched_gt:
                    self.small_tp += 1
            elif area < 96 * 96:
                self.medium_gt += 1
                if i in matched_gt:
                    self.medium_tp += 1
            else:
                self.large_gt += 1
                if i in matched_gt:
                    self.large_tp += 1

    def summarize(self):
        def safe_div(a, b):
            return a / b if b > 0 else 0.0

        precision = safe_div(self.tp, self.tp + self.fp)
        recall = safe_div(self.tp, self.tp + self.fn)
        f1 = safe_div(2 * precision * recall, precision + recall)
        mean_iou = safe_div(self.iou_sum, self.tp)
        return {
            "images": self.images,
            "gt_total": self.gt_total,
            "pred_total": self.pred_total,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "mean_iou_on_matches": mean_iou,
            "small_gt": self.small_gt,
            "small_recall": safe_div(self.small_tp, self.small_gt),
            "medium_gt": self.medium_gt,
            "medium_recall": safe_div(self.medium_tp, self.medium_gt),
            "large_gt": self.large_gt,
            "large_recall": safe_div(self.large_tp, self.large_gt),
        }


def sanitize_text(text: str) -> str:
    return "".join(ch for ch in (text or "").upper().strip() if ch.isalnum())


def get_iou(box1: List[int], box2: List[int]) -> float:
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    return inter_area / max(union, 1e-6)


def greedy_match(gts: List[GTBox], preds: List[PredBox], iou_thresh: float) -> List[MatchResult]:
    candidates: List[MatchResult] = []
    for gi, gt in enumerate(gts):
        for pi, pred in enumerate(preds):
            iou = get_iou(gt.bbox, pred.bbox)
            if iou >= iou_thresh:
                candidates.append(MatchResult(gt_idx=gi, pred_idx=pi, iou=iou))
    candidates.sort(key=lambda x: x.iou, reverse=True)

    used_gt = set()
    used_pred = set()
    matches: List[MatchResult] = []
    for c in candidates:
        if c.gt_idx in used_gt or c.pred_idx in used_pred:
            continue
        used_gt.add(c.gt_idx)
        used_pred.add(c.pred_idx)
        matches.append(c)
    return matches


def resolve_coco_dir(data_dir: str, split: str | None = None) -> str:
    data_dir = os.path.abspath(data_dir)

    if split:
        candidate = os.path.join(data_dir, split)
        ann_path = os.path.join(candidate, "_annotations.coco.json")
        if os.path.exists(ann_path):
            return candidate
        raise FileNotFoundError(f"COCO annotation file not found for split '{split}': {ann_path}")

    direct_ann = os.path.join(data_dir, "_annotations.coco.json")
    if os.path.exists(direct_ann):
        return data_dir

    for candidate_split in COCO_SPLITS:
        candidate = os.path.join(data_dir, candidate_split)
        ann_path = os.path.join(candidate, "_annotations.coco.json")
        if os.path.exists(ann_path):
            return candidate

    raise FileNotFoundError(
        f"COCO annotation file not found in '{data_dir}' or its split folders {COCO_SPLITS}"
    )


def load_coco_samples(data_dir: str) -> List[dict]:
    ann_path = os.path.join(data_dir, "_annotations.coco.json")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"COCO annotation file not found: {ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    categories = {cat["id"]: sanitize_text(cat.get("name", "")) for cat in coco.get("categories", [])}
    images = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image: Dict[int, List[GTBox]] = {}

    for ann in coco.get("annotations", []):
        image_id = ann.get("image_id")
        bbox = ann.get("bbox", [])
        if image_id not in images or len(bbox) != 4:
            continue
        x, y, w, h = bbox
        x1 = int(round(float(x)))
        y1 = int(round(float(y)))
        x2 = int(round(float(x + w)))
        y2 = int(round(float(y + h)))
        if x2 <= x1 or y2 <= y1:
            continue
        label = categories.get(ann.get("category_id"), "PLATE")
        anns_by_image.setdefault(image_id, []).append(GTBox(bbox=[x1, y1, x2, y2], label=label))

    samples: List[dict] = []
    for image_id, img in sorted(images.items(), key=lambda kv: kv[1].get("file_name", "")):
        image_path = os.path.join(data_dir, img.get("file_name", ""))
        samples.append(
            {
                "image_path": image_path,
                "image_relpath": img.get("file_name", ""),
                "gts": anns_by_image.get(image_id, []),
            }
        )
    return samples


def predict_boxes(model: RFDETRNano, image_bgr, conf: float) -> List[PredBox]:
    raw = model.predict(image_bgr, conf=conf)
    preds: List[PredBox] = []
    if len(raw) == 0 or raw.confidence is None or len(raw.confidence) == 0:
        return preds

    for i in range(len(raw.xyxy)):
        x1, y1, x2, y2 = map(int, raw.xyxy[i])
        if x2 <= x1 or y2 <= y1:
            continue
        score = float(raw.confidence[i]) if raw.confidence is not None else 1.0
        preds.append(PredBox(bbox=[x1, y1, x2, y2], score=score))
    return preds


def print_summary(title: str, summary: dict):
    print(f"===== {title} =====")
    print(f"Images evaluated         : {summary['images']}")
    print(f"Total GT boxes           : {summary['gt_total']}")
    print(f"Total predicted boxes    : {summary['pred_total']}")
    print(f"TP / FP / FN             : {summary['tp']} / {summary['fp']} / {summary['fn']}")
    print(f"Precision                : {summary['precision'] * 100:.2f}%")
    print(f"Recall                   : {summary['recall'] * 100:.2f}%")
    print(f"F1                       : {summary['f1'] * 100:.2f}%")
    print(f"Mean IoU on matches      : {summary['mean_iou_on_matches'] * 100:.2f}%")
    print(f"Small GT / Recall        : {summary['small_gt']} / {summary['small_recall'] * 100:.2f}%")
    print(f"Medium GT / Recall       : {summary['medium_gt']} / {summary['medium_recall'] * 100:.2f}%")
    print(f"Large GT / Recall        : {summary['large_gt']} / {summary['large_recall'] * 100:.2f}%")


def save_csv(path: str, rows: List[dict]):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Evaluate RF-DETR license plate detection on a COCO-format dataset")
    parser.add_argument("--data-dir", type=str, required=True, help="Folder containing images and _annotations.coco.json, or a dataset root with split folders")
    parser.add_argument("--split", type=str, default=None, choices=COCO_SPLITS, help="Optional split to evaluate when --data-dir points to a dataset root")
    parser.add_argument("--det-weights", type=str, required=True, help="Path to RF-DETR checkpoint")
    parser.add_argument("--det-conf", type=float, default=0.5, help="Detection confidence threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold for matching")
    parser.add_argument("--extra-iou-thresh", type=float, nargs="*", default=[0.75], help="Additional IoU thresholds to report")
    parser.add_argument("--save-csv", type=str, default=None, help="Optional per-image metrics CSV")
    args = parser.parse_args()

    resolved_data_dir = resolve_coco_dir(args.data_dir, split=args.split)
    print(f"Using COCO annotations from: {resolved_data_dir}")

    samples = load_coco_samples(resolved_data_dir)
    if not samples:
        raise FileNotFoundError(f"No images found in COCO annotation file under: {resolved_data_dir}")

    model = RFDETRNano(classes=["license_plate"], pretrain_weights=args.det_weights)

    thresholds = [args.iou_thresh] + [t for t in args.extra_iou_thresh if t not in {args.iou_thresh}]
    metrics_by_thr = {thr: RunningMetrics() for thr in thresholds}
    csv_rows: List[dict] = []

    for sample in tqdm(samples, desc="Evaluating detection"):
        image_path = sample["image_path"]
        if not os.path.exists(image_path):
            continue
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            continue

        gts = sample["gts"]
        preds = predict_boxes(model, image_bgr, conf=args.det_conf)

        row = {
            "image": sample["image_relpath"],
            "gt_count": len(gts),
            "pred_count": len(preds),
        }

        for thr in thresholds:
            matches = greedy_match(gts, preds, thr)
            metrics_by_thr[thr].update(gts, preds, matches)
            row[f"tp_iou_{thr}"] = len(matches)
            row[f"fp_iou_{thr}"] = max(0, len(preds) - len(matches))
            row[f"fn_iou_{thr}"] = max(0, len(gts) - len(matches))
            row[f"mean_iou_iou_{thr}"] = (sum(m.iou for m in matches) / len(matches)) if matches else 0.0

        csv_rows.append(row)

    for idx, thr in enumerate(thresholds):
        title = f"HASIL EVALUASI DETEKSI (IoU >= {thr:.2f})"
        print_summary(title, metrics_by_thr[thr].summarize())
        if idx < len(thresholds) - 1:
            print()

    if args.save_csv:
        save_csv(args.save_csv, csv_rows)
        print(f"\nSaved per-image CSV to: {args.save_csv}")


if __name__ == "__main__":
    main()
