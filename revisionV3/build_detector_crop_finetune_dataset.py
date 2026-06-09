import argparse
import csv
import os
import random
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Tuple

import cv2
from tqdm import tqdm

from rfdetr import RFDETRNano


def sanitize_plate_text(text: str) -> str:
    return "".join(ch for ch in str(text).upper() if ch.isalnum())


def find_image_for_xml(xml_path: str) -> Optional[str]:
    stem = os.path.splitext(xml_path)[0]
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = stem + ext
        if os.path.exists(candidate):
            return candidate
    return None


def parse_xml(xml_path: str) -> Tuple[Optional[str], List[Dict]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_path = find_image_for_xml(xml_path)
    gts = []
    for obj in root.findall("object"):
        name_node = obj.find("name")
        bbox_node = obj.find("bndbox")
        if name_node is None or bbox_node is None:
            continue
        text = sanitize_plate_text(name_node.text or "")
        if not text:
            continue
        try:
            xmin = int(float(bbox_node.find("xmin").text))
            ymin = int(float(bbox_node.find("ymin").text))
            xmax = int(float(bbox_node.find("xmax").text))
            ymax = int(float(bbox_node.find("ymax").text))
        except Exception:
            continue
        if xmax <= xmin or ymax <= ymin:
            continue
        gts.append({"bbox": [xmin, ymin, xmax, ymax], "text": text})
    return image_path, gts


def iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter = (x_right - x_left) * (y_bottom - y_top)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / max(union, 1e-6)


def greedy_match(gts, preds, iou_thresh):
    cand = []
    for gi, gt in enumerate(gts):
        for pi, pred in enumerate(preds):
            ov = iou(gt["bbox"], pred["bbox"])
            if ov >= iou_thresh:
                cand.append((ov, gi, pi))
    cand.sort(reverse=True, key=lambda x: x[0])
    used_g = set()
    used_p = set()
    out = []
    for ov, gi, pi in cand:
        if gi in used_g or pi in used_p:
            continue
        used_g.add(gi)
        used_p.add(pi)
        out.append((gi, pi, ov))
    return out


def collect_xml_files(data_dir: str) -> List[str]:
    xml_files = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".xml"):
                xml_files.append(os.path.join(root, file))
    xml_files.sort()
    return xml_files


def split_items(items: List[str], train_ratio: float, valid_ratio: float, seed: int):
    rng = random.Random(seed)
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_train = int(round(n * train_ratio))
    n_valid = int(round(n * valid_ratio))
    train = items[:n_train]
    valid = items[n_train:n_train + n_valid]
    test = items[n_train + n_valid:]
    return train, valid, test


def save_rows(csv_path: str, rows: List[Dict]):
    fieldnames = [
        "filename", "label", "xml_path", "image_path", "iou", "det_conf", "gt_bbox", "pred_bbox"
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Bangun dataset fine-tune OCR dari crop hasil detector")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--det-weights", type=str, required=True)
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--det-conf", type=float, default=0.5)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--crop-expand-x", type=float, default=0.08)
    parser.add_argument("--crop-expand-y", type=float, default=0.12)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--valid-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    images_dir = os.path.join(args.out_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    xml_files = collect_xml_files(args.data_dir)
    if not xml_files:
        raise FileNotFoundError(f"No XML files found in {args.data_dir}")

    train_xml, valid_xml, test_xml = split_items(xml_files, args.train_ratio, args.valid_ratio, args.seed)
    split_map = {p: "train" for p in train_xml}
    split_map.update({p: "valid" for p in valid_xml})
    split_map.update({p: "test" for p in test_xml})

    detector = RFDETRNano(classes=["license_plate"], pretrain_weights=args.det_weights)

    rows_by_split = {"train": [], "valid": [], "test": []}
    stats = {"images": 0, "matched_crops": 0, "gt_total": 0, "pred_total": 0}
    crop_idx = 0

    for xml_path in tqdm(xml_files, desc="Building detector-crop dataset"):
        split = split_map[xml_path]
        image_path, gts = parse_xml(xml_path)
        if image_path is None or not os.path.exists(image_path) or not gts:
            continue
        image = cv2.imread(image_path)
        if image is None:
            continue

        preds_raw = detector.predict(image, conf=args.det_conf)
        preds = []
        if len(preds_raw) > 0 and preds_raw.confidence is not None and len(preds_raw.confidence) > 0:
            for i in range(len(preds_raw.xyxy)):
                x1, y1, x2, y2 = map(int, preds_raw.xyxy[i])
                preds.append({
                    "bbox": [x1, y1, x2, y2],
                    "det_conf": float(preds_raw.confidence[i]),
                })

        matches = greedy_match(gts, preds, args.iou_thresh)
        stats["images"] += 1
        stats["gt_total"] += len(gts)
        stats["pred_total"] += len(preds)

        h, w = image.shape[:2]
        for gi, pi, ov in matches:
            gt = gts[gi]
            pred = preds[pi]
            x1, y1, x2, y2 = pred["bbox"]
            pad_x = int((x2 - x1) * args.crop_expand_x)
            pad_y = int((y2 - y1) * args.crop_expand_y)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w, x2 + pad_x)
            y2 = min(h, y2 + pad_y)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            fname = f"detcrop_{crop_idx:07d}.jpg"
            crop_idx += 1
            cv2.imwrite(os.path.join(images_dir, fname), crop)
            rows_by_split[split].append({
                "filename": fname,
                "label": gt["text"],
                "xml_path": xml_path,
                "image_path": image_path,
                "iou": round(ov, 6),
                "det_conf": round(pred["det_conf"], 6),
                "gt_bbox": gt["bbox"],
                "pred_bbox": pred["bbox"],
            })
            stats["matched_crops"] += 1

    for split, rows in rows_by_split.items():
        save_rows(os.path.join(args.out_dir, f"{split}.csv"), rows)

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("===== DETECTOR-CROP DATASET SUMMARY =====\n")
        f.write(f"Images processed : {stats['images']}\n")
        f.write(f"GT plates total  : {stats['gt_total']}\n")
        f.write(f"Pred plates total: {stats['pred_total']}\n")
        f.write(f"Matched crops    : {stats['matched_crops']}\n")
        for split in ("train", "valid", "test"):
            f.write(f"{split.title()} rows      : {len(rows_by_split[split])}\n")

    print("===== DETECTOR-CROP DATASET BUILT =====")
    print(f"Images processed : {stats['images']}")
    print(f"GT plates total  : {stats['gt_total']}")
    print(f"Pred plates total: {stats['pred_total']}")
    print(f"Matched crops    : {stats['matched_crops']}")
    for split in ("train", "valid", "test"):
        print(f"{split.title()} rows      : {len(rows_by_split[split])}")
    print(f"Output dir       : {args.out_dir}")


if __name__ == "__main__":
    main()
