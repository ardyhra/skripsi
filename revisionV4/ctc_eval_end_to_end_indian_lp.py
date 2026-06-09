import argparse
import csv
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from rfdetr import RFDETRNano

from ctc_config import CROP_HEIGHT, CROP_WIDTH, DEVICE, NUM_CLASSES, ctc_greedy_decode, letterbox_plate_image, levenshtein, sanitize_plate_text
from ctc_model_recog_only import PlateRecognizerCTC

_post_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


@dataclass
class GTPlate:
    bbox: List[int]
    text: str


@dataclass
class PredPlate:
    bbox: List[int]
    text: str
    det_conf: float
    text_conf: float


@dataclass
class MatchResult:
    gt_idx: int
    pred_idx: int
    iou: float


def get_iou(box1: List[int], box2: List[int]) -> float:
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    box2_area = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = box1_area + box2_area - inter_area
    return inter_area / max(union, 1e-6)


def find_image_for_xml(xml_path: str) -> Optional[str]:
    stem = os.path.splitext(xml_path)[0]
    for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp"):
        candidate = stem + ext
        if os.path.exists(candidate):
            return candidate
    return None


def parse_indian_lp_xml(xml_path: str) -> Tuple[Optional[str], List[GTPlate]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    image_path = find_image_for_xml(xml_path)
    gt_objects: List[GTPlate] = []
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
        gt_objects.append(GTPlate(bbox=[xmin, ymin, xmax, ymax], text=text))
    return image_path, gt_objects


def greedy_match(gts: List[GTPlate], preds: List[PredPlate], iou_thresh: float) -> List[MatchResult]:
    candidates: List[Tuple[float, int, int]] = []
    for gi, gt in enumerate(gts):
        for pi, pred in enumerate(preds):
            iou = get_iou(gt.bbox, pred.bbox)
            if iou >= iou_thresh:
                candidates.append((iou, gi, pi))
    candidates.sort(reverse=True, key=lambda x: x[0])

    used_gts = set()
    used_preds = set()
    matches: List[MatchResult] = []
    for iou, gi, pi in candidates:
        if gi in used_gts or pi in used_preds:
            continue
        used_gts.add(gi)
        used_preds.add(pi)
        matches.append(MatchResult(gt_idx=gi, pred_idx=pi, iou=iou))
    return matches


def crop_to_tensor(crop_bgr: np.ndarray) -> torch.Tensor:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    crop_rgb = letterbox_plate_image(crop_rgb, CROP_HEIGHT, CROP_WIDTH)
    return _post_transform(image=crop_rgb)["image"].unsqueeze(0)


class EndToEndALPRCTC:
    def __init__(self, det_weights: str, rec_weights: str, det_conf: float, max_chars: Optional[int], expected_chars: Optional[int], crop_expand_x: float, crop_expand_y: float):
        print("Loading detector...")
        self.detector = RFDETRNano(classes=["license_plate"], pretrain_weights=det_weights)
        self.det_conf = det_conf
        self.max_chars = max_chars
        self.expected_chars = expected_chars
        self.crop_expand_x = crop_expand_x
        self.crop_expand_y = crop_expand_y

        print("Loading recognizer CTC...")
        ckpt = torch.load(rec_weights, map_location=DEVICE, weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        num_classes = ckpt.get("num_classes", NUM_CLASSES) if isinstance(ckpt, dict) else NUM_CLASSES
        self.recognizer = PlateRecognizerCTC(num_classes=num_classes).to(DEVICE)
        self.recognizer.load_state_dict(state_dict, strict=False)
        self.recognizer.eval()

    def predict(self, image_bgr: np.ndarray) -> List[PredPlate]:
        h_orig, w_orig = image_bgr.shape[:2]
        predictions = self.detector.predict(image_bgr, conf=self.det_conf)
        if len(predictions) == 0 or predictions.confidence is None or len(predictions.confidence) == 0:
            return []

        results: List[PredPlate] = []
        for i in range(len(predictions.xyxy)):
            x1, y1, x2, y2 = map(int, predictions.xyxy[i])
            det_conf = float(predictions.confidence[i])
            pad_x = int((x2 - x1) * self.crop_expand_x)
            pad_y = int((y2 - y1) * self.crop_expand_y)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w_orig, x2 + pad_x)
            y2 = min(h_orig, y2 + pad_y)

            crop = image_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_tensor = crop_to_tensor(crop).to(DEVICE)
            with torch.no_grad():
                logits, _ = self.recognizer(crop_tensor)
                texts, confs = ctc_greedy_decode(logits, max_chars=self.max_chars)
            text = sanitize_plate_text(texts[0])
            if self.expected_chars is not None:
                text = text[:self.expected_chars]
            results.append(PredPlate(bbox=[x1, y1, x2, y2], text=text, det_conf=det_conf, text_conf=confs[0]))
        return results


class RunningMetrics:
    def __init__(self):
        self.images = 0
        self.gt_total = 0
        self.pred_total = 0
        self.det_tp = 0
        self.det_fp = 0
        self.det_fn = 0
        self.ocr_exact_on_matched = 0
        self.e2e_tp = 0
        self.sum_cer = 0.0
        self.cer_count = 0
        self.gt_chars_matched = 0
        self.correct_chars_matched = 0

    def update(self, gts: List[GTPlate], preds: List[PredPlate], matches: List[MatchResult]):
        self.images += 1
        self.gt_total += len(gts)
        self.pred_total += len(preds)
        self.det_tp += len(matches)
        self.det_fp += max(0, len(preds) - len(matches))
        self.det_fn += max(0, len(gts) - len(matches))

        for m in matches:
            gt = gts[m.gt_idx]
            pred = preds[m.pred_idx]
            dist = levenshtein(pred.text, gt.text)
            self.sum_cer += dist / max(len(gt.text), 1)
            self.cer_count += 1
            self.gt_chars_matched += len(gt.text)
            self.correct_chars_matched += max(len(gt.text) - dist, 0)
            if pred.text == gt.text:
                self.ocr_exact_on_matched += 1
                self.e2e_tp += 1

    def summary(self):
        det_precision = self.det_tp / max(self.det_tp + self.det_fp, 1)
        det_recall = self.det_tp / max(self.det_tp + self.det_fn, 1)
        det_f1 = 2 * det_precision * det_recall / max(det_precision + det_recall, 1e-12)
        ocr_exact = self.ocr_exact_on_matched / max(self.det_tp, 1)
        ocr_char_acc = self.correct_chars_matched / max(self.gt_chars_matched, 1)
        mean_cer = self.sum_cer / max(self.cer_count, 1)
        e2e_precision = self.e2e_tp / max(self.pred_total, 1)
        e2e_recall = self.e2e_tp / max(self.gt_total, 1)
        e2e_f1 = 2 * e2e_precision * e2e_recall / max(e2e_precision + e2e_recall, 1e-12)
        return {
            "images": self.images,
            "gt_total": self.gt_total,
            "pred_total": self.pred_total,
            "det_tp": self.det_tp,
            "det_fp": self.det_fp,
            "det_fn": self.det_fn,
            "det_precision": det_precision,
            "det_recall": det_recall,
            "det_f1": det_f1,
            "ocr_exact_on_matched": ocr_exact,
            "ocr_char_acc_on_matched": ocr_char_acc,
            "mean_cer_on_matched": mean_cer,
            "e2e_tp": self.e2e_tp,
            "e2e_precision": e2e_precision,
            "e2e_recall": e2e_recall,
            "e2e_f1": e2e_f1,
        }


def iter_xml_files(data_dir: str):
    for root, _, files in os.walk(data_dir):
        for fname in files:
            if fname.lower().endswith(".xml"):
                yield os.path.join(root, fname)


def main():
    parser = argparse.ArgumentParser(description="End-to-end evaluation on Indian LP XML with transformer+CTC recognizer")
    parser.add_argument("--data-dir", type=str, required=True, help="Root folder Indian LP berisi image + XML")
    parser.add_argument("--det-weights", type=str, required=True)
    parser.add_argument("--rec-weights", type=str, required=True)
    parser.add_argument("--iou-thresh", type=float, default=0.5)
    parser.add_argument("--det-conf", type=float, default=0.5)
    parser.add_argument("--crop-expand-x", type=float, default=0.08)
    parser.add_argument("--crop-expand-y", type=float, default=0.12)
    parser.add_argument("--max-chars", type=int, default=None)
    parser.add_argument("--expected-chars", type=int, default=None)
    parser.add_argument("--save-csv", type=str, default=None, help="Simpan detail per match ke CSV")
    args = parser.parse_args()

    system = EndToEndALPRCTC(
        det_weights=args.det_weights,
        rec_weights=args.rec_weights,
        det_conf=args.det_conf,
        max_chars=args.max_chars,
        expected_chars=args.expected_chars,
        crop_expand_x=args.crop_expand_x,
        crop_expand_y=args.crop_expand_y,
    )

    metrics = RunningMetrics()
    rows = []
    xml_files = sorted(iter_xml_files(args.data_dir))
    for xml_path in tqdm(xml_files, desc="Evaluating"):
        image_path, gts = parse_indian_lp_xml(xml_path)
        if image_path is None or not gts:
            continue
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            continue
        preds = system.predict(image_bgr)
        matches = greedy_match(gts, preds, args.iou_thresh)
        metrics.update(gts, preds, matches)

        if args.save_csv:
            matched_gt = {m.gt_idx for m in matches}
            matched_pred = {m.pred_idx for m in matches}
            for m in matches:
                gt = gts[m.gt_idx]
                pred = preds[m.pred_idx]
                dist = levenshtein(pred.text, gt.text)
                rows.append({
                    "xml_path": xml_path,
                    "image_path": image_path,
                    "status": "matched",
                    "iou": m.iou,
                    "gt_text": gt.text,
                    "pred_text": pred.text,
                    "det_conf": pred.det_conf,
                    "text_conf": pred.text_conf,
                    "cer": dist / max(len(gt.text), 1),
                    "exact": int(gt.text == pred.text),
                })
            for gi, gt in enumerate(gts):
                if gi not in matched_gt:
                    rows.append({
                        "xml_path": xml_path,
                        "image_path": image_path,
                        "status": "gt_missed",
                        "iou": 0.0,
                        "gt_text": gt.text,
                        "pred_text": "",
                        "det_conf": 0.0,
                        "text_conf": 0.0,
                        "cer": 1.0,
                        "exact": 0,
                    })
            for pi, pred in enumerate(preds):
                if pi not in matched_pred:
                    rows.append({
                        "xml_path": xml_path,
                        "image_path": image_path,
                        "status": "pred_unmatched",
                        "iou": 0.0,
                        "gt_text": "",
                        "pred_text": pred.text,
                        "det_conf": pred.det_conf,
                        "text_conf": pred.text_conf,
                        "cer": "",
                        "exact": 0,
                    })

    summary = metrics.summary()
    print("===== HASIL EVALUASI END-TO-END CTC =====")
    print(f"Images evaluated            : {summary['images']}")
    print(f"Total GT plates             : {summary['gt_total']}")
    print(f"Total predicted plates      : {summary['pred_total']}")
    print(f"Detection TP / FP / FN      : {summary['det_tp']} / {summary['det_fp']} / {summary['det_fn']}")
    print(f"Detection Precision         : {summary['det_precision'] * 100:.2f}%")
    print(f"Detection Recall            : {summary['det_recall'] * 100:.2f}%")
    print(f"Detection F1                : {summary['det_f1'] * 100:.2f}%")
    print(f"OCR Exact on matched boxes  : {summary['ocr_exact_on_matched'] * 100:.2f}%")
    print(f"OCR Char Acc on matched     : {summary['ocr_char_acc_on_matched'] * 100:.2f}%")
    print(f"Mean CER on matched         : {summary['mean_cer_on_matched'] * 100:.2f}%")
    print(f"End-to-End TP               : {summary['e2e_tp']}")
    print(f"End-to-End Precision        : {summary['e2e_precision'] * 100:.2f}%")
    print(f"End-to-End Recall           : {summary['e2e_recall'] * 100:.2f}%")
    print(f"End-to-End F1               : {summary['e2e_f1'] * 100:.2f}%")

    if args.save_csv:
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["xml_path"])
            writer.writeheader()
            if rows:
                writer.writerows(rows)
        print(f"Detail evaluasi disimpan ke {args.save_csv}")


if __name__ == "__main__":
    main()
