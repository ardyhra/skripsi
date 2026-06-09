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
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

from rfdetr import RFDETRNano

from revised_pad_eos_config import DEVICE, IDX2CHAR, PAD_IDX, EOS_IDX, NUM_CLASSES, MAX_SEQ_LEN, sanitize_plate_text
from revised_pad_eos_model_recog_only import PlateRecognizer


CROP_HEIGHT = 64
CROP_WIDTH = 256

crop_transform = A.Compose([
    A.Resize(CROP_HEIGHT, CROP_WIDTH),
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


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (0 if ca == cb else 1)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def decode_text(pred_logits: torch.Tensor, max_chars: Optional[int] = None, expected_chars: Optional[int] = None) -> Tuple[str, float]:
    probs = F.softmax(pred_logits, dim=2)
    max_probs, pred_indices = torch.max(probs, dim=2)

    chars = []
    confidences = []
    for i, idx in enumerate(pred_indices[0]):
        idx = int(idx.item())
        prob = float(max_probs[0][i].item())
        if idx == EOS_IDX:
            break
        if idx == PAD_IDX:
            continue
        if idx in IDX2CHAR:
            chars.append(IDX2CHAR[idx])
            confidences.append(prob)
            if expected_chars is not None and len(chars) >= expected_chars:
                break
            if max_chars is not None and len(chars) >= max_chars:
                break

    text = "".join(chars)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, avg_conf


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


class EndToEndALPR:
    def __init__(
        self,
        det_weights: str,
        rec_weights: str,
        det_conf: float,
        max_chars: Optional[int],
        expected_chars: Optional[int],
        crop_expand_x: float,
        crop_expand_y: float,
    ):
        print("Loading detector...")
        self.detector = RFDETRNano(classes=["license_plate"], pretrain_weights=det_weights)
        self.det_conf = det_conf
        self.max_chars = max_chars
        self.expected_chars = expected_chars
        self.crop_expand_x = crop_expand_x
        self.crop_expand_y = crop_expand_y

        print("Loading recognizer...")
        ckpt = torch.load(rec_weights, map_location=DEVICE, weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        num_classes = ckpt.get("num_classes", NUM_CLASSES) if isinstance(ckpt, dict) else NUM_CLASSES
        max_seq_len = ckpt.get("max_seq_len", MAX_SEQ_LEN) if isinstance(ckpt, dict) else MAX_SEQ_LEN

        self.recognizer = PlateRecognizer(num_classes=num_classes, max_seq_len=max_seq_len).to(DEVICE)
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

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_tensor = crop_transform(image=crop_rgb)["image"].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_chars = self.recognizer(crop_tensor)
                text, text_conf = decode_text(pred_chars, max_chars=self.max_chars, expected_chars=self.expected_chars)

            results.append(PredPlate(
                bbox=[x1, y1, x2, y2],
                text=sanitize_plate_text(text),
                det_conf=det_conf,
                text_conf=text_conf,
            ))
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
            gt_text = gts[m.gt_idx].text
            pred_text = preds[m.pred_idx].text
            if pred_text == gt_text:
                self.ocr_exact_on_matched += 1
                self.e2e_tp += 1
            dist = levenshtein(pred_text, gt_text)
            denom = max(len(gt_text), 1)
            self.sum_cer += dist / denom
            self.cer_count += 1

            self.gt_chars_matched += len(gt_text)
            overlap = sum(1 for a, b in zip(pred_text, gt_text) if a == b)
            self.correct_chars_matched += overlap

    def summarize(self):
        def safe_div(a, b):
            return a / b if b > 0 else 0.0

        det_precision = safe_div(self.det_tp, self.det_tp + self.det_fp)
        det_recall = safe_div(self.det_tp, self.det_tp + self.det_fn)
        det_f1 = safe_div(2 * det_precision * det_recall, det_precision + det_recall)

        ocr_exact_on_matched = safe_div(self.ocr_exact_on_matched, self.det_tp)
        mean_cer_on_matched = safe_div(self.sum_cer, self.cer_count)
        matched_char_acc = safe_div(self.correct_chars_matched, self.gt_chars_matched)

        e2e_precision = safe_div(self.e2e_tp, self.pred_total)
        e2e_recall = safe_div(self.e2e_tp, self.gt_total)
        e2e_f1 = safe_div(2 * e2e_precision * e2e_recall, e2e_precision + e2e_recall)

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
            "ocr_exact_on_matched": ocr_exact_on_matched,
            "matched_char_acc": matched_char_acc,
            "mean_cer_on_matched": mean_cer_on_matched,
            "e2e_tp": self.e2e_tp,
            "e2e_precision": e2e_precision,
            "e2e_recall": e2e_recall,
            "e2e_f1": e2e_f1,
        }


def collect_xml_files(data_dir: str) -> List[str]:
    xml_files: List[str] = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith(".xml"):
                xml_files.append(os.path.join(root, file))
    xml_files.sort()
    return xml_files


def main():
    parser = argparse.ArgumentParser(description="Evaluasi end-to-end ALPR pada dataset Indian LP XML")
    parser.add_argument("--data-dir", type=str, required=True, help="Folder root yang berisi pasangan image + XML Indian LP")
    parser.add_argument("--det-weights", type=str, required=True, help="Path checkpoint RF-DETR")
    parser.add_argument("--rec-weights", type=str, required=True, help="Path checkpoint recognizer")
    parser.add_argument("--det-conf", type=float, default=0.5, help="Confidence threshold detector")
    parser.add_argument("--iou-thresh", type=float, default=0.5, help="IoU threshold untuk match bbox")
    parser.add_argument("--max-chars", type=int, default=None, help="Batas maksimum karakter saat decode")
    parser.add_argument("--expected-chars", type=int, default=None, help="Pakai hanya jika memang ingin mengunci panjang karakter")
    parser.add_argument("--crop-expand-x", type=float, default=0.08, help="Ekspansi crop horizontal dari bbox prediksi")
    parser.add_argument("--crop-expand-y", type=float, default=0.12, help="Ekspansi crop vertikal dari bbox prediksi")
    parser.add_argument("--save-csv", type=str, default=None, help="Simpan detail hasil per-objek ke CSV")
    parser.add_argument("--limit", type=int, default=None, help="Batasi jumlah file XML untuk uji cepat")
    args = parser.parse_args()

    xml_files = collect_xml_files(args.data_dir)
    if args.limit is not None:
        xml_files = xml_files[: args.limit]
    if not xml_files:
        raise FileNotFoundError(f"Tidak ada file XML ditemukan di: {args.data_dir}")

    system = EndToEndALPR(
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

    for xml_path in tqdm(xml_files, desc="Evaluating"):
        try:
            image_path, gts = parse_indian_lp_xml(xml_path)
        except Exception as exc:
            print(f"[WARN] Gagal parse XML {xml_path}: {exc}")
            continue

        if image_path is None or not os.path.exists(image_path):
            print(f"[WARN] Image pasangan untuk XML tidak ditemukan: {xml_path}")
            continue
        if not gts:
            continue

        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARN] Gagal membaca image: {image_path}")
            continue

        preds = system.predict(image)
        matches = greedy_match(gts, preds, args.iou_thresh)
        metrics.update(gts, preds, matches)

        matched_gt = {m.gt_idx for m in matches}
        matched_pred = {m.pred_idx for m in matches}

        for m in matches:
            gt = gts[m.gt_idx]
            pred = preds[m.pred_idx]
            rows.append({
                "xml_path": xml_path,
                "image_path": image_path,
                "gt_text": gt.text,
                "pred_text": pred.text,
                "gt_bbox": gt.bbox,
                "pred_bbox": pred.bbox,
                "iou": round(m.iou, 6),
                "det_conf": round(pred.det_conf, 6),
                "text_conf": round(pred.text_conf, 6),
                "bbox_match": 1,
                "text_exact": int(gt.text == pred.text),
                "e2e_exact": int(gt.text == pred.text),
                "cer": round(levenshtein(pred.text, gt.text) / max(len(gt.text), 1), 6),
            })

        for gi, gt in enumerate(gts):
            if gi in matched_gt:
                continue
            rows.append({
                "xml_path": xml_path,
                "image_path": image_path,
                "gt_text": gt.text,
                "pred_text": "",
                "gt_bbox": gt.bbox,
                "pred_bbox": "",
                "iou": 0.0,
                "det_conf": 0.0,
                "text_conf": 0.0,
                "bbox_match": 0,
                "text_exact": 0,
                "e2e_exact": 0,
                "cer": 1.0,
            })

        for pi, pred in enumerate(preds):
            if pi in matched_pred:
                continue
            rows.append({
                "xml_path": xml_path,
                "image_path": image_path,
                "gt_text": "",
                "pred_text": pred.text,
                "gt_bbox": "",
                "pred_bbox": pred.bbox,
                "iou": 0.0,
                "det_conf": round(pred.det_conf, 6),
                "text_conf": round(pred.text_conf, 6),
                "bbox_match": 0,
                "text_exact": 0,
                "e2e_exact": 0,
                "cer": "",
            })

    summary = metrics.summarize()

    print("\n===== HASIL EVALUASI END-TO-END =====")
    print(f"Images evaluated            : {summary['images']}")
    print(f"Total GT plates             : {summary['gt_total']}")
    print(f"Total predicted plates      : {summary['pred_total']}")
    print(f"Detection TP / FP / FN      : {summary['det_tp']} / {summary['det_fp']} / {summary['det_fn']}")
    print(f"Detection Precision         : {summary['det_precision'] * 100:.2f}%")
    print(f"Detection Recall            : {summary['det_recall'] * 100:.2f}%")
    print(f"Detection F1                : {summary['det_f1'] * 100:.2f}%")
    print(f"OCR Exact on matched boxes  : {summary['ocr_exact_on_matched'] * 100:.2f}%")
    print(f"OCR Char Acc on matched     : {summary['matched_char_acc'] * 100:.2f}%")
    print(f"Mean CER on matched         : {summary['mean_cer_on_matched'] * 100:.2f}%")
    print(f"End-to-End TP               : {summary['e2e_tp']}")
    print(f"End-to-End Precision        : {summary['e2e_precision'] * 100:.2f}%")
    print(f"End-to-End Recall           : {summary['e2e_recall'] * 100:.2f}%")
    print(f"End-to-End F1               : {summary['e2e_f1'] * 100:.2f}%")

    if args.save_csv:
        fieldnames = [
            "xml_path", "image_path", "gt_text", "pred_text", "gt_bbox", "pred_bbox",
            "iou", "det_conf", "text_conf", "bbox_match", "text_exact", "e2e_exact", "cer"
        ]
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nDetail hasil disimpan ke: {args.save_csv}")


if __name__ == "__main__":
    main()
