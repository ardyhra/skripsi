import argparse
from collections import deque

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont

from rfdetr import RFDETRNano

from revised_pad_eos_config import DEVICE, IDX2CHAR, PAD_IDX, EOS_IDX, NUM_CLASSES, MAX_SEQ_LEN
from revised_pad_eos_model_recog_only import PlateRecognizer

RFDETR_WEIGHTS = "rfdetr/runs/rfdetr_nano_result/checkpoint_best_total.pth"
RECOG_WEIGHTS = "rfdetr/checkpoints_pure_recog_merge_pad_eos/best_recognizer_pad_eos.pth"
FONT_PATH = "simhei.ttf"
CROP_HEIGHT = 64
CROP_WIDTH = 256

crop_transform = A.Compose([
    A.Resize(CROP_HEIGHT, CROP_WIDTH),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def decode_text(pred_logits, max_chars=None, expected_chars=None):
    probs = F.softmax(pred_logits, dim=2)
    max_probs, pred_indices = torch.max(probs, dim=2)

    chars = []
    confidences = []
    for i, idx in enumerate(pred_indices[0]):
        idx = idx.item()
        prob = max_probs[0][i].item()
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


def draw_text(img, text, pos, color=(0, 255, 0)):
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(FONT_PATH, 30)
    except Exception:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


def get_iou(box1, box2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])
    if x_right <= x_left or y_bottom <= y_top:
        return 0.0
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - inter_area
    return inter_area / float(max(union, 1e-6))


class PlateTracker:
    def __init__(self, max_history=10, iou_thresh=0.3, recency_decay=0.9, min_obs_score=0.55, max_missed=6, max_chars=None, expected_chars=None):
        self.tracks = []
        self.max_history = max_history
        self.iou_thresh = iou_thresh
        self.recency_decay = recency_decay
        self.min_obs_score = min_obs_score
        self.max_missed = max_missed
        self.max_chars = max_chars
        self.expected_chars = expected_chars

    def _new_track(self, bbox):
        return {"bbox": bbox, "history": deque(maxlen=self.max_history), "missed": 0}

    def _effective_score(self, obs, age_idx, total):
        recency_weight = self.recency_decay ** (total - 1 - age_idx)
        return obs["score"] * recency_weight

    def _stabilize_text(self, track):
        history = list(track["history"])
        if not history:
            return "", 0.0

        total = len(history)
        length_votes = {}
        for age_idx, obs in enumerate(history):
            eff = self._effective_score(obs, age_idx, total)
            text_len = len(obs["text"])
            if self.expected_chars is not None:
                text_len = min(text_len, self.expected_chars)
            if self.max_chars is not None:
                text_len = min(text_len, self.max_chars)
            length_votes[text_len] = length_votes.get(text_len, 0.0) + eff

        chosen_len = max(length_votes.items(), key=lambda kv: kv[1])[0]
        if self.expected_chars is not None:
            chosen_len = self.expected_chars
        if self.max_chars is not None:
            chosen_len = min(chosen_len, self.max_chars)

        chars = []
        char_scores = []
        for pos in range(chosen_len):
            votes = {}
            for age_idx, obs in enumerate(history):
                if pos >= len(obs["text"]):
                    continue
                eff = self._effective_score(obs, age_idx, total)
                ch = obs["text"][pos]
                votes[ch] = votes.get(ch, 0.0) + eff
            if not votes:
                break
            best_char, best_score = max(votes.items(), key=lambda kv: kv[1])
            chars.append(best_char)
            char_scores.append(best_score)

        stabilized_text = "".join(chars)
        stabilized_score = sum(char_scores) / len(char_scores) if char_scores else 0.0
        return stabilized_text, stabilized_score

    def update(self, detections):
        assigned_tracks = set()
        updated_tracks = []

        for det in detections:
            bbox = det["bbox"]
            best_iou = 0.0
            best_idx = -1
            for idx, track in enumerate(self.tracks):
                if idx in assigned_tracks:
                    continue
                iou = get_iou(bbox, track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            if best_idx >= 0 and best_iou > self.iou_thresh:
                track = self.tracks[best_idx]
                assigned_tracks.add(best_idx)
            else:
                track = self._new_track(bbox)

            track["bbox"] = bbox
            track["missed"] = 0
            combined_score = 0.65 * det["text_conf"] + 0.35 * det["det_conf"]
            if det["text"] and combined_score >= self.min_obs_score:
                track["history"].append({"text": det["text"], "score": combined_score})
            updated_tracks.append(track)

        for idx, track in enumerate(self.tracks):
            if idx in assigned_tracks:
                continue
            track["missed"] += 1
            if track["missed"] <= self.max_missed:
                updated_tracks.append(track)

        self.tracks = updated_tracks

        stabilized_results = []
        for track in self.tracks:
            text, score = self._stabilize_text(track)
            if text:
                stabilized_results.append((track["bbox"], text, score))
        return stabilized_results


class HybridALPR:
    def __init__(self, det_weights, rec_weights, det_conf=0.5, rec_conf=0.55, max_chars=None, expected_chars=None, track_history=10):
        print("Loading RF-DETR Detector...")
        self.detector = RFDETRNano(classes=["license_plate"], pretrain_weights=det_weights)
        self.det_conf = det_conf
        self.rec_conf = rec_conf
        self.max_chars = max_chars
        self.expected_chars = expected_chars

        print("Loading PAD+EOS Plate Recognizer...")
        ckpt = torch.load(rec_weights, map_location=DEVICE, weights_only=False)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        num_classes = ckpt.get("num_classes", NUM_CLASSES) if isinstance(ckpt, dict) else NUM_CLASSES
        max_seq_len = ckpt.get("max_seq_len", MAX_SEQ_LEN) if isinstance(ckpt, dict) else MAX_SEQ_LEN

        self.recognizer = PlateRecognizer(num_classes=num_classes, max_seq_len=max_seq_len).to(DEVICE)
        self.recognizer.load_state_dict(state_dict, strict=False)
        self.recognizer.eval()

        self.tracker = PlateTracker(
            max_history=track_history,
            iou_thresh=0.3,
            recency_decay=0.9,
            min_obs_score=rec_conf,
            max_missed=6,
            max_chars=max_chars,
            expected_chars=expected_chars,
        )

    def process_frame(self, frame):
        h_orig, w_orig, _ = frame.shape
        predictions = self.detector.predict(frame, conf=self.det_conf)
        if len(predictions) == 0 or predictions.confidence is None or len(predictions.confidence) == 0:
            return frame

        detections = []
        for i in range(len(predictions.xyxy)):
            x1, y1, x2, y2 = map(int, predictions.xyxy[i])
            det_conf = float(predictions.confidence[i])

            pad_x = int((x2 - x1) * 0.08)
            pad_y = int((y2 - y1) * 0.12)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(w_orig, x2 + pad_x)
            y2 = min(h_orig, y2 + pad_y)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_tensor = crop_transform(image=crop_rgb)["image"].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred_chars = self.recognizer(crop_tensor)
                text, text_conf = decode_text(pred_chars, max_chars=self.max_chars, expected_chars=self.expected_chars)

            detections.append({
                "bbox": [x1, y1, x2, y2],
                "text": text,
                "text_conf": text_conf,
                "det_conf": det_conf,
            })

        stabilized_results = self.tracker.update(detections)

        for bbox, text, score in stabilized_results:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if score >= self.rec_conf else (0, 200, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            banner_w = max(160, 18 * max(len(text), 1))
            cv2.rectangle(frame, (x1, max(0, y1 - 35)), (min(frame.shape[1], x1 + banner_w), y1), (0, 0, 0), -1)
            frame = draw_text(frame, f"{text}", (x1, max(0, y1 - 30)), color=color)

        return frame


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path ke video atau gambar")
    parser.add_argument("--output", type=str, default="hasil_pad_eos.mp4", help="Path output file")
    parser.add_argument("--det-weights", type=str, default=RFDETR_WEIGHTS)
    parser.add_argument("--rec-weights", type=str, default=RECOG_WEIGHTS)
    parser.add_argument("--det-conf", type=float, default=0.5)
    parser.add_argument("--rec-conf", type=float, default=0.55)
    parser.add_argument("--max-chars", type=int, default=None, help="Batas maksimum karakter saat decode")
    parser.add_argument("--expected-chars", type=int, default=None, help="Pakai hanya jika Anda yakin panjang plat target tetap")
    parser.add_argument("--track-history", type=int, default=10)
    args = parser.parse_args()

    system = HybridALPR(
        det_weights=args.det_weights,
        rec_weights=args.rec_weights,
        det_conf=args.det_conf,
        rec_conf=args.rec_conf,
        max_chars=args.max_chars,
        expected_chars=args.expected_chars,
        track_history=args.track_history,
    )

    is_image = args.input.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp"))
    if is_image:
        frame = cv2.imread(args.input)
        if frame is None:
            raise FileNotFoundError(f"Gagal membaca input image: {args.input}")
        result = system.process_frame(frame)
        cv2.imwrite(args.output, result)
        print(f"Selesai. Output disimpan di {args.output}")
        return

    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        raise FileNotFoundError(f"Gagal membuka video: {args.input}")
    width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    fps = fps if fps > 0 else 25.0
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    print(f"Memproses video... Menyimpan ke {args.output}")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        res_frame = system.process_frame(frame)
        out.write(res_frame)
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Diproses {frame_count} frame...")

    cap.release()
    out.release()
    print("Selesai!")


if __name__ == "__main__":
    run()
