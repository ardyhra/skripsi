# alpr_core.py
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
from rfdetr import RFDETRNano
from collections import deque
import os

# Import konfigurasi
from config import (
    DEVICE, DETECTOR_WEIGHTS, RECOGNIZER_WEIGHTS,
    DETECTION_CONFIDENCE, CROP_EXPAND_X, CROP_EXPAND_Y,
    CROP_HEIGHT, CROP_WIDTH, NUM_CLASSES, MAX_SEQ_LEN,
    IDX2CHAR, PAD_IDX, EOS_IDX, CHARS
)

# Import model recognizer
from revised_pad_eos_model_recog_only import PlateRecognizer

crop_transform = A.Compose([
    A.Resize(CROP_HEIGHT, CROP_WIDTH),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

def decode_text(pred_logits, max_chars=None, expected_chars=None):
    """Decode output model PAD+EOS menjadi teks."""
    probs = F.softmax(pred_logits, dim=2)
    max_probs, pred_indices = torch.max(probs, dim=2)
    chars = []
    confidences = []
    for i, idx in enumerate(pred_indices[0]):
        idx_val = idx.item()
        prob = max_probs[0][i].item()
        if idx_val == EOS_IDX:
            break
        if idx_val == PAD_IDX:
            continue
        if idx_val in IDX2CHAR:
            chars.append(IDX2CHAR[idx_val])
            confidences.append(prob)
            if expected_chars is not None and len(chars) >= expected_chars:
                break
            if max_chars is not None and len(chars) >= max_chars:
                break
    text = "".join(chars)
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
    return text, avg_conf

def debug_predictions(pred_logits, gt_text=None):
    """Fungsi debug untuk melihat output model."""
    probs = F.softmax(pred_logits, dim=2)
    pred_indices = torch.argmax(probs, dim=2)
    print("DEBUG - pred_indices shape:", pred_indices.shape)
    print("DEBUG - pred_indices[0]:", pred_indices[0].tolist())
    print("DEBUG - EOS_IDX:", EOS_IDX, "PAD_IDX:", PAD_IDX)
    print("DEBUG - IDX2CHAR mapping:", {k: v for k, v in IDX2CHAR.items() if k < 40})
    # Hitung distribusi prediksi
    all_idx = pred_indices[0].tolist()
    from collections import Counter
    print("DEBUG - Frekuensi indeks:", Counter(all_idx))
    if gt_text:
        print("DEBUG - Ground truth:", gt_text)

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
    return inter_area / max(union, 1e-6)

class PlateTracker:
    def __init__(self, max_history=10, iou_thresh=0.3, recency_decay=0.9, min_obs_score=0.55, max_missed=6):
        self.tracks = []
        self.max_history = max_history
        self.iou_thresh = iou_thresh
        self.recency_decay = recency_decay
        self.min_obs_score = min_obs_score
        self.max_missed = max_missed

    def _effective_score(self, obs, age_idx, total):
        return obs["score"] * (self.recency_decay ** (total - 1 - age_idx))

    def _stabilize_text(self, track):
        history = list(track["history"])
        if not history:
            return "", 0.0
        total = len(history)
        length_votes = {}
        for age_idx, obs in enumerate(history):
            eff = self._effective_score(obs, age_idx, total)
            length_votes[len(obs["text"])] = length_votes.get(len(obs["text"]), 0.0) + eff
        chosen_len = max(length_votes.items(), key=lambda kv: kv[1])[0]
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
                track = {"bbox": bbox, "history": deque(maxlen=self.max_history), "missed": 0}
            track["bbox"] = bbox
            track["missed"] = 0
            combined_score = 0.65 * det["text_conf"] + 0.35 * det["det_conf"]
            if det["text"] and combined_score >= self.min_obs_score:
                track["history"].append({"text": det["text"], "score": combined_score})
            updated_tracks.append(track)

        for idx, track in enumerate(self.tracks):
            if idx not in assigned_tracks:
                track["missed"] += 1
                if track["missed"] <= self.max_missed:
                    updated_tracks.append(track)

        self.tracks = updated_tracks
        stabilized = []
        for track in self.tracks:
            text, score = self._stabilize_text(track)
            if text:
                stabilized.append((track["bbox"], text, score))
        return stabilized

class ALPRSystem:
    def __init__(self, detector_weights=None, recognizer_weights=None, debug=False):
        self.debug = debug
        if detector_weights is None:
            detector_weights = DETECTOR_WEIGHTS
        if recognizer_weights is None:
            recognizer_weights = RECOGNIZER_WEIGHTS

        # Load detector
        print(f"Loading RF-DETR detector from {detector_weights}...")
        self.detector = RFDETRNano(classes=["license_plate"], pretrain_weights=detector_weights)

        # Load recognizer
        print(f"Loading recognizer from {recognizer_weights}...")
        if not os.path.exists(recognizer_weights):
            raise FileNotFoundError(f"Recognizer weights not found: {recognizer_weights}")
        ckpt = torch.load(recognizer_weights, map_location=DEVICE, weights_only=False)
        print(f"Checkpoint keys: {ckpt.keys() if isinstance(ckpt, dict) else 'direct state_dict'}")
        
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
            # Cek apakah checkpoint menyimpan num_classes
            self.num_classes = ckpt.get("num_classes", NUM_CLASSES)
            self.max_seq_len = ckpt.get("max_seq_len", MAX_SEQ_LEN)
        else:
            state_dict = ckpt
            self.num_classes = NUM_CLASSES
            self.max_seq_len = MAX_SEQ_LEN
        
        print(f"Using num_classes={self.num_classes}, max_seq_len={self.max_seq_len}")
        
        self.recognizer = PlateRecognizer(num_classes=self.num_classes, max_seq_len=self.max_seq_len).to(DEVICE)
        self.recognizer.load_state_dict(state_dict, strict=False)
        self.recognizer.eval()
        print("Recognizer loaded successfully.")

        self.tracker = PlateTracker(
            max_history=10, iou_thresh=0.3, min_obs_score=0.55
        )

    def process_frame(self, frame, use_tracker=True):
        h_orig, w_orig = frame.shape[:2]
        predictions = self.detector.predict(frame, conf=DETECTION_CONFIDENCE)
        detections = []
        
        if len(predictions) > 0 and predictions.confidence is not None:
            for i in range(len(predictions.xyxy)):
                x1, y1, x2, y2 = map(int, predictions.xyxy[i])
                det_conf = float(predictions.confidence[i])
                pad_x = int((x2 - x1) * CROP_EXPAND_X)
                pad_y = int((y2 - y1) * CROP_EXPAND_Y)
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
                    if self.debug:
                        debug_predictions(pred_chars)
                    text, text_conf = decode_text(pred_chars)
                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "text": text,
                    "text_conf": text_conf,
                    "det_conf": det_conf,
                })
                if self.debug and text:
                    print(f"DEBUG - Recognized: '{text}' (conf={text_conf:.4f})")

        if use_tracker:
            stabilized = self.tracker.update(detections)
        else:
            stabilized = [(d["bbox"], d["text"], d["text_conf"]) for d in detections if d["text"]]

        annotated = frame.copy()
        results = []
        for bbox, text, score in stabilized:
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if score >= 0.55 else (0, 200, 255)
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            # Background teks
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
            y_text = max(y1 - 5, th + 5)
            cv2.rectangle(annotated, (x1, y_text - th - 2), (x1 + tw + 5, y_text + 2), (0, 0, 0), -1)
            cv2.putText(annotated, text, (x1, y_text), font, font_scale, color, thickness)
            results.append({"bbox": bbox, "text": text, "confidence": score})
        return annotated, results