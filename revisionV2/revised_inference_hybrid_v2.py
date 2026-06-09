import argparse
from collections import deque, defaultdict

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image, ImageDraw, ImageFont
from rfdetr import RFDETRNano

from revised_model_recog_only import PlateRecognizer
from revised_config import (
    NUM_CLASSES,
    DEVICE,
    IDX2CHAR,
    PAD_IDX,
    CROP_HEIGHT,
    CROP_WIDTH,
    MAX_PLATE_LEN,
)

RFDETR_WEIGHTS = "rfdetr/runs/rfdetr_nano_result/checkpoint_best_total.pth"
RECOG_WEIGHTS = "rfdetr/checkpoints_pure_recog_merge_v2/best_recognizer.pth"
FONT_PATH = "simhei.ttf"

crop_transform = A.Compose([
    A.Resize(CROP_HEIGHT, CROP_WIDTH),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def decode_text(pred_logits: torch.Tensor):
    probs = F.softmax(pred_logits, dim=2)
    max_probs, pred_indices = torch.max(probs, dim=2)

    chars = []
    char_confs = []
    for idx, prob in zip(pred_indices[0], max_probs[0]):
        idx = idx.item()
        prob = prob.item()
        if idx == PAD_IDX:
            continue
        if idx in IDX2CHAR:
            chars.append(IDX2CHAR[idx])
            char_confs.append(prob)

    text = "".join(chars).strip()
    avg_conf = float(sum(char_confs) / len(char_confs)) if char_confs else 0.0
    return text, avg_conf, char_confs


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
    box1_area = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    box2_area = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))
    return inter_area / float(box1_area + box2_area - inter_area)


class PlateTracker:
    def __init__(self, max_history=20, iou_thresh=0.3, conf_thresh=0.55):
        self.tracks = []
        self.max_history = max_history
        self.iou_thresh = iou_thresh
        self.conf_thresh = conf_thresh

    def _vote_text(self, records):
        pos_scores = [defaultdict(float) for _ in range(MAX_PLATE_LEN)]
        length_scores = defaultdict(float)

        for record in records:
            text = record["text"]
            conf = record["conf"]
            length_scores[len(text)] += conf
            for pos, ch in enumerate(text[:MAX_PLATE_LEN]):
                pos_scores[pos][ch] += conf

        voted_len = max(length_scores.items(), key=lambda x: x[1])[0] if length_scores else 0
        out = []
        for pos in range(voted_len):
            if not pos_scores[pos]:
                continue
            out.append(max(pos_scores[pos].items(), key=lambda x: x[1])[0])
        return "".join(out)

    def update(self, current_items):
        updated_tracks = []
        for item in current_items:
            bbox = item["bbox"]
            best_iou = 0.0
            best_idx = -1
            for j, track in enumerate(self.tracks):
                iou = get_iou(bbox, track["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_iou > self.iou_thresh:
                track = self.tracks.pop(best_idx)
                track["bbox"] = bbox
                if item["text"] and item["conf"] >= self.conf_thresh:
                    track["records"].append({"text": item["text"], "conf": item["conf"]})
                updated_tracks.append(track)
            else:
                new_track = {"bbox": bbox, "records": deque(maxlen=self.max_history)}
                if item["text"] and item["conf"] >= self.conf_thresh:
                    new_track["records"].append({"text": item["text"], "conf": item["conf"]})
                updated_tracks.append(new_track)

        self.tracks = updated_tracks
        stabilized_results = []
        for track in self.tracks:
            voted_text = self._vote_text(track["records"])
            if voted_text:
                stabilized_results.append((track["bbox"], voted_text))
        return stabilized_results


class HybridALPR:
    def __init__(self, det_weights, rec_weights):
        print("Loading RF-DETR Detector...")
        self.detector = RFDETRNano(classes=["license_plate"], pretrain_weights=det_weights)

        print("Loading Dedicated Plate Recognizer...")
        self.recognizer = PlateRecognizer(NUM_CLASSES).to(DEVICE)
        ckpt_rec = torch.load(rec_weights, map_location=DEVICE, weights_only=False)
        state_dict = ckpt_rec["state_dict"] if isinstance(ckpt_rec, dict) and "state_dict" in ckpt_rec else ckpt_rec
        self.recognizer.load_state_dict(state_dict, strict=True)
        self.recognizer.eval()
        self.tracker = PlateTracker(max_history=20, iou_thresh=0.3, conf_thresh=0.55)

    def process_frame(self, frame):
        h_orig, w_orig, _ = frame.shape
        predictions = self.detector.predict(frame, conf=0.5)
        if len(predictions) == 0 or predictions.confidence is None or len(predictions.confidence) == 0:
            return frame

        current_items = []
        for box, det_conf in zip(predictions.xyxy, predictions.confidence):
            x1, y1, x2, y2 = map(int, box)

            pad_x = int((x2 - x1) * 0.08)
            pad_y = int((y2 - y1) * 0.12)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(w_orig, x2 + pad_x), min(h_orig, y2 + pad_y)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_tensor = crop_transform(image=crop_rgb)["image"].unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                pred_chars = self.recognizer(crop_tensor)
                text, text_conf, char_confs = decode_text(pred_chars)

            # Filter sederhana agar frame buruk tidak ikut memengaruhi voting.
            if not text or len(text) < 4:
                continue
            if text_conf < 0.35:
                continue

            current_items.append({
                "bbox": [x1, y1, x2, y2],
                "text": text,
                "conf": float(text_conf * float(det_conf)),
                "char_confs": char_confs,
            })

        stabilized_results = self.tracker.update(current_items)

        for bbox, text in stabilized_results:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, max(0, y1 - 35)), (x1 + 220, y1), (0, 0, 0), -1)
            frame = draw_text(frame, text, (x1, max(0, y1 - 30)))
        return frame



def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path ke video (.mp4)")
    parser.add_argument("--output", type=str, default="hasil_skripsi_final_v2.mp4", help="Path output file")
    args = parser.parse_args()

    system = HybridALPR(RFDETR_WEIGHTS, RECOG_WEIGHTS)
    cap = cv2.VideoCapture(args.input)
    width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    print(f"Memproses video... Menyimpan ke {args.output}")
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(system.process_frame(frame))
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Diproses {frame_count} frame...")

    cap.release()
    out.release()
    print("Selesai! Silakan cek file video output.")


if __name__ == "__main__":
    run()
