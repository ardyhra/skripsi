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
from model_recog_only import PlateRecognizer
from config import NUM_CLASSES, DEVICE, IDX2CHAR

RFDETR_WEIGHTS = 'rfdetr/runs/rfdetr_nano_result/checkpoint_best_total.pth'
RECOG_WEIGHTS = 'rfdetr/checkpoints_pure_recog_merge/best_recognizer.pth'
FONT_PATH = 'simhei.ttf'
CROP_HEIGHT = 64
CROP_WIDTH = 256
MAX_MODEL_LEN = 7

crop_transform = A.Compose([
    A.Resize(CROP_HEIGHT, CROP_WIDTH),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])


def trim_by_confidence(chars, confs, max_chars=None, tail_drop=0.18, min_tail_conf=0.45):
    if not chars:
        return '', 0.0
    keep = len(chars)
    for i in range(len(chars) - 1, 0, -1):
        cur = confs[i]
        prev = confs[i - 1]
        if cur < min_tail_conf or (prev - cur) > tail_drop:
            keep = i
        else:
            break
    if max_chars is not None:
        keep = min(keep, max_chars)
    keep = max(1, keep)
    out_chars = chars[:keep]
    out_confs = confs[:keep]
    return ''.join(out_chars), sum(out_confs) / len(out_confs)


def decode_text(pred_logits, max_chars=None):
    probs = F.softmax(pred_logits, dim=2)
    max_probs, pred_indices = torch.max(probs, dim=2)
    chars = []
    confs = []
    for i, idx in enumerate(pred_indices[0]):
        idx = idx.item()
        prob = max_probs[0][i].item()
        if idx in IDX2CHAR:
            chars.append(IDX2CHAR[idx])
            confs.append(prob)
    text, avg_conf = trim_by_confidence(chars, confs, max_chars=max_chars)
    min_conf = min(confs[:len(text)]) if text else 0.0
    return text, avg_conf, min_conf


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
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    inter_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return inter_area / float(box1_area + box2_area - inter_area + 1e-6)


class PlateTracker:
    def __init__(self, max_history=7, iou_thresh=0.3, min_accept_conf=0.65, switch_margin=1.20, switch_frames=3):
        self.tracks = []
        self.max_history = max_history
        self.iou_thresh = iou_thresh
        self.min_accept_conf = min_accept_conf
        self.switch_margin = switch_margin
        self.switch_frames = switch_frames

    def _aggregate(self, history):
        pos_scores = [defaultdict(float) for _ in range(MAX_MODEL_LEN)]
        for item in history:
            for i, ch in enumerate(item['text'][:MAX_MODEL_LEN]):
                pos_scores[i][ch] += item['weight']
        chars = []
        score_sum = 0.0
        for d in pos_scores:
            if not d:
                break
            ch, sc = max(d.items(), key=lambda kv: kv[1])
            chars.append(ch)
            score_sum += sc
        return ''.join(chars), score_sum

    def update(self, detections):
        updated = []
        remaining = self.tracks[:]
        for det in detections:
            bbox = det['bbox']
            best_iou, best_idx = 0.0, -1
            for j, track in enumerate(remaining):
                iou = get_iou(bbox, track['bbox'])
                if iou > best_iou:
                    best_iou, best_idx = iou, j
            if best_iou > self.iou_thresh and best_idx >= 0:
                track = remaining.pop(best_idx)
            else:
                track = {'bbox': bbox, 'history': deque(maxlen=self.max_history), 'best_text': '', 'best_score': 0.0, 'candidate_text': '', 'candidate_count': 0}
            track['bbox'] = bbox
            text = det['text']
            weight = det['weight']
            if text and weight >= self.min_accept_conf:
                track['history'].append({'text': text, 'weight': weight})
                agg_text, agg_score = self._aggregate(track['history'])
                if not track['best_text']:
                    track['best_text'], track['best_score'] = agg_text, agg_score
                elif agg_text == track['best_text']:
                    track['best_score'] = max(track['best_score'], agg_score)
                    track['candidate_text'], track['candidate_count'] = '', 0
                elif agg_score > track['best_score'] * self.switch_margin:
                    if agg_text == track['candidate_text']:
                        track['candidate_count'] += 1
                    else:
                        track['candidate_text'], track['candidate_count'] = agg_text, 1
                    if track['candidate_count'] >= self.switch_frames:
                        track['best_text'], track['best_score'] = agg_text, agg_score
                        track['candidate_text'], track['candidate_count'] = '', 0
            updated.append(track)
        self.tracks = updated
        return [(t['bbox'], t['best_text']) for t in self.tracks if t['best_text']]


class HybridALPR:
    def __init__(self, det_weights, rec_weights, max_chars=None):
        self.detector = RFDETRNano(classes=['license_plate'], pretrain_weights=det_weights)
        self.recognizer = PlateRecognizer(NUM_CLASSES).to(DEVICE)
        ckpt_rec = torch.load(rec_weights, map_location=DEVICE, weights_only=False)
        if 'state_dict' in ckpt_rec:
            self.recognizer.load_state_dict(ckpt_rec['state_dict'], strict=False)
        else:
            self.recognizer.load_state_dict(ckpt_rec, strict=False)
        self.recognizer.eval()
        self.tracker = PlateTracker()
        self.max_chars = max_chars

    def process_frame(self, frame):
        h_orig, w_orig, _ = frame.shape
        predictions = self.detector.predict(frame, conf=0.5)
        if len(predictions) == 0 or predictions.confidence is None or len(predictions.confidence) == 0:
            return frame
        detections = []
        for i in range(len(predictions.xyxy)):
            det_conf = float(predictions.confidence[i]) if predictions.confidence is not None else 1.0
            x1, y1, x2, y2 = map(int, predictions.xyxy[i])
            pad_x = int((x2 - x1) * 0.08)
            pad_y = int((y2 - y1) * 0.12)
            x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
            x2, y2 = min(w_orig, x2 + pad_x), min(h_orig, y2 + pad_y)
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_tensor = crop_transform(image=crop_rgb)['image'].unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                pred_chars = self.recognizer(crop_tensor)
                text, avg_conf, min_conf = decode_text(pred_chars, max_chars=self.max_chars)
            if len(text) < 4:
                continue
            weight = 0.65 * avg_conf + 0.20 * min_conf + 0.15 * det_conf
            detections.append({'bbox': [x1, y1, x2, y2], 'text': text, 'weight': weight})
        results = self.tracker.update(detections)
        for bbox, text in results:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_w = max(150, 20 + 18 * len(text))
            cv2.rectangle(frame, (x1, max(0, y1 - 35)), (min(frame.shape[1], x1 + label_w), y1), (0, 0, 0), -1)
            frame = draw_text(frame, text, (x1, max(0, y1 - 30)))
        return frame


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output', type=str, default='hasil_currentmodel_tuned.mp4')
    parser.add_argument('--max-chars', type=int, default=None, help='Batasi panjang karakter output pada model lama')
    args = parser.parse_args()
    system = HybridALPR(RFDETR_WEIGHTS, RECOG_WEIGHTS, max_chars=args.max_chars)
    ext = args.input.lower().split('.')[-1]
    if ext in ['jpg', 'jpeg', 'png', 'bmp', 'webp']:
        frame = cv2.imread(args.input)
        res = system.process_frame(frame)
        cv2.imwrite(args.output, res)
        print(f'Hasil disimpan ke {args.output}')
        return
    cap = cv2.VideoCapture(args.input)
    width, height, fps = int(cap.get(3)), int(cap.get(4)), cap.get(5)
    fps = fps if fps > 0 else 25
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        out.write(system.process_frame(frame))
    cap.release()
    out.release()
    print(f'Selesai. Output: {args.output}')


if __name__ == '__main__':
    run()
