import re
from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}

BLANK_IDX = len(CHARS)
NUM_CLASSES = len(CHARS) + 1
MAX_PLATE_LEN = 10

CROP_HEIGHT = 64
CROP_WIDTH = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sanitize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper()).strip()


def encode_plate_text(text: str, max_plate_len: int = MAX_PLATE_LEN) -> List[int]:
    text = sanitize_plate_text(text)
    return [CHAR2IDX[c] for c in text if c in CHAR2IDX][:max_plate_len]


def decode_ctc_indices(indices: Sequence[int], blank_idx: int = BLANK_IDX, max_chars: Optional[int] = None) -> str:
    chars: List[str] = []
    prev_idx: Optional[int] = None
    for idx in indices:
        idx = int(idx)
        if idx == blank_idx:
            prev_idx = idx
            continue
        if idx == prev_idx:
            continue
        if idx in IDX2CHAR:
            chars.append(IDX2CHAR[idx])
            if max_chars is not None and len(chars) >= max_chars:
                break
        prev_idx = idx
    return "".join(chars)


def ctc_greedy_decode(logits: torch.Tensor, blank_idx: int = BLANK_IDX, max_chars: Optional[int] = None) -> Tuple[List[str], List[float]]:
    """
    logits: [B, T, C]
    returns: (texts, confidences)
    """
    probs = torch.softmax(logits, dim=-1)
    max_probs, pred_idx = probs.max(dim=-1)

    texts: List[str] = []
    confs: List[float] = []
    for seq_idx, seq in enumerate(pred_idx):
        chars: List[str] = []
        kept_probs: List[float] = []
        prev_idx: Optional[int] = None
        for t, idx in enumerate(seq.tolist()):
            prob = float(max_probs[seq_idx, t].item())
            if idx == blank_idx:
                prev_idx = idx
                continue
            if idx == prev_idx:
                continue
            if idx in IDX2CHAR:
                chars.append(IDX2CHAR[idx])
                kept_probs.append(prob)
                if max_chars is not None and len(chars) >= max_chars:
                    break
            prev_idx = idx
        texts.append("".join(chars))
        confs.append(sum(kept_probs) / len(kept_probs) if kept_probs else 0.0)
    return texts, confs


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


def letterbox_plate_image(image: np.ndarray, target_h: int = CROP_HEIGHT, target_w: int = CROP_WIDTH, pad_value: int = 0) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return np.zeros((target_h, target_w, 3), dtype=np.uint8)

    scale = min(target_w / float(w), target_h / float(h))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_h, target_w, 3), pad_value, dtype=np.uint8)

    x0 = (target_w - new_w) // 2
    y0 = (target_h - new_h) // 2
    canvas[y0:y0 + new_h, x0:x0 + new_w] = resized
    return canvas
