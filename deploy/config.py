# config.py
import os
import torch

# ================== PATH MODEL ==================
# Sesuaikan dengan lokasi file bobot Anda
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DETECTOR_WEIGHTS = os.path.join(BASE_DIR, "models", "checkpoint_best_total.pth")      # checkpoint RF-DETR
RECOGNIZER_WEIGHTS = os.path.join(BASE_DIR, "models", "best_recognizer_detector_crops.pth")  # checkpoint recognizer PAD+EOS

# ================== PARAMETER INFERENSI ==================
DETECTION_CONFIDENCE = 0.5
IOU_THRESH = 0.5
CROP_EXPAND_X = 0.08
CROP_EXPAND_Y = 0.12
TRACK_HISTORY = 10
RECOGNITION_CONFIDENCE = 0.55

# ================== KONFIGURASI MODEL RECOGNIZER ==================
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}
PAD_IDX = len(CHARS)          # 36
EOS_IDX = len(CHARS) + 1      # 37
NUM_CLASSES = len(CHARS) + 2  # 38
MAX_PLATE_LEN = 10
MAX_SEQ_LEN = MAX_PLATE_LEN + 1  # 11 (EOS di akhir)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CROP_HEIGHT = 64
CROP_WIDTH = 256