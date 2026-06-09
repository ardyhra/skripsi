# revised_config.py
import torch

# Vocabulary universal (angka + huruf latin kapital)
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}

# Token spesial untuk fixed-length decoding
PAD_IDX = len(CHARS)
NUM_CLASSES = len(CHARS) + 1

# Panjang label maksimum mengikuti statistik dataset gabungan.
# Dari test.csv terlihat label sampai 10 karakter.
MAX_PLATE_LEN = 10

# Ukuran crop recognizer
CROP_HEIGHT = 64
CROP_WIDTH = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sanitize_plate_text(text: str) -> str:
    text = "" if text is None else str(text).upper().strip()
    return "".join(ch for ch in text if ch in CHAR2IDX)
