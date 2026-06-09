import re
import torch

CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}

PAD_IDX = len(CHARS)
EOS_IDX = len(CHARS) + 1
NUM_CLASSES = len(CHARS) + 2

MAX_PLATE_LEN = 10
MAX_SEQ_LEN = MAX_PLATE_LEN + 1  # extra 1 slot for EOS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def sanitize_plate_text(text: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(text).upper()).strip()


def encode_plate_text(text: str, max_plate_len: int = MAX_PLATE_LEN, max_seq_len: int = MAX_SEQ_LEN):
    text = sanitize_plate_text(text)
    token_ids = [CHAR2IDX[c] for c in text if c in CHAR2IDX][:max_plate_len]
    token_ids.append(EOS_IDX)
    if len(token_ids) < max_seq_len:
        token_ids += [PAD_IDX] * (max_seq_len - len(token_ids))
    else:
        token_ids = token_ids[:max_seq_len]
        token_ids[-1] = EOS_IDX
    return token_ids


def decode_token_ids(token_ids, stop_at_eos: bool = True, max_chars: int | None = None):
    chars = []
    for idx in token_ids:
        idx = int(idx)
        if idx == EOS_IDX and stop_at_eos:
            break
        if idx == PAD_IDX:
            continue
        if idx in IDX2CHAR:
            chars.append(IDX2CHAR[idx])
            if max_chars is not None and len(chars) >= max_chars:
                break
    return "".join(chars)
