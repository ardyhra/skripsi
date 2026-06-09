# config.py

# Vocabulary Universal (Angka + Huruf)
# Tambahkan karakter spesial jika perlu (misal '-')
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR2IDX = {c: i for i, c in enumerate(CHARS)}
IDX2CHAR = {i: c for i, c in enumerate(CHARS)}
NUM_CLASSES = len(CHARS) + 1 # +1 untuk blank token (opsional, tapi bagus untuk CTC/Transformer)


DEVICE = "cuda"