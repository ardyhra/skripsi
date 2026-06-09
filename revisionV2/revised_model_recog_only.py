import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from revised_config import MAX_PLATE_LEN


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.size(1)]


class PlateRecognizer(nn.Module):
    def __init__(self, num_classes: int, max_len: int = MAX_PLATE_LEN, dropout: float = 0.1):
        super().__init__()
        self.max_len = max_len

        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-3])
        self.backbone_dim = 256
        self.hidden_dim = 256

        self.pos_enc = PositionalEncoding(self.hidden_dim, max_len=512)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.dropout = nn.Dropout(dropout)
        self.char_head = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)           # [B, 256, H, W]
        seq = features.flatten(2).permute(0, 2, 1)  # [B, HW, 256]
        seq = self.pos_enc(seq)
        mem = self.transformer(seq)           # [B, HW, 256]

        # Tetap fixed-length, tetapi panjang sekarang mengikuti dataset (10)
        mem = mem.permute(0, 2, 1)            # [B, 256, HW]
        reduced = F.adaptive_avg_pool1d(mem, self.max_len)
        reduced = reduced.permute(0, 2, 1)    # [B, max_len, 256]
        reduced = self.dropout(reduced)
        pred_chars = self.char_head(reduced)  # [B, max_len, num_classes]
        return pred_chars
