import math

import torch
import torch.nn as nn
import torchvision.models as models

from ctc_config import NUM_CLASSES


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


class PlateRecognizerCTC(nn.Module):
    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        hidden_dim: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-3])
        self.backbone_dim = 256

        self.proj = nn.Linear(self.backbone_dim, hidden_dim)
        self.pos_enc = PositionalEncoding(hidden_dim, max_len=256)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_encoder_layers)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor):
        features = self.backbone(x)          # [B, C, H, W]
        seq = features.mean(dim=2)           # [B, C, W]  <-- only collapse height
        seq = seq.permute(0, 2, 1)           # [B, W, C]  <-- preserve horizontal order
        seq = self.proj(seq)
        seq = self.pos_enc(seq)
        seq = self.encoder(seq)
        seq = self.dropout(seq)
        logits = self.classifier(seq)        # [B, T, C]
        input_lengths = torch.full((logits.size(0),), logits.size(1), dtype=torch.long, device=logits.device)
        return logits, input_lengths
