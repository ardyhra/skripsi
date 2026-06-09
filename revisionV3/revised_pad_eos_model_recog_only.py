import math
import torch
import torch.nn as nn
import torchvision.models as models

from revised_pad_eos_config import MAX_SEQ_LEN, NUM_CLASSES


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class PlateRecognizer(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES, max_seq_len: int = MAX_SEQ_LEN, hidden_dim: int = 256):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.hidden_dim = hidden_dim

        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(base.children())[:-3])
        self.backbone_dim = 256

        self.pos_enc = PositionalEncoding(self.hidden_dim, max_len=256)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        self.dropout = nn.Dropout(0.1)
        self.char_head = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        seq = features.flatten(2).permute(0, 2, 1)
        seq = self.pos_enc(seq)
        mem = self.transformer(seq)
        mem = mem.permute(0, 2, 1)
        reduced = torch.nn.functional.adaptive_avg_pool1d(mem, self.max_seq_len)
        reduced = reduced.permute(0, 2, 1)
        reduced = self.dropout(reduced)
        pred_chars = self.char_head(reduced)
        return pred_chars
