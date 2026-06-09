import torch
import torch.nn as nn
import torchvision.models as models

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].to(x.device)

class PlateRecognizer(nn.Module):
    def __init__(self, num_classes):
        super(PlateRecognizer, self).__init__()
        
        # 1. BACKBONE (ResNet18 sudah sangat cukup untuk gambar crop kecil)
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # Ambil sampai layer 3 (stride 16)
        self.backbone = nn.Sequential(*list(base.children())[:-3])
        self.backbone_dim = 256
        
        # 2. TRANSFORMER ENCODER
        self.hidden_dim = 256
        self.pos_enc = PositionalEncoding(self.hidden_dim, max_len=128) # Sesuai ukuran feature map
        
        enc_layer = nn.TransformerEncoderLayer(d_model=self.hidden_dim, nhead=8, dim_feedforward=512, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=3)
        
        # 3. OUTPUT HEAD (7 Karakter)
        self.char_head = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, x):
        # Input x: [Batch, 3, 64, 256] (Tinggi 64, Lebar 256)
        
        # Ekstrak Fitur
        features = self.backbone(x) # Output: [B, 256, 4, 16] (Karena stride 16)
        
        # Flatten spatial dimensions (Tinggi x Lebar)
        B, C, H, W = features.shape
        # [B, 256, 4, 16] -> [B, 256, 64] -> Permute: [B, 64, 256]
        seq = features.flatten(2).permute(0, 2, 1)
        
        # Tambahkan Positional Encoding & Masuk Transformer
        seq = self.pos_enc(seq)
        mem = self.transformer(seq) # [B, 64, 256]
        
        # Pooling ke 7 Karakter (Adaptive Average Pooling)
        mem = mem.permute(0, 2, 1) # [B, 256, 64]
        reduced = torch.nn.functional.adaptive_avg_pool1d(mem, 7) # [B, 256, 7]
        reduced = reduced.permute(0, 2, 1) # [B, 7, 256]
        
        # Prediksi Huruf
        pred_chars = self.char_head(reduced) # [B, 7, NumClasses]
        
        return pred_chars