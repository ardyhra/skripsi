import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from dataset_recog_only import CropDataset, train_crop_transform, val_crop_transform
from model_recog_only import PlateRecognizer
from config import NUM_CLASSES, DEVICE

def main():
    # 1. Dataset Baru (Unified)
    DATA_DIR = 'data_prepared/dataset_unified_recog'
    
    train_ds = CropDataset(
        root_dir=os.path.join(DATA_DIR, 'images'), 
        csv_file=os.path.join(DATA_DIR, 'train.csv'), # Kolom sekarang 'filename' & 'label' (cek dataset_recog_crop.py pastikan kolom ini sama)
        transform=train_crop_transform
    )
    val_ds = CropDataset(
        root_dir=os.path.join(DATA_DIR, 'images'), 
        csv_file=os.path.join(DATA_DIR, 'valid.csv'), 
        transform=val_crop_transform
    )
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    # 2. Init Model
    model = PlateRecognizer(NUM_CLASSES).to(DEVICE)
    
    # 3. Optimizer & Loss
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    EPOCHS = 30
    os.makedirs('rfdetr/checkpoints_pure_recog_merge', exist_ok=True)
    best_val_loss = float('inf')
    
    print("Mulai Training Plate Recognizer Murni dengan Dataset Super...")
    for epoch in range(EPOCHS):
        model.train()
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [TRAIN]")
        
        for crop_imgs, gt_labels in loop:
            crop_imgs, gt_labels = crop_imgs.to(DEVICE), gt_labels.to(DEVICE)
            optimizer.zero_grad()
            pred_chars = model(crop_imgs)
            loss = criterion(pred_chars.reshape(-1, NUM_CLASSES), gt_labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            loop.set_postfix(loss=loss.item())
            
        # --- VALIDASI ---
        model.eval()
        val_loss = 0
        correct, total = 0, 0
        with torch.no_grad():
            for crop_imgs, gt_labels in val_loader:
                crop_imgs, gt_labels = crop_imgs.to(DEVICE), gt_labels.to(DEVICE)
                pred_chars = model(crop_imgs)
                loss = criterion(pred_chars.reshape(-1, NUM_CLASSES), gt_labels.reshape(-1))
                val_loss += loss.item()
                
                # Cek Akurasi Karakter
                pred_idx = torch.argmax(pred_chars, dim=2)
                correct += (pred_idx == gt_labels).sum().item()
                total += gt_labels.numel()
                
        avg_val_loss = val_loss / len(val_loader)
        val_acc = (correct/total)*100
        print(f"--> Valid Loss: {avg_val_loss:.4f} | Valid Char Acc: {val_acc:.2f}%")
        
        # Simpan Model Terbaik
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'rfdetr/checkpoints_pure_recog_merge/best_recognizer.pth')
            print("    [!] Best Model Saved")

if __name__ == "__main__":
    main()