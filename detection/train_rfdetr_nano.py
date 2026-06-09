from rfdetr import RFDETRNano
import os

def train():
    # 1. Inisialisasi Model Nano
    # Kita definisikan kelas secara eksplisit
    print("Initializing RF-DETR Nano...")
    model = RFDETRNano(
        classes=["license_plate"]  # Penting: definisikan nama kelas Anda
    )
    
    # 2. Mulai Training
    # RF-DETR akan otomatis mencari _annotations.coco.json di folder dataset_dir
    print("Starting Training...")
    model.train(
        dataset_dir="data_prepared/lprdataset_rfdetr_coco",  # Folder hasil konversi tadi
        epochs=10,
        batch_size=4,        # Sesuaikan dengan VRAM (4GB coba 4 atau 8)
        grad_accum_steps=4,  # Akumulasi gradien untuk stabilisasi batch kecil
        learning_rate=5e-4,  # LR standar untuk Nano
        output_dir="rfdetr/runs/rfdetr_nano_result",
        device="cuda"
    )

if __name__ == "__main__":
    train()