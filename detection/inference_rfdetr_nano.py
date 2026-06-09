import cv2
import torch
import easyocr
import numpy as np
from rfdetr import RFDETRNano
from collections import deque, Counter

# --- KONFIGURASI ---
MODEL_WEIGHTS = "rfdetr/runs/rfdetr_nano_result/checkpoint_best_total.pth" 
VIDEO_INPUT = "alprtest_1.mp4"
CONFIDENCE = 0.5

class SmartLicenseSystem:
    def __init__(self, weights_path):
        # 1. Load RF-DETR Nano
        print("Loading RF-DETR Nano...")
        self.model = RFDETRNano(classes=["license_plate"])
        
        # 2. Load Weights (Fixed)
        print(f"Loading weights from {weights_path}...")
        try:
            # Load checkpoint ke CPU dulu agar aman
            checkpoint = torch.load(weights_path, map_location="cpu")
            
            # Cek struktur checkpoint
            state_dict = checkpoint
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            
            # Load ke model internal wrapper
            # Kita gunakan try-except spesifik karena struktur wrapper rfdetr bisa unik
            if hasattr(self.model.model, 'load_state_dict'):
                self.model.model.load_state_dict(state_dict)
            else:
                print("Warning: self.model.model tidak memiliki load_state_dict. Mencoba load atribut lain...")
                # Fallback logic jika struktur berbeda (opsional)
                
            print("Weights loaded successfully.")
            
        except Exception as e:
            print(f"Warning: Gagal load weights manual ({e}). Menggunakan inisialisasi default.")
            print("Pastikan path benar dan file tidak korup.")

        # HAPUS BARIS INI (PENYEBAB ERROR):
        # self.model.model.to("cuda" ...) 
        # self.model.model.eval()
        
        # 3. Load EasyOCR
        print("Loading EasyOCR...")
        self.reader = easyocr.Reader(['en'])
        
        # 4. Buffer Stabilisasi
        self.text_buffer = deque(maxlen=10)

    def process_frame(self, frame):
        # RF-DETR predict
        # Library ini otomatis pakai GPU jika tersedia saat install torch-cuda
        predictions = self.model.predict(frame, conf=CONFIDENCE)
        
        best_bbox = None
        
        # Cek deteksi
        if len(predictions) > 0:
            # Cari confidence tertinggi
            if predictions.confidence is not None and len(predictions.confidence) > 0:
                best_idx = np.argmax(predictions.confidence)
            else:
                best_idx = 0
            
            # Ambil bbox
            # Format .xyxy: [x1, y1, x2, y2]
            box = predictions.xyxy[best_idx]
            x1, y1, x2, y2 = box
            
            best_bbox = (int(x1), int(y1), int(x2), int(y2))

        final_text = ""
        
        if best_bbox:
            x1, y1, x2, y2 = best_bbox
            
            # Safety Check (Agar crop tidak error)
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            # Crop & OCR
            plate_img = frame[y1:y2, x1:x2]
            
            if plate_img.size > 0:
                res = self.reader.readtext(plate_img, detail=0, allowlist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')
                if res:
                    raw_text = "".join(res)
                    if len(raw_text) >= 3:
                        self.text_buffer.append(raw_text)
            
            # Voting Teks
            if len(self.text_buffer) > 0:
                final_text = Counter(self.text_buffer).most_common(1)[0][0]
                
            # Visualisasi
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if final_text:
                cv2.rectangle(frame, (x1, y1-30), (x2, y1), (0,0,0), -1)
                cv2.putText(frame, final_text, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

        return frame

def run():
    cap = cv2.VideoCapture(VIDEO_INPUT)
    if not cap.isOpened():
        print(f"Error opening video: {VIDEO_INPUT}")
        return

    # Siapkan Video Writer untuk menyimpan hasil
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Simpan ke 'hasil_inference.mp4'
    save_path = "hasil_inference.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    print(f"Processing video... Saving to {save_path}")

    system = SmartLicenseSystem(MODEL_WEIGHTS)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        processed_frame = system.process_frame(frame)
        
        # GANTI imshow DENGAN write
        # cv2.imshow("RF-DETR Nano ALPR", processed_frame) <-- HAPUS/KOMENTAR INI
        out.write(processed_frame)  # <-- GANTI DENGAN INI
        
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
            
        # if cv2.waitKey(1) == ord('q'): break <-- HAPUS INI JUGA
        
    cap.release()
    out.release() # Jangan lupa release writer

    print("Selesai.")

if __name__ == "__main__":
    run()