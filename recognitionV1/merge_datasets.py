import os
import cv2
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm

# ==========================================
# KONFIGURASI PATH ASAL (SESUAIKAN FOLDER ANDA)
# ==========================================
DIR_YAZDANI = 'data_prepared/yazdani_lp'         # Folder Yazdani lama (ada lpr.csv & cropped_lps)
DIR_DATASET_1 = 'data_prepared/european_lp'     # Folder dataset 1 (nama file = label, misal 5B40001.png)
DIR_DATASET_2 = 'data_prepared/indian_lp'      # Folder dataset 2 (campur image & xml / folder terpisah)

# ==========================================
# KONFIGURASI PATH OUTPUT (DATASET GABUNGAN BARU)
# ==========================================
OUT_DIR = 'data_prepared/dataset_unified_recog'
OUT_IMG_DIR = os.path.join(OUT_DIR, 'images')

os.makedirs(OUT_IMG_DIR, exist_ok=True)

def clean_label(text):
    # Hanya ambil Huruf Besar dan Angka, buang spasi/simbol
    return re.sub(r'[^A-Z0-9]', '', str(text).upper())

def main():
    data_records = []
    image_counter = 0 # Untuk memberi nama unik pada gambar baru agar tidak bentrok
    
    # ---------------------------------------------------------
    # 1. PROSES DATASET YAZDANI LAMA
    # ---------------------------------------------------------
    print("Memproses Dataset Yazdani...")
    if os.path.exists(DIR_YAZDANI):
        df_yazdani = pd.read_csv(os.path.join(DIR_YAZDANI, 'lpr.csv'))
        for _, row in tqdm(df_yazdani.iterrows(), total=len(df_yazdani)):
            img_name = row['images']
            label = clean_label(row['labels'])
            
            src_path = os.path.join(DIR_YAZDANI, 'cropped_lps', img_name)
            if not os.path.exists(src_path) or len(label) == 0: continue
            
            # Copy dengan nama baru
            new_img_name = f"yz_{image_counter}.jpg"
            shutil.copy(src_path, os.path.join(OUT_IMG_DIR, new_img_name))
            
            data_records.append({'filename': new_img_name, 'label': label})
            image_counter += 1

    # ---------------------------------------------------------
    # 2. PROSES DATASET 1 (Sudah Crop, Label di Nama File)
    # ---------------------------------------------------------
    print("Memproses Dataset 1 (Label di Nama File)...")
    if os.path.exists(DIR_DATASET_1):
        # Kalau sebelumnya ini dibagi folder train/val/test, Anda bisa arahkan script
        # untuk membaca semua subfoldernya. Misal kita baca semua file:
        for root, dirs, files in os.walk(DIR_DATASET_1):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label_raw = os.path.splitext(file)[0] # Ambil nama file tanpa ekstensi
                    label = clean_label(label_raw)
                    
                    src_path = os.path.join(root, file)
                    if len(label) == 0: continue
                    
                    new_img_name = f"ds1_{image_counter}.jpg"
                    shutil.copy(src_path, os.path.join(OUT_IMG_DIR, new_img_name))
                    
                    data_records.append({'filename': new_img_name, 'label': label})
                    image_counter += 1

    # ---------------------------------------------------------
    # 3. PROSES DATASET 2 (Belum Crop, Format XML)
    # ---------------------------------------------------------
    print("Memproses Dataset 2 (Format XML)...")
    if os.path.exists(DIR_DATASET_2):
        # Asumsi gambar dan XML ada di dalam folder yang sama/subfolder
        for root, dirs, files in os.walk(DIR_DATASET_2):
            for file in tqdm(files):
                if file.lower().endswith('.xml'):
                    xml_path = os.path.join(root, file)
                    
                    try:
                        tree = ET.parse(xml_path)
                        xml_root = tree.getroot()
                        
                        # Cari nama file gambar (biasanya sama dengan nama xml)
                        # Kita cari file gambarnya di folder yang sama
                        img_filename = os.path.splitext(file)[0] + '.jpg'
                        img_path = os.path.join(root, img_filename)
                        
                        # Fallback cek ekstensi lain jika .jpg tidak ada
                        if not os.path.exists(img_path):
                            img_path = os.path.join(root, os.path.splitext(file)[0] + '.png')
                            
                        if not os.path.exists(img_path):
                            img_path = os.path.join(root, os.path.splitext(file)[0] + '.jpeg')
                            
                        if not os.path.exists(img_path): continue

                        # Buka gambar aslinya
                        img = cv2.imread(img_path)
                        if img is None: continue

                        # Ekstrak semua objek plat nomor
                        for obj in xml_root.findall('object'):
                            label_raw = obj.find('name').text
                            label = clean_label(label_raw)
                            
                            if len(label) == 0: continue
                            
                            bbox = obj.find('bndbox')
                            xmin = int(float(bbox.find('xmin').text))
                            ymin = int(float(bbox.find('ymin').text))
                            xmax = int(float(bbox.find('xmax').text))
                            ymax = int(float(bbox.find('ymax').text))
                            
                            # CROP GAMBAR
                            crop_img = img[ymin:ymax, xmin:xmax]
                            if crop_img.size == 0: continue
                            
                            new_img_name = f"ds2_{image_counter}.jpg"
                            cv2.imwrite(os.path.join(OUT_IMG_DIR, new_img_name), crop_img)
                            
                            data_records.append({'filename': new_img_name, 'label': label})
                            image_counter += 1
                    except Exception as e:
                        print(f"Error reading {xml_path}: {e}")

    # ---------------------------------------------------------
    # 4. SPLIT DATASET (Train 80%, Valid 10%, Test 10%)
    # ---------------------------------------------------------
    print(f"\nTotal Data Terkumpul: {len(data_records)} Plat Nomor")
    
    df_all = pd.DataFrame(data_records)
    
    # Split Train (80%) dan Sisa (20%)
    df_train, df_temp = train_test_split(df_all, test_size=0.2, random_state=42)
    # Split Sisa menjadi Valid (10%) dan Test (10%)
    df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)
    
    df_train.to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
    df_valid.to_csv(os.path.join(OUT_DIR, 'valid.csv'), index=False)
    df_test.to_csv(os.path.join(OUT_DIR, 'test.csv'), index=False)
    
    print("\nBERHASIL!")
    print(f"Train: {len(df_train)} | Valid: {len(df_valid)} | Test: {len(df_test)}")
    print(f"Tersimpan di folder: {OUT_DIR}")

if __name__ == "__main__":
    # Pastikan pip install scikit-learn pandas opencv-python sudah dilakukan
    main()