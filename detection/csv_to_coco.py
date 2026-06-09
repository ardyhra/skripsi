import os
import pandas as pd
import json
import shutil
from tqdm import tqdm

# --- KONFIGURASI ---
SOURCE_ROOT = 'data_prepared/lprdataset' 
OUTPUT_ROOT = 'data_prepared/lprdataset_rfdetr_coco'      

def create_coco_json_safe(csv_path, output_json_path, source_img_dir, dest_img_dir):
    if not os.path.exists(csv_path):
        return

    df = pd.read_csv(csv_path)
    
    # KITA UBAH ID JADI 0 (License Plate) AGAR AMAN
    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 0, "name": "license_plate", "supercategory": "none"}]
    }
    
    img_name_to_id = {}
    ann_id = 1
    os.makedirs(dest_img_dir, exist_ok=True)
    
    # Iterasi per gambar
    for filename, group in tqdm(df.groupby('filename'), desc=f"Processing {os.path.basename(output_json_path)}"):
        src_path = os.path.join(source_img_dir, filename)
        dst_path = os.path.join(dest_img_dir, filename)
        
        # 1. Cek File Gambar Ada/Tidak
        if not os.path.exists(src_path):
            continue
            
        # 2. Ambil Dimensi & Cek Validitas Dimensi
        w_img, h_img = group.iloc[0]['width'], group.iloc[0]['height']
        if w_img <= 0 or h_img <= 0:
            print(f"Skipping {filename}: Invalid image dimensions {w_img}x{h_img}")
            continue

        # Copy Image
        shutil.copy(src_path, dst_path)
        
        img_id = len(coco_data["images"])
        img_name_to_id[filename] = img_id
        
        coco_data["images"].append({
            "id": img_id,
            "file_name": filename,
            "width": int(w_img),
            "height": int(h_img)
        })
        
        # 3. Proses Annotations dengan FILTER KETAT
        for _, row in group.iterrows():
            xmin, ymin = row['xmin'], row['ymin']
            xmax, ymax = row['xmax'], row['ymax']
            
            # Clamp koordinat (jangan sampai minus atau melebihi gambar)
            xmin = max(0, xmin)
            ymin = max(0, ymin)
            xmax = min(w_img, xmax)
            ymax = min(h_img, ymax)
            
            bbox_w = xmax - xmin
            bbox_h = ymax - ymin
            
            # FILTER: Buang box yang luasnya 0 atau negatif
            if bbox_w <= 1 or bbox_h <= 1:
                # print(f"Skipping bad box in {filename}: {bbox_w}x{bbox_h}")
                continue
            
            coco_data["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 0, # FORCE ID 0
                "bbox": [float(xmin), float(ymin), float(bbox_w), float(bbox_h)],
                "area": float(bbox_w * bbox_h),
                "iscrowd": 0
            })
            ann_id += 1
            
    with open(output_json_path, 'w') as f:
        json.dump(coco_data, f)
    print(f"Saved safe JSON to {output_json_path}")

def main():
    if os.path.exists(OUTPUT_ROOT):
        shutil.rmtree(OUTPUT_ROOT) # Bersihkan folder lama biar fresh
        
    for split in ['train', 'valid', 'test']:
        csv_file = os.path.join(SOURCE_ROOT, split, '_annotations.csv')
        src_img_dir = os.path.join(SOURCE_ROOT, split)
        dest_split_dir = os.path.join(OUTPUT_ROOT, split)
        json_out = os.path.join(dest_split_dir, '_annotations.coco.json')
        
        create_coco_json_safe(csv_file, json_out, src_img_dir, dest_split_dir)

if __name__ == "__main__":
    main()