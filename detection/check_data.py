import json
import os
import numpy as np

def validate_coco(json_path):
    print(f"Checking {json_path}...")
    with open(json_path, 'r') as f:
        data = json.load(f)

    valid_imgs = 0
    valid_anns = 0
    invalid_anns = 0

    # 1. Cek Image Size
    images_map = {}
    for img in data['images']:
        if img['width'] <= 0 or img['height'] <= 0:
            print(f"[ERROR] Image ID {img['id']} ({img['file_name']}) has invalid size: {img['width']}x{img['height']}")
        images_map[img['id']] = img

    # 2. Cek Bounding Box
    for ann in data['annotations']:
        x, y, w, h = ann['bbox']
        img = images_map[ann['image_id']]
        
        # Cek 1: Ukuran box negatif/nol
        if w <= 0 or h <= 0:
            print(f"[ERROR] Ann ID {ann['id']} on Image {img['file_name']}: Invalid Box Size w={w}, h={h}")
            invalid_anns += 1
            continue
            
        # Cek 2: Box keluar dari gambar
        if x < 0 or y < 0 or (x+w) > img['width'] or (y+h) > img['height']:
            print(f"[WARNING] Ann ID {ann['id']} on Image {img['file_name']}: Box out of bounds. Box: {ann['bbox']}, Img: {img['width']}x{img['height']}")
            # Warning biasanya masih bisa jalan (di-clip), tapi sebaiknya dibetulkan
            
        # Cek 3: NaN
        if np.isnan(x) or np.isnan(y) or np.isnan(w) or np.isnan(h):
            print(f"[FATAL] Ann ID {ann['id']} contains NaN!")
            invalid_anns += 1

    print(f"Result: {len(data['annotations'])} total annotations. {invalid_anns} INVALID found.")

if __name__ == "__main__":
    # Sesuaikan path folder dataset Anda
    base_dir = "data_prepared/lprdataset_rfdetr_coco" 
    for split in ['train', 'valid', 'test']:
        p = os.path.join(base_dir, split, '_annotations.coco.json')
        if os.path.exists(p):
            validate_coco(p)