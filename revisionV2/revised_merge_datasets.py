import os
import cv2
import shutil
import pandas as pd
import xml.etree.ElementTree as ET
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm

MAX_LABEL_LEN = 10
DIR_YAZDANI = 'data_prepared/yazdani_lp'
DIR_DATASET_1 = 'data_prepared/european_lp'
DIR_DATASET_2 = 'data_prepared/indian_lp'
OUT_DIR = 'data_prepared/dataset_unified_recog'
OUT_IMG_DIR = os.path.join(OUT_DIR, 'images')
os.makedirs(OUT_IMG_DIR, exist_ok=True)


def clean_label(text):
    label = re.sub(r'[^A-Z0-9]', '', str(text).upper())
    return label[:MAX_LABEL_LEN]


def safe_add_record(records, src_path, out_name, label, copy_mode='copy', crop_img=None):
    if not label:
        return False
    if copy_mode == 'copy':
        if not os.path.exists(src_path):
            return False
        shutil.copy(src_path, os.path.join(OUT_IMG_DIR, out_name))
    else:
        if crop_img is None or crop_img.size == 0:
            return False
        cv2.imwrite(os.path.join(OUT_IMG_DIR, out_name), crop_img)
    records.append({'filename': out_name, 'label': label, 'label_len': len(label)})
    return True


def main():
    data_records = []
    image_counter = 0

    print('Memproses Dataset Yazdani...')
    if os.path.exists(DIR_YAZDANI):
        df_yazdani = pd.read_csv(os.path.join(DIR_YAZDANI, 'lpr.csv'))
        for _, row in tqdm(df_yazdani.iterrows(), total=len(df_yazdani)):
            img_name = row['images']
            label = clean_label(row['labels'])
            src_path = os.path.join(DIR_YAZDANI, 'cropped_lps', img_name)
            new_img_name = f'yz_{image_counter}.jpg'
            if safe_add_record(data_records, src_path, new_img_name, label):
                image_counter += 1

    print('Memproses Dataset 1 (Label di Nama File)...')
    if os.path.exists(DIR_DATASET_1):
        for root, _, files in os.walk(DIR_DATASET_1):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    label = clean_label(os.path.splitext(file)[0])
                    src_path = os.path.join(root, file)
                    new_img_name = f'ds1_{image_counter}.jpg'
                    if safe_add_record(data_records, src_path, new_img_name, label):
                        image_counter += 1

    print('Memproses Dataset 2 (Format XML)...')
    if os.path.exists(DIR_DATASET_2):
        for root, _, files in os.walk(DIR_DATASET_2):
            for file in tqdm(files):
                if not file.lower().endswith('.xml'):
                    continue
                xml_path = os.path.join(root, file)
                try:
                    tree = ET.parse(xml_path)
                    xml_root = tree.getroot()
                    stem = os.path.splitext(file)[0]
                    img_path = None
                    for ext in ('.jpg', '.png', '.jpeg'):
                        candidate = os.path.join(root, stem + ext)
                        if os.path.exists(candidate):
                            img_path = candidate
                            break
                    if img_path is None:
                        continue
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    h, w = img.shape[:2]

                    for obj in xml_root.findall('object'):
                        label = clean_label(obj.find('name').text)
                        bbox = obj.find('bndbox')
                        xmin = max(0, int(float(bbox.find('xmin').text)))
                        ymin = max(0, int(float(bbox.find('ymin').text)))
                        xmax = min(w, int(float(bbox.find('xmax').text)))
                        ymax = min(h, int(float(bbox.find('ymax').text)))
                        crop_img = img[ymin:ymax, xmin:xmax]
                        new_img_name = f'ds2_{image_counter}.jpg'
                        if safe_add_record(data_records, img_path, new_img_name, label, copy_mode='write', crop_img=crop_img):
                            image_counter += 1
                except Exception as e:
                    print(f'Error reading {xml_path}: {e}')

    df_all = pd.DataFrame(data_records)
    print(f'\nTotal Data Terkumpul: {len(df_all)} plat nomor')
    if len(df_all) == 0:
        raise RuntimeError('Tidak ada data yang berhasil dikumpulkan.')

    # Stratify minimal pada panjang label agar distribusi panjang tidak terlalu timpang.
    stratify_col = df_all['label_len'] if df_all['label_len'].value_counts().min() >= 2 else None
    df_train, df_temp = train_test_split(df_all, test_size=0.2, random_state=42, stratify=stratify_col)

    stratify_temp = df_temp['label_len'] if df_temp['label_len'].value_counts().min() >= 2 else None
    df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=stratify_temp)

    df_train[['filename', 'label']].to_csv(os.path.join(OUT_DIR, 'train.csv'), index=False)
    df_valid[['filename', 'label']].to_csv(os.path.join(OUT_DIR, 'valid.csv'), index=False)
    df_test[['filename', 'label']].to_csv(os.path.join(OUT_DIR, 'test.csv'), index=False)

    print('BERHASIL!')
    print(f"Train: {len(df_train)} | Valid: {len(df_valid)} | Test: {len(df_test)}")
    print('Distribusi panjang label:')
    print(df_all['label_len'].value_counts().sort_index())
    print(f'Tersimpan di folder: {OUT_DIR}')


if __name__ == '__main__':
    main()
