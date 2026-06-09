import os
import re
import shutil
import cv2
import pandas as pd
import xml.etree.ElementTree as ET
from collections import Counter
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from revised_pad_eos_config import MAX_PLATE_LEN, sanitize_plate_text

DIR_YAZDANI = "data_prepared/yazdani_lp"
DIR_DATASET_1 = "data_prepared/european_lp"
DIR_DATASET_2 = "data_prepared/indian_lp"
OUT_DIR = "data_prepared/dataset_unified_recog"
OUT_IMG_DIR = os.path.join(OUT_DIR, "images")

os.makedirs(OUT_IMG_DIR, exist_ok=True)


def clean_label(text):
    text = sanitize_plate_text(text)
    return text[:MAX_PLATE_LEN]


def safe_copy(src_path, dst_path):
    try:
        shutil.copy(src_path, dst_path)
        return True
    except Exception:
        return False


def stratify_key(df):
    lengths = df["label"].str.len()
    counts = lengths.value_counts().to_dict()
    return lengths.map(lambda x: x if counts.get(x, 0) >= 3 else -1)


def main():
    data_records = []
    image_counter = 0

    print("Memproses Dataset Yazdani...")
    if os.path.exists(DIR_YAZDANI):
        df_yazdani = pd.read_csv(os.path.join(DIR_YAZDANI, "lpr.csv"))
        for _, row in tqdm(df_yazdani.iterrows(), total=len(df_yazdani)):
            img_name = row["images"]
            label = clean_label(row["labels"])
            src_path = os.path.join(DIR_YAZDANI, "cropped_lps", img_name)
            if not os.path.exists(src_path) or not label:
                continue
            new_img_name = f"yz_{image_counter}.jpg"
            dst_path = os.path.join(OUT_IMG_DIR, new_img_name)
            if safe_copy(src_path, dst_path):
                data_records.append({"filename": new_img_name, "label": label})
                image_counter += 1

    print("Memproses Dataset 1 (label di nama file)...")
    if os.path.exists(DIR_DATASET_1):
        for root, _, files in os.walk(DIR_DATASET_1):
            for file in files:
                if not file.lower().endswith((".png", ".jpg", ".jpeg")):
                    continue
                label = clean_label(os.path.splitext(file)[0])
                if not label:
                    continue
                src_path = os.path.join(root, file)
                new_img_name = f"ds1_{image_counter}.jpg"
                dst_path = os.path.join(OUT_IMG_DIR, new_img_name)
                if safe_copy(src_path, dst_path):
                    data_records.append({"filename": new_img_name, "label": label})
                    image_counter += 1

    print("Memproses Dataset 2 (format XML)...")
    if os.path.exists(DIR_DATASET_2):
        for root, _, files in os.walk(DIR_DATASET_2):
            for file in tqdm(files):
                if not file.lower().endswith(".xml"):
                    continue
                xml_path = os.path.join(root, file)
                try:
                    tree = ET.parse(xml_path)
                    xml_root = tree.getroot()
                    stem = os.path.splitext(file)[0]
                    img_path = None
                    for ext in (".jpg", ".jpeg", ".png"):
                        candidate = os.path.join(root, stem + ext)
                        if os.path.exists(candidate):
                            img_path = candidate
                            break
                    if img_path is None:
                        continue

                    img = cv2.imread(img_path)
                    if img is None:
                        continue

                    for obj in xml_root.findall("object"):
                        label_node = obj.find("name")
                        bbox = obj.find("bndbox")
                        if label_node is None or bbox is None:
                            continue
                        label = clean_label(label_node.text)
                        if not label:
                            continue

                        xmin = int(float(bbox.find("xmin").text))
                        ymin = int(float(bbox.find("ymin").text))
                        xmax = int(float(bbox.find("xmax").text))
                        ymax = int(float(bbox.find("ymax").text))
                        xmin, ymin = max(0, xmin), max(0, ymin)
                        xmax, ymax = min(img.shape[1], xmax), min(img.shape[0], ymax)
                        crop_img = img[ymin:ymax, xmin:xmax]
                        if crop_img.size == 0:
                            continue

                        new_img_name = f"ds2_{image_counter}.jpg"
                        dst_path = os.path.join(OUT_IMG_DIR, new_img_name)
                        cv2.imwrite(dst_path, crop_img)
                        data_records.append({"filename": new_img_name, "label": label})
                        image_counter += 1
                except Exception as exc:
                    print(f"Error reading {xml_path}: {exc}")

    print(f"\nTotal data terkumpul: {len(data_records)}")
    df_all = pd.DataFrame(data_records).drop_duplicates(subset=["filename"]).reset_index(drop=True)
    print("Distribusi panjang label:", Counter(df_all["label"].str.len()))

    stratify_series = stratify_key(df_all)
    stratify_target = stratify_series if (stratify_series != -1).sum() > 0 else None

    try:
        df_train, df_temp = train_test_split(df_all, test_size=0.2, random_state=42, stratify=stratify_target)
    except Exception:
        df_train, df_temp = train_test_split(df_all, test_size=0.2, random_state=42)

    temp_stratify = stratify_key(df_temp)
    temp_target = temp_stratify if (temp_stratify != -1).sum() > 0 else None
    try:
        df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42, stratify=temp_target)
    except Exception:
        df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

    df_train.to_csv(os.path.join(OUT_DIR, "train.csv"), index=False)
    df_valid.to_csv(os.path.join(OUT_DIR, "valid.csv"), index=False)
    df_test.to_csv(os.path.join(OUT_DIR, "test.csv"), index=False)

    print("\nBERHASIL!")
    print(f"Train: {len(df_train)} | Valid: {len(df_valid)} | Test: {len(df_test)}")
    print(f"Tersimpan di folder: {OUT_DIR}")


if __name__ == "__main__":
    main()
