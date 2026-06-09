# Automatic License Plate Recognition (ALPR) Berbasis RF-DETR & Transformer

Proyek penelitian skripsi ini mengimplementasikan sistem deteksi dan rekognisi pelat nomor kendaraan otomatis (ALPR) menggunakan kombinasi model deteksi objek modern berbasis Transformer (**RF-DETR Nano**) dan model pengenal karakter (**ResNet18 + Transformer Encoder**).

Sistem dirancang menggunakan pendekatan **Modular Two-Stage Pipeline**:
1. **Stage 1 (Detection)**: Mendeteksi dan memotong area pelat nomor kendaraan dari citra/video masukan secara real-time menggunakan **RF-DETR Nano**.
2. **Stage 2 (Recognition)**: Mengekstrak karakter pelat nomor dari potongan citra (crop) pelat tersebut menggunakan arsitektur berbasis **Transformer** dengan beberapa variasi decoding (Cross-Entropy, PAD+EOS, dan CTC Loss).

Berdasarkan hasil eksperimen *end-to-end* pada dataset Indian LP (non-cropped), modul **V3 (PAD+EOS dengan Fixed Slots)** yang di-*fine-tune* pada hasil crop detektor memberikan performa terbaik (**Akurasi Karakter: 98.84%, E2E F1-Score: 79.88%**). Modul **V4 (CTC)** digunakan sebagai arsitektur pembanding akademik (**Akurasi Karakter: 93.45%, E2E F1-Score: 61.62%**).

---

## Struktur Proyek

Berikut adalah struktur folder utama dari repositori proyek skripsi ini:

```text
├── data_prepared/                          # Direktori Dataset & File Pendukung
│   ├── lprdataset_rfdetr_coco/             # Dataset deteksi pelat nomor dalam format COCO
│   ├── dataset_unified_recog/              # Dataset rekognisi karakter terpadu (train.csv, valid.csv, test.csv)
│   ├── indian_lp/                          # Dataset pengujian eksternal (plat nomor India)
│   ├── indian_lp_detector_crops/           # Hasil crop pelat nomor India dari detektor untuk fine-tuning
│   └── (zip & dataset pendukung lainnya)
│
└── rfdetr/                                 # Direktori Kode Utama Proyek
    ├── detection/                          # Stage 1: Deteksi Pelat Nomor (RF-DETR)
    │   ├── csv_to_coco.py                  # Konversi anotasi CSV ke format COCO JSON
    │   ├── check_data.py                   # Validasi visual dataset deteksi
    │   ├── train_rfdetr_nano.py            # Script pelatihan RF-DETR Nano
    │   └── inference_rfdetr_nano.py        # Pengujian deteksi mandiri
    │
    ├── recognitionV1/                      # Stage 2: Rekognisi Pelat V1 (Fixed-Length 7 Chars)
    │   ├── config.py                       # Kamus karakter (vocabulary) & perangkat (device)
    │   ├── dataset_recog_only.py           # Dataset loader gambar crop & label
    │   ├── model_recog_only.py             # Arsitektur ResNet18 + Transformer Encoder + 1D Avg Pool (7 Chars)
    │   ├── train_recog_only.py             # Pelatihan model rekognisi V1
    │   └── merge_datasets.py               # Penggabungan berbagai sumber dataset rekognisi
    │
    ├── revisionV2/                         # Stage 2: Rekognisi Pelat V2 (Fixed-Length 10 Chars + PAD)
    │   ├── revised_config.py               # Penambahan PAD token & panjang maksimum 10 karakter
    │   ├── revised_dataset_recog_only.py   # Dataset loader dengan token padding
    │   ├── revised_model_recog_only.py     # Model adaptif dengan pooling ke 10 karakter
    │   ├── revised_train_recog_only.py     # Pelatihan model V2
    │   └── revised_inference_hybrid_v2.py  # Pipeline inferensi gabungan Detektor + Recognizer V2
    │
    ├── revisionV3/                         # Stage 2: Rekognisi Pelat V3 (Variable-Length PAD + EOS) - TERBAIK
    │   ├── revised_pad_eos_config.py       # Penambahan token EOS (End of Sequence)
    │   ├── revised_pad_eos_dataset_recog.py# Dataset loader dengan padding dan penanda EOS
    │   ├── revised_pad_eos_model_recog.py  # Model dengan pooling spasial ke panjang maksimum 11 (10+EOS)
    │   ├── revised_pad_eos_train_recog.py  # Pelatihan model V3 dengan Cross-Entropy Loss
    │   ├── build_detector_crop_finetune.py # Pembuatan dataset khusus dari prediksi detektor untuk fine-tuning
    │   ├── finetune_recog_on_detector_crops.py # Fine-tuning model V3 pada hasil crop detektor
    │   ├── eval_end_to_end_indian_lp.py    # Evaluasi akurasi E2E pada dataset Indian LP
    │   └── analyze_ocr_errors.py           # Analisis kesalahan karakter (CER, WER, visualisasi kesalahan)
    │
    ├── revisionV4/                         # Stage 2: Rekognisi Pelat V4 (CTC Loss - Pembanding)
    │   ├── ctc_config.py                   # Pengaturan decoding CTC & metrik Levenshtein
    │   ├── ctc_dataset_recog_only.py       # Dataset loader & Collate function untuk CTC (tanpa pad label eksplisit)
    │   ├── ctc_model_recog_only.py         # ResNet18 (Height-collapse Mean Pool) + Transformer Encoder + CTC Head
    │   ├── ctc_train_recog_only.py         # Pelatihan OCR berbasis CTC Loss
    │   ├── ctc_eval_recog_only.py          # Evaluasi CER & akurasi plat murni pada test split
    │   ├── ctc_finetune_recog_on_detector_crops.py # Fine-tuning model CTC pada deteksi realistik
    │   └── ctc_experiment_guide.md         # Panduan eksperimen bertahap untuk modul CTC
    │
    └── deploy/                             # Aplikasi Antarmuka Streamlit (Deployment)
        ├── app.py                          # Aplikasi Streamlit (Webcam, Upload File, Live CCTV)
        ├── alpr_core.py                    # Integrasi deteksi & rekognisi V3 dengan Stabilisasi Teks (Tracker)
        ├── config.py                       # Konfigurasi deploy, ambang batas deteksi, & path bobot model
        ├── utils.py                        # Helper konversi citra dan pemrosesan video
        └── models/                         # Folder penyimpanan bobot model untuk deployment
```

---

## Dataset & Sitasi

Berikut adalah sumber dataset yang digunakan dalam penelitian ini:

### 1. Dataset Deteksi Objek (Stage 1)
*   **License Plate Recognition Dataset**: 
    Roboflow Universe Projects. (2025). *License Plate Recognition Dataset*. Kaggle.
    [Tautan Dataset Kaggle](https://www.kaggle.com/datasets/roboflow/license-plate-recognition-dataset)
    Lisensi: CC BY 4.0 Attribution 4.0 International.

### 2. Dataset Rekognisi Karakter (Stage 2)
*   **Indian vehicle license plate dataset**:
    Sai Sirisha, Pranav Gaur, and Abhilash Bhardwaj. (2022). *Indian vehicle license plate dataset*. Kaggle.
    [Tautan Dataset Kaggle](https://www.kaggle.com/datasets/saisirishapg/indian-vehicle-license-plate-dataset) (https://doi.org/10.34740/KAGGLE/DSV/3369458)
    Lisensi: CC BY-SA 4.0.
*   **License Plate Text Recognition Dataset**:
    Nick Yazdani. ( Nick Yazdani ). (2022). *License Plate Text Recognition Dataset*. Kaggle.
    [Tautan Dataset Kaggle](https://www.kaggle.com/datasets/nickyazdani/license-plate-text-recognition-dataset) (https://doi.org/10.34740/KAGGLE/DSV/4781724)
    Lisensi: CC BY 0 Public Domain.
*   **European License Plates Dataset**:
    Abdulhamid Zakaria. (2024). *European License Plates Dataset*. Kaggle.
    [Tautan Dataset Kaggle](https://www.kaggle.com/datasets/abdulhamidzakaria/european-license-plates-dataset)
    Lisensi: MIT.

---

## Evolusi Arsitektur & Perbandingan Kinerja

Berikut adalah detail evolusi modul OCR pada dataset eksternal **Indian LP (non-cropped)** dengan total 1.695 citra uji:

| Metrik Evaluasi E2E | V3 (Sebelum FT) | V3 (Setelah FT - TERBAIK) | V4 CTC (Setelah FT - Pembanding) |
| :--- | :---: | :---: | :---: |
| **Arsitektur OCR** | ResNet18 + Transf. Enc | ResNet18 + Transf. Enc | Height-Collapse Mean Pool + CTC |
| **Metode Sekuens** | PAD + EOS (Slot Kaku 11) | PAD + EOS (Slot Kaku 11) | Sekuens Horizontal Dinamis |
| **Precision Deteksi** | 84.88% | 84.88% | 84.88% |
| **Recall Deteksi** | 85.13% | 85.13% | 85.13% |
| **OCR Exact Match** | 50.94% | **93.97%** | 72.49% |
| **OCR Char Accuracy**| 87.88% | **98.84%** | 93.45% |
| **Mean CER** | 11.57% | **1.17%** | 6.61% |
| **End-to-End TP** | 735 | **1356** | 1046 |
| **End-to-End F1-Score**| 43.30% | **79.88%** | 61.62% |

> [!NOTE]
> Eksperimen menunjukkan bahwa **V3 (PAD+EOS dengan slot kaku)** yang di-*fine-tune* menggunakan dataset deteksi terkelola jauh lebih unggul dibandingkan dengan arsitektur **V4 (CTC)**. CTC menghasilkan akurasi yang lebih rendah (F1-score 61.62%) dikarenakan sifat *greedy decoding* yang kurang stabil pada gambar dengan tingkat keburaman tinggi atau pencahayaan ekstrem.

---

## Panduan Menjalankan Eksperimen & Perintah Lengkap

### 1. Persiapan Lingkungan (Environment)
```bash
# Untuk pelatihan detektor & pustaka umum
pip install -r requirements.txt

# Untuk modul OCR (Transformer, CTC, & Levenshtein)
pip install -r requirements_ocr.txt
```

### 2. Tahap 1: Deteksi Pelat Nomor (RF-DETR)
```bash
# Konversi data ke format COCO JSON
python rfdetr/detection/csv_to_coco.py

# Pelatihan RF-DETR Nano
python rfdetr/detection/train_rfdetr_nano.py

# Inferensi Deteksi Mandiri (Single Image)
python rfdetr/detection/inference_rfdetr_nano.py
```

### 3. Tahap 2: Pelatihan, Evaluasi, & Inferensi OCR (Semua Versi)

#### A. Versi V1: Rekognisi Kaku 7 Karakter
```bash
# Pelatihan Model V1
python rfdetr/recognitionV1/train_recog_only.py

# Penggabungan dataset awal
python rfdetr/recognitionV1/merge_datasets.py
```

#### B. Versi V2: Rekognisi 10 Karakter + PAD
```bash
# Pelatihan Model V2
python rfdetr/revisionV2/revised_train_recog_only.py

# Uji coba inferensi hybrid (Detektor + OCR V2)
python rfdetr/revisionV2/revised_inference_hybrid_v2.py --input path/to/image.jpg
```

#### C. Versi V3: Rekognisi PAD + EOS (TERBAIK)
```bash
# Pelatihan Model V3 Dasar
python rfdetr/revisionV3/revised_pad_eos_train_recog_only.py

# Pembuatan Dataset Fine-Tuning dari Crop Bounding Box Detektor
python build_detector_crop_finetune_dataset.py \
  --data-dir data_prepared/indian_lp \
  --det-weights rfdetr/runs/rfdetr_nano_result/checkpoint_best_total.pth \
  --out-dir data_prepared/indian_lp_detector_crops

# Fine-Tuning Model V3 pada Hasil Crop Detektor
python rfdetr/revisionV3/finetune_recog_on_detector_crops.py \
  --data-dir data_prepared/indian_lp_detector_crops \
  --init-weights rfdetr/checkpoints_pure_recog_merge_pad_eos/best_recognizer_pad_eos.pth \
  --out-dir rfdetr/checkpoints_recog_detector_crops

# Evaluasi End-to-End pada Dataset Indian LP
python rfdetr/revisionV3/eval_end_to_end_indian_lp.py \
  --data-dir data_prepared/indian_lp \
  --det-weights rfdetr/runs/rfdetr_nano_result/checkpoint_best_total.pth \
  --rec-weights rfdetr/checkpoints_recog_detector_crops/best_recognizer_detector_crops.pth

# Evaluasi E2E Alternatif
python rfdetr/revisionV3/e2e_evaluation.py

# Analisis Kesalahan OCR & Distribusi Karakter Terdeteksi
python rfdetr/revisionV3/analyze_ocr_errors.py

# Inferensi Hybrid V3 pada Citra/Video
python rfdetr/revisionV3/revised_pad_eos_inference_hybrid_v2.py \
  --input alprtest_1.mp4 \
  --output hasil_currentmodel_tuned.mp4
```

#### D. Versi V4: Rekognisi CTC Loss (Pembanding)
```bash
# Pelatihan Model V4 Dasar
python rfdetr/revisionV4/ctc_train_recog_only.py \
  --data-dir data_prepared/dataset_unified_recog \
  --out-dir rfdetr/checkpoints_recog_ctc

# Evaluasi OCR CTC saja pada data uji
python rfdetr/revisionV4/ctc_eval_recog_only.py \
  --data-dir data_prepared/dataset_unified_recog \
  --split test \
  --weights rfdetr/checkpoints_recog_ctc/best_recognizer_ctc.pth

# Fine-Tuning Model CTC pada Hasil Crop Detektor
python rfdetr/revisionV4/ctc_finetune_recog_on_detector_crops.py \
  --data-dir data_prepared/indian_lp_detector_crops \
  --init-weights rfdetr/checkpoints_recog_ctc/best_recognizer_ctc.pth \
  --out-dir rfdetr/checkpoints_recog_ctc_detector_crops

# Evaluasi End-to-End CTC pada Dataset Indian LP
python rfdetr/revisionV4/ctc_eval_end_to_end_indian_lp.py \
  --data-dir data_prepared/indian_lp \
  --det-weights rfdetr/runs/rfdetr_nano_result/checkpoint_best_total.pth \
  --rec-weights rfdetr/checkpoints_recog_ctc_detector_crops/best_recognizer_ctc_detector_crops.pth

# Inferensi Hybrid CTC pada Citra/Video
python rfdetr/revisionV4/ctc_inference_hybrid_v1.py \
  --input alprtest_1.mp4 \
  --output hasil_ctc.mp4
```

### 4. Menjalankan Aplikasi Deployment (Streamlit)
Salin bobot model detektor (`checkpoint_best_total.pth`) dan rekognisi V3 terbaik Anda (`best_recognizer_detector_crops.pth`) ke folder `rfdetr/deploy/models/`, lalu jalankan antarmuka Streamlit:

```bash
streamlit run rfdetr/deploy/app.py
```

---



