# Transformer + CTC recognizer experiment

## Files
- `ctc_config.py`
- `ctc_dataset_recog_only.py`
- `ctc_model_recog_only.py`
- `ctc_train_recog_only.py`
- `ctc_eval_recog_only.py`
- `ctc_finetune_recog_on_detector_crops.py`
- `ctc_inference_hybrid_v1.py`
- `ctc_eval_end_to_end_indian_lp.py`

## Recommended order

### 1) Train base recognizer CTC
```bash
python rfdetr/revisionV4/ctc_train_recog_only.py `
  --data-dir data_prepared/dataset_unified_recog `
  --out-dir rfdetr/checkpoints_recog_ctc
```

### 2) Evaluate on cropped recognition test split
```bash
python rfdetr/revisionV4/ctc_eval_recog_only.py `
  --data-dir data_prepared/dataset_unified_recog `
  --split test `
  --weights rfdetr/checkpoints_recog_ctc/best_recognizer_ctc.pth `
  --save-csv ctc_recog_test.csv
```

### 3) Fine-tune on detector crops
Reuse the existing dataset builder:
```bash
python build_detector_crop_finetune_dataset.py `
  --data-dir data_prepared/indian_lp `
  --det-weights rfdetr/runs/rfdetr_nano_result/checkpoint_best_total.pth `
  --out-dir data_prepared/indian_lp_detector_crops `
  --det-conf 0.5 `
  --iou-thresh 0.5
```

Then fine-tune:
```bash
python rfdetr/revisionV4/ctc_finetune_recog_on_detector_crops.py `
  --data-dir data_prepared/indian_lp_detector_crops `
  --init-weights rfdetr/checkpoints_recog_ctc/best_recognizer_ctc.pth `
  --out-dir rfdetr/checkpoints_recog_ctc_detector_crops `
  --epochs 10 `
  --lr 1e-4
```

### 4) End-to-end evaluation on Indian LP XML
```bash
python rfdetr/revisionV4/ctc_eval_end_to_end_indian_lp.py `
  --data-dir data_prepared/indian_lp `
  --det-weights rfdetr/runs/rfdetr_nano_result/checkpoint_best_total.pth `
  --rec-weights rfdetr/checkpoints_recog_ctc_detector_crops/best_recognizer_ctc_detector_crops.pth `
  --iou-thresh 0.5 `
  --det-conf 0.5 `
  --save-csv rfdetr/revisionV4/ctc_e2e_indian_lp.csv
```

### 5) Video / image inference
```bash
python rfdetr/revisionV4/ctc_inference_hybrid_v1.py `
  --input alprtest_1.mp4 `
  --output hasil_ctc.mp4 `
  --det-weights rfdetr/runs/rfdetr_nano_result/checkpoint_best_total.pth `
  --rec-weights rfdetr/checkpoints_recog_ctc_detector_crops/best_recognizer_ctc_detector_crops.pth
```

## Notes
- `max_chars` and `expected_chars` in CTC scripts are **optional post-processing constraints**, not part of the CTC objective itself.
- For the fairest comparison with the PAD+EOS recognizer, keep detector, crop expansion, and evaluation protocol the same.
