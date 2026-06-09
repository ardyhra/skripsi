import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import pandas as pd


def levenshtein_ops(gt: str, pred: str) -> List[Tuple[str, str, str]]:
    """Return alignment operations from gt -> pred.

    Each op is a tuple of:
      - op type: match / sub / del / ins
      - gt char ('' for insertion)
      - pred char ('' for deletion)
    """
    m, n = len(gt), len(pred)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if gt[i - 1] == pred[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )

    ops: List[Tuple[str, str, str]] = []
    i, j = m, n
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if gt[i - 1] == pred[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                if cost == 0:
                    ops.append(("match", gt[i - 1], pred[j - 1]))
                else:
                    ops.append(("sub", gt[i - 1], pred[j - 1]))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("del", gt[i - 1], ""))
            i -= 1
            continue
        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("ins", "", pred[j - 1]))
            j -= 1
            continue
        # Fallback for numerical ties.
        if i > 0 and j > 0:
            ops.append(("sub", gt[i - 1], pred[j - 1]))
            i -= 1
            j -= 1
        elif i > 0:
            ops.append(("del", gt[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", pred[j - 1]))
            j -= 1

    ops.reverse()
    return ops


def safe_float(value, default=0.0):
    try:
        if pd.isna(value) or value == "":
            return default
        return float(value)
    except Exception:
        return default


def load_eval_rows(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {
        "gt_text", "pred_text", "bbox_match", "text_exact", "det_conf", "text_conf", "iou"
    }
    missing = sorted(required_cols - set(df.columns))
    if missing:
        raise ValueError(f"CSV tidak punya kolom yang dibutuhkan: {missing}")

    df["gt_text"] = df["gt_text"].fillna("").astype(str)
    df["pred_text"] = df["pred_text"].fillna("").astype(str)
    df["bbox_match"] = df["bbox_match"].fillna(0).astype(int)
    df["text_exact"] = df["text_exact"].fillna(0).astype(int)
    df["det_conf"] = df["det_conf"].apply(safe_float)
    df["text_conf"] = df["text_conf"].apply(safe_float)
    df["iou"] = df["iou"].apply(safe_float)
    if "cer" in df.columns:
        df["cer"] = df["cer"].apply(lambda x: safe_float(x, default=None) if x != "" else None)
    return df


def build_summary(df: pd.DataFrame) -> Dict:
    matched = df[(df["bbox_match"] == 1) & (df["gt_text"] != "")].copy()
    exact = matched[matched["text_exact"] == 1].copy()
    errors = matched[matched["text_exact"] == 0].copy()
    unmatched_gt = df[(df["bbox_match"] == 0) & (df["gt_text"] != "")].copy()
    unmatched_pred = df[(df["bbox_match"] == 0) & (df["pred_text"] != "")].copy()

    gt_len_counter = Counter()
    exact_len_counter = Counter()
    pred_len_counter = Counter()

    op_counter = Counter()
    substitution_counter = Counter()
    deletion_counter = Counter()
    insertion_counter = Counter()
    position_correct = Counter()
    position_total = Counter()
    prefix_correct = Counter()
    suffix_correct = Counter()
    error_examples = []
    text_conf_exact = []
    text_conf_error = []

    total_gt_chars = 0
    total_matches = 0
    total_subs = 0
    total_ins = 0
    total_dels = 0
    total_correct_chars = 0

    for _, row in matched.iterrows():
        gt = row["gt_text"]
        pred = row["pred_text"]
        gt_len_counter[len(gt)] += 1
        pred_len_counter[len(pred)] += 1
        if row["text_exact"] == 1:
            exact_len_counter[len(gt)] += 1
            text_conf_exact.append(row["text_conf"])
        else:
            text_conf_error.append(row["text_conf"])

        ops = levenshtein_ops(gt, pred)
        total_matches += 1
        total_gt_chars += len(gt)

        aligned_gt_pos = 0
        for op, gt_c, pred_c in ops:
            op_counter[op] += 1
            if op == "match":
                total_correct_chars += 1
                position_correct[aligned_gt_pos] += 1
                position_total[aligned_gt_pos] += 1
                aligned_gt_pos += 1
            elif op == "sub":
                total_subs += 1
                substitution_counter[(gt_c, pred_c)] += 1
                position_total[aligned_gt_pos] += 1
                aligned_gt_pos += 1
            elif op == "del":
                total_dels += 1
                deletion_counter[gt_c] += 1
                position_total[aligned_gt_pos] += 1
                aligned_gt_pos += 1
            elif op == "ins":
                total_ins += 1
                insertion_counter[pred_c] += 1

        # Simple prefix / suffix sanity checks.
        if gt[:1] == pred[:1] and len(gt) >= 1 and len(pred) >= 1:
            prefix_correct[1] += 1
        if gt[:2] == pred[:2] and len(gt) >= 2 and len(pred) >= 2:
            prefix_correct[2] += 1
        if gt[-1:] == pred[-1:] and len(gt) >= 1 and len(pred) >= 1:
            suffix_correct[1] += 1
        if gt[-2:] == pred[-2:] and len(gt) >= 2 and len(pred) >= 2:
            suffix_correct[2] += 1

        dist = sum(1 for op, _, _ in ops if op != "match")
        error_examples.append({
            "gt_text": gt,
            "pred_text": pred,
            "gt_len": len(gt),
            "pred_len": len(pred),
            "edit_distance": dist,
            "normalized_edit": dist / max(len(gt), 1),
            "iou": row["iou"],
            "det_conf": row["det_conf"],
            "text_conf": row["text_conf"],
            "xml_path": row.get("xml_path", ""),
            "image_path": row.get("image_path", ""),
        })

    char_acc_aligned = total_correct_chars / total_gt_chars if total_gt_chars else 0.0
    exact_rate = len(exact) / len(matched) if len(matched) else 0.0

    summary = {
        "rows_total": int(len(df)),
        "matched_boxes": int(len(matched)),
        "matched_exact": int(len(exact)),
        "matched_errors": int(len(errors)),
        "matched_exact_rate": exact_rate,
        "gt_missed_detection": int(len(unmatched_gt)),
        "pred_false_positives": int(len(unmatched_pred)),
        "aligned_char_accuracy": char_acc_aligned,
        "total_gt_chars_aligned": int(total_gt_chars),
        "total_correct_chars_aligned": int(total_correct_chars),
        "edit_ops": {
            "substitutions": int(total_subs),
            "insertions": int(total_ins),
            "deletions": int(total_dels),
        },
        "mean_text_conf_exact": float(sum(text_conf_exact) / len(text_conf_exact)) if text_conf_exact else 0.0,
        "mean_text_conf_error": float(sum(text_conf_error) / len(text_conf_error)) if text_conf_error else 0.0,
    }

    length_stats = []
    for gt_len in sorted(gt_len_counter):
        total = gt_len_counter[gt_len]
        exact_n = exact_len_counter.get(gt_len, 0)
        length_stats.append({
            "gt_length": gt_len,
            "count": total,
            "exact_count": exact_n,
            "exact_rate": exact_n / total if total else 0.0,
        })

    position_stats = []
    for pos in sorted(position_total):
        total = position_total[pos]
        corr = position_correct.get(pos, 0)
        position_stats.append({
            "char_position_1based": pos + 1,
            "total": total,
            "correct": corr,
            "accuracy": corr / total if total else 0.0,
        })

    confusion_rows = []
    for (gt_c, pred_c), count in substitution_counter.most_common():
        confusion_rows.append({
            "gt_char": gt_c,
            "pred_char": pred_c,
            "count": count,
        })

    insertion_rows = [{"pred_char": ch, "count": count} for ch, count in insertion_counter.most_common()]
    deletion_rows = [{"gt_char": ch, "count": count} for ch, count in deletion_counter.most_common()]

    examples_df = pd.DataFrame(error_examples)
    if not examples_df.empty:
        examples_df = examples_df.sort_values(
            by=["normalized_edit", "edit_distance", "text_conf", "iou"],
            ascending=[False, False, False, True],
        )

    return {
        "summary": summary,
        "length_stats": pd.DataFrame(length_stats),
        "position_stats": pd.DataFrame(position_stats),
        "confusions": pd.DataFrame(confusion_rows),
        "insertions": pd.DataFrame(insertion_rows),
        "deletions": pd.DataFrame(deletion_rows),
        "hard_errors": examples_df,
    }


def write_outputs(results: Dict, out_dir: str, top_k: int):
    os.makedirs(out_dir, exist_ok=True)

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results["summary"], f, ensure_ascii=False, indent=2)

    results["length_stats"].to_csv(os.path.join(out_dir, "length_stats.csv"), index=False)
    results["position_stats"].to_csv(os.path.join(out_dir, "position_accuracy.csv"), index=False)
    results["confusions"].head(top_k).to_csv(os.path.join(out_dir, "top_substitutions.csv"), index=False)
    results["insertions"].head(top_k).to_csv(os.path.join(out_dir, "top_insertions.csv"), index=False)
    results["deletions"].head(top_k).to_csv(os.path.join(out_dir, "top_deletions.csv"), index=False)
    results["hard_errors"].head(top_k).to_csv(os.path.join(out_dir, "hard_error_examples.csv"), index=False)

    summary_txt_path = os.path.join(out_dir, "summary.txt")
    s = results["summary"]
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("===== OCR ERROR ANALYSIS SUMMARY =====\n")
        f.write(f"Matched boxes                 : {s['matched_boxes']}\n")
        f.write(f"Exact matches                 : {s['matched_exact']}\n")
        f.write(f"Matched exact rate            : {s['matched_exact_rate'] * 100:.2f}%\n")
        f.write(f"Aligned char accuracy         : {s['aligned_char_accuracy'] * 100:.2f}%\n")
        f.write(f"Missed detections (GT unmatched): {s['gt_missed_detection']}\n")
        f.write(f"False-positive detections     : {s['pred_false_positives']}\n")
        f.write(f"Substitutions / Insertions / Deletions : {s['edit_ops']['substitutions']} / {s['edit_ops']['insertions']} / {s['edit_ops']['deletions']}\n")
        f.write(f"Mean text_conf exact          : {s['mean_text_conf_exact']:.4f}\n")
        f.write(f"Mean text_conf error          : {s['mean_text_conf_error']:.4f}\n")

        f.write("\nTop substitutions:\n")
        if results["confusions"].empty:
            f.write("  (none)\n")
        else:
            for _, row in results["confusions"].head(top_k).iterrows():
                f.write(f"  {row['gt_char']} -> {row['pred_char']}: {int(row['count'])}\n")

        f.write("\nTop insertions:\n")
        if results["insertions"].empty:
            f.write("  (none)\n")
        else:
            for _, row in results["insertions"].head(top_k).iterrows():
                f.write(f"  +{row['pred_char']}: {int(row['count'])}\n")

        f.write("\nTop deletions:\n")
        if results["deletions"].empty:
            f.write("  (none)\n")
        else:
            for _, row in results["deletions"].head(top_k).iterrows():
                f.write(f"  -{row['gt_char']}: {int(row['count'])}\n")


def main():
    parser = argparse.ArgumentParser(description="Analisis error OCR dari CSV evaluasi end-to-end")
    parser.add_argument("--csv", type=str, required=True, help="CSV hasil dari eval_end_to_end_indian_lp.py --save-csv")
    parser.add_argument("--out-dir", type=str, default="ocr_error_analysis", help="Folder output analisis")
    parser.add_argument("--top-k", type=int, default=25, help="Jumlah error teratas yang disimpan")
    args = parser.parse_args()

    df = load_eval_rows(args.csv)
    results = build_summary(df)
    write_outputs(results, args.out_dir, args.top_k)

    s = results["summary"]
    print("===== OCR ERROR ANALYSIS =====")
    print(f"Matched boxes                 : {s['matched_boxes']}")
    print(f"Exact matches                 : {s['matched_exact']}")
    print(f"Matched exact rate            : {s['matched_exact_rate'] * 100:.2f}%")
    print(f"Aligned char accuracy         : {s['aligned_char_accuracy'] * 100:.2f}%")
    print(f"Missed detections (GT unmatched): {s['gt_missed_detection']}")
    print(f"False-positive detections     : {s['pred_false_positives']}")
    print(
        "Substitutions / Insertions / Deletions : "
        f"{s['edit_ops']['substitutions']} / {s['edit_ops']['insertions']} / {s['edit_ops']['deletions']}"
    )
    print(f"Mean text_conf exact          : {s['mean_text_conf_exact']:.4f}")
    print(f"Mean text_conf error          : {s['mean_text_conf_error']:.4f}")
    print(f"\nOutput saved to: {args.out_dir}")


if __name__ == "__main__":
    main()
