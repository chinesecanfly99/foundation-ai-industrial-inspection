"""Frozen-threshold held-out evaluation; never selects a test-set threshold."""
import csv
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

from .data import read_manifest, resolve_root, validate_dataset
from .detectors import load_detector


def evaluate(manifest: Path, model_dir: Path, output: Path, device="cpu") -> dict:
    output.mkdir(parents=True, exist_ok=True)
    validation = validate_dataset(manifest)
    rows = [r for r in read_manifest(manifest) if r["split"] == "test"]
    if not rows:
        raise ValueError("没有独立test集")
    root = resolve_root(manifest)
    model = load_detector(model_dir, device=device)
    card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if card.get("manifest_sha256") != manifest_hash:
        raise ValueError("模型与待评估manifest不匹配；请用同一划分重新训练")
    records, pixel_true, pixel_scores, dice, iou = [], [], [], [], []
    # Warm-up time is recorded separately; latency numbers below are warm CPU/GPU inference.
    first = np.asarray(Image.open(root / rows[0]["image_path"]).convert("RGB"))
    start = time.perf_counter()
    model.predict(first)
    warmup_ms = (time.perf_counter() - start) * 1000
    for row in rows:
        rgb = np.asarray(Image.open(root / row["image_path"]).convert("RGB"))
        start = time.perf_counter()
        pred = model.predict(rgb)
        latency_ms = (time.perf_counter() - start) * 1000
        target = Image.open(root / row["mask_path"]).convert("L") if row["mask_path"] else Image.new("L", (rgb.shape[1], rgb.shape[0]))
        target_arr = np.asarray(target.resize((128, 128), Image.Resampling.NEAREST)) > 0
        score_arr = np.asarray(Image.fromarray(pred.heatmap.astype(np.float32)).resize((128, 128), Image.Resampling.BILINEAR))
        binary = score_arr > pred.pixel_threshold
        pixel_true.append(target_arr.reshape(-1))
        pixel_scores.append(score_arr.reshape(-1))
        intersection = np.logical_and(target_arr, binary).sum()
        union = np.logical_or(target_arr, binary).sum()
        row_dice = float(2 * intersection / max(target_arr.sum() + binary.sum(), 1))
        row_iou = float(intersection / max(union, 1))
        if row["label"]:
            dice.append(row_dice)
            iou.append(row_iou)
        records.append({"image_path": row["image_path"], "defect_type": row["defect_type"], "label": row["label"],
                        "score": float(pred.score), "predicted_anomaly": int(pred.score > pred.image_threshold),
                        "image_threshold": float(pred.image_threshold), "pixel_threshold": float(pred.pixel_threshold),
                        "dice_128": row_dice, "iou_128": row_iou, "latency_ms": latency_ms})
    y = [r["label"] for r in records]
    score = [r["score"] for r in records]
    binary = [r["predicted_anomaly"] for r in records]
    pixel_y = np.concatenate(pixel_true)
    pixel_s = np.concatenate(pixel_scores)
    latencies = [r["latency_ms"] for r in records]
    tn, fp, fn, tp = confusion_matrix(y, binary, labels=[0, 1]).ravel().tolist()
    result = {
        "model": model_dir.name, "method": card.get("method"), "dataset_source": card.get("dataset_source"),
        "data_validation": validation,
        "model_card_sha256": hashlib.sha256((model_dir / "model_card.json").read_bytes()).hexdigest(),
        "evaluation_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "manifest_sha256": manifest_hash, "n_test": len(rows), "n_normal": int(len(y) - sum(y)), "n_anomaly": int(sum(y)),
        "image_auroc": float(roc_auc_score(y, score)) if len(set(y)) == 2 else None,
        "image_average_precision": float(average_precision_score(y, score)) if sum(y) else None,
        "precision": float(precision_score(y, binary, zero_division=0)), "recall": float(recall_score(y, binary, zero_division=0)),
        "f1": float(f1_score(y, binary, zero_division=0)), "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "pixel_auroc_128": float(roc_auc_score(pixel_y, pixel_s)) if len(np.unique(pixel_y)) == 2 else None,
        "mean_anomaly_dice_128": float(np.mean(dice)) if dice else None,
        "mean_anomaly_iou_128": float(np.mean(iou)) if iou else None,
        "inference_median_ms": float(np.median(latencies)), "inference_p95_ms": float(np.percentile(latencies, 95)),
        "warmup_ms": warmup_ms, "device": device,
        "protocol": "Frozen normal-calibration thresholds; all held-out test images; pooled pixel AUROC at 128x128, NOT official full-resolution MVTec evaluation or AUPRO. No quality-decision ground truth.",
        "threshold_selection": "Normal calibration images only. Test data never changes thresholds.",
        "synthetic_only": str(card.get("dataset_source", "")).startswith("synthetic")
    }
    (output / "metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (output / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "model_card_snapshot.json").write_text(json.dumps(card, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
