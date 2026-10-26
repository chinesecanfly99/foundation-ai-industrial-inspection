"""Fit/reuse category models and evaluate every held-out image at frozen thresholds."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from industrial_inspection.catalog import PRODUCTS, REAL_PRODUCTS
from industrial_inspection.cli import fit_model
from industrial_inspection.evaluation import evaluate


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_summary(root: Path, output: Path) -> dict:
    rows, datasets, errors, stale_records = [], [], [], []
    for category in REAL_PRODUCTS:
        data_path = root / "data" / "manifests" / f"{category}.summary.json"
        manifest = data_path.with_name(f"{category}.csv")
        if not data_path.is_file() or not manifest.is_file():
            continue
        data = json.loads(data_path.read_text(encoding="utf-8"))
        manifest_hash = sha256(manifest)
        if data["manifest_sha256"] != manifest_hash:
            raise ValueError(f"Manifest and split summary disagree: {category}")
        with manifest.open(encoding="utf-8", newline="") as handle:
            test_paths = {item["image_path"] for item in csv.DictReader(handle) if item["split"] == "test"}
        datasets.append({"category": category, "label": PRODUCTS[category]["label"],
                         "images": data["image_count"], "splits": data["split_counts"],
                         "test_labels": data["split_label_counts"]["test"],
                         "manifest_sha256": manifest_hash})
        for method in ("pixel", "patch"):
            folder = output / category / method
            metrics_path = folder / "metrics.json"
            if not metrics_path.is_file():
                continue
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            model_dir = root / "models" / f"{category}_{method}"
            card_path, artifact = model_dir / "model_card.json", model_dir / "model.npz"
            expected_method = {"pixel": "pixel_reference", "patch": "patch_memory"}[method]
            card = json.loads(card_path.read_text(encoding="utf-8")) if card_path.is_file() else {}
            compatible = (artifact.is_file() and metrics.get("model") == model_dir.name
                          and metrics.get("method") == card.get("method") == expected_method
                          and metrics.get("dataset_source") == card.get("dataset_source") == f"mvtec_ad/{category}"
                          and metrics.get("manifest_sha256") == card.get("manifest_sha256") == manifest_hash
                          and metrics.get("model_card_sha256") == sha256(card_path)
                          and metrics.get("evaluation_code_sha256") == sha256(root / "src/industrial_inspection/evaluation.py")
                          and card.get("artifact_sha256") == sha256(artifact)
                          and metrics.get("n_test") == data["split_counts"]["test"]
                          and metrics.get("n_normal") == data["split_label_counts"]["test"]["0"]
                          and metrics.get("n_anomaly") == data["split_label_counts"]["test"]["1"])
            if not compatible:
                stale_records.append({"category": category, "method": method, "reason": "Evaluation does not match the current model, manifest or evaluation code."})
                continue
            row = {"category": category, "product": PRODUCTS[category]["label"], "method": method,
                   "model": metrics["model"], "n_test": metrics["n_test"],
                   "image_auroc": metrics["image_auroc"], "precision": metrics["precision"],
                   "recall": metrics["recall"], "f1": metrics["f1"],
                   **metrics["confusion"], "pixel_auroc_128": metrics["pixel_auroc_128"],
                   "mean_anomaly_iou_128": metrics["mean_anomaly_iou_128"],
                   "inference_median_ms": metrics["inference_median_ms"]}
            with (folder / "predictions.csv").open(encoding="utf-8", newline="") as handle:
                predictions = list(csv.DictReader(handle))
            if len(predictions) != metrics["n_test"] or {item["image_path"] for item in predictions} != test_paths:
                raise ValueError(f"Predictions do not cover the exact held-out test set: {folder}")
            rows.append(row)
            for prediction in predictions:
                label, predicted = int(prediction["label"]), int(prediction["predicted_anomaly"])
                if label != predicted:
                    errors.append({"category": category, "model": metrics["model"],
                                   "error": "FN" if label else "FP", **prediction})
    output.mkdir(parents=True, exist_ok=True)
    complete = len(rows) == len(REAL_PRODUCTS) * 2 and len(datasets) == len(REAL_PRODUCTS)
    macro_counts = {method: sum(item["method"] == method for item in rows) for method in ("pixel", "patch")}
    summary = {
        "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "complete": complete, "datasets": datasets, "models": rows, "stale_records": stale_records,
        "completed_model_count": len(rows), "macro_category_count": macro_counts,
        "total_real_images": sum(item["images"] for item in datasets),
        "distinct_test_images": sum(item["splits"]["test"] for item in datasets),
        "display_samples": {category: len(list((root / "samples" / category).glob("*.png"))) for category in REAL_PRODUCTS},
        "macro_image_auroc": {method: (sum(item["image_auroc"] for item in rows if item["method"] == method) / len(REAL_PRODUCTS)
                                     if macro_counts[method] == len(REAL_PRODUCTS) else None) for method in ("pixel", "patch")},
        "policy_sha256": {category: sha256(root / PRODUCTS[category]["policy_file"]) for category in REAL_PRODUCTS},
        "protocol": "Each category has separate normal-only fitting/calibration, two models, and its complete original test set. No test threshold tuning. Pixel evaluation is pooled at 128x128, not official full-resolution MVTec scoring. Quality rules have no independently validated business labels.",
        "scope": "Three-category course prototype; synthetic examples excluded from every reported detection metric.",
    }
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    comparison_fields = ["category", "product", "method", "model", "n_test", "image_auroc", "precision", "recall", "f1", "tn", "fp", "fn", "tp", "pixel_auroc_128", "mean_anomaly_iou_128", "inference_median_ms"]
    with (output / "comparison.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=comparison_fields)
        writer.writeheader()
        writer.writerows(rows)
    error_fields = ["category", "model", "error", "image_path", "defect_type", "label", "score", "predicted_anomaly", "image_threshold", "pixel_threshold", "dice_128", "iou_128", "latency_ms"]
    with (output / "errors.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=error_fields)
        writer.writeheader()
        writer.writerows(errors)
    markdown = ["# 三类模型独立测试结果", "", f"生成时间：{summary['created_at']}", "",
                "每类单独建模、正常校准后固定阈值；以下是视觉异常检测指标，质量处置正确率尚未独立验证。", "",
                "| 产品 | 方法 | 测试图 | 图像 AUROC | 召回率 | 正常误报 FP | 缺陷漏检 FN |", "| --- | --- | ---: | ---: | ---: | ---: | ---: |"]
    markdown.extend(f"| {row['product']} | {row['method']} | {row['n_test']} | {row['image_auroc']:.4f} | {row['recall']:.4f} | {row['fp']} | {row['fn']} |" for row in rows)
    markdown += ["", "完整逐图分数见各产品/方法的 predictions.csv；误报和漏检汇总见 errors.csv。", "",
                 "像素定位指标按128×128评估，不等于官方全分辨率指标。合成图不计入此表。", ""]
    (output / "summary.md").write_text("\n".join(markdown), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--categories", nargs="+", choices=REAL_PRODUCTS, default=list(REAL_PRODUCTS))
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--retrain", action="store_true", help="Explicitly replace existing category models")
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "results" / "three_products"
    if not args.summary_only:
        import torch
        torch.set_num_threads(4)
        for category in args.categories:
            manifest = root / "data" / "manifests" / f"{category}.csv"
            data = json.loads(manifest.with_suffix(".summary.json").read_text(encoding="utf-8"))
            for method in ("pixel", "patch"):
                model_dir = root / "models" / f"{category}_{method}"
                card_path, artifact = model_dir / "model_card.json", model_dir / "model.npz"
                reuse = card_path.is_file() and artifact.is_file() and not args.retrain
                if reuse:
                    card = json.loads(card_path.read_text(encoding="utf-8"))
                    expected_method = {"pixel": "pixel_reference", "patch": "patch_memory"}[method]
                    compatible = (card.get("manifest_sha256") == sha256(manifest)
                                  and card.get("artifact_sha256") == sha256(artifact)
                                  and card.get("method") == expected_method
                                  and card.get("dataset_source") == f"mvtec_ad/{category}"
                                  and card.get("training_count") == data["split_counts"]["train"]
                                  and card.get("calibration_count") == data["split_counts"]["calibration"]
                                  and card.get("image_size") == 128 and card.get("seed") == 42
                                  and (method != "patch" or card.get("memory_size") == 2048))
                    if not compatible:
                        raise ValueError(f"Existing model is not compatible with this manifest/artifact: {model_dir}; inspect it before --retrain.")
                    print(f"Reusing frozen {model_dir.name}", flush=True)
                else:
                    print(f"Fitting {category}/{method} with {data['split_counts']['train']} normal images", flush=True)
                    fit_model(manifest, method, model_dir, device="cpu")
                metrics = evaluate(manifest, model_dir, output / category / method, device="cpu")
                print(json.dumps({"category": category, "method": method, "n_test": metrics["n_test"],
                                  "image_auroc": metrics["image_auroc"], "recall": metrics["recall"],
                                  "confusion": metrics["confusion"]}), flush=True)
                write_summary(root, output)
    summary = write_summary(root, output)
    print(json.dumps({"complete": summary["complete"], "total_real_images": summary["total_real_images"],
                      "distinct_test_images": summary["distinct_test_images"], "models": len(summary["models"])}), flush=True)


if __name__ == "__main__":
    main()
