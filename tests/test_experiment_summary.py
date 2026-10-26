"""A summary may claim completion only for current, fully covered evaluations.

Tiny text artifacts stand in for immutable model bytes. These tests exercise
the published report contract without loading a model, images, or optional UI
and deep-learning packages.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_three_products.py"
SPEC = importlib.util.spec_from_file_location("three_product_experiment", SCRIPT)
runner = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(runner)

CATEGORIES = ("bottle", "metal_nut", "tile")
METHODS = ("pixel", "patch")
METHOD_IDENTITIES = {"pixel": "pixel_reference", "patch": "patch_memory"}
PREDICTION_FIELDS = ["image_path", "defect_type", "label", "score", "predicted_anomaly",
                     "image_threshold", "pixel_threshold", "dice_128", "iou_128", "latency_ms"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def csv_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


class Experiment:
    def __init__(self, root: Path):
        self.root = root
        self.output = root / "results" / "three_products"
        self.evaluation_code = root / "src/industrial_inspection/evaluation.py"
        self.evaluation_code.parent.mkdir(parents=True)
        self.evaluation_code.write_text("# frozen evaluator fixture\n", encoding="utf-8")
        self.n_test = {category: index + 2 for index, category in enumerate(CATEGORIES)}
        self.test_records: dict[str, list[dict]] = {}
        for index, category in enumerate(CATEGORIES):
            self._dataset(category)
            policy = root / runner.PRODUCTS[category]["policy_file"]
            write_json(policy, {"product_category": category, "version": "fixture-v1"})
            sample_dir = root / "samples" / category
            sample_dir.mkdir(parents=True)
            for sample in range(index + 1):
                (sample_dir / f"good_{sample:03}.png").write_bytes(b"sample fixture; never decoded")
            for method in METHODS:
                model_dir = self.model_dir(category, method)
                model_dir.mkdir(parents=True)
                artifact = model_dir / "model.npz"
                artifact.write_bytes(f"immutable artifact:{category}:{method}".encode())
                write_json(model_dir / "model_card.json", {
                    "method": METHOD_IDENTITIES[method], "dataset_source": f"mvtec_ad/{category}",
                    "manifest_sha256": digest(self.manifest(category)), "artifact_sha256": digest(artifact),
                    "training_count": 1, "calibration_count": 1,
                })

    def manifest(self, category: str) -> Path:
        return self.root / "data/manifests" / f"{category}.csv"

    def model_dir(self, category: str, method: str) -> Path:
        return self.root / "models" / f"{category}_{method}"

    def result_dir(self, category: str, method: str) -> Path:
        return self.output / category / method

    def _dataset(self, category: str) -> None:
        records = [{"image_path": f"{category}/train/good/{split}.png", "split": split,
                    "label": 0, "defect_type": "good"} for split in ("train", "calibration")]
        test = [{"image_path": f"{category}/test/{'good' if index == 0 else 'defect'}/{index:03}.png",
                 "split": "test", "label": int(index != 0), "defect_type": "good" if index == 0 else "defect"}
                for index in range(self.n_test[category])]
        self.test_records[category] = test
        write_csv(self.manifest(category), ["image_path", "split", "label", "defect_type"], records + test)
        write_json(self.manifest(category).with_suffix(".summary.json"), {
            "category": category, "image_count": len(records) + len(test),
            "split_counts": {"train": 1, "calibration": 1, "test": len(test)},
            "split_label_counts": {"test": {"0": 1, "1": len(test) - 1}},
            "manifest_sha256": digest(self.manifest(category)),
        })

    def add_result(self, category: str, method: str, *, false_negative: bool = False) -> None:
        folder = self.result_dir(category, method)
        predictions = []
        for index, record in enumerate(self.test_records[category]):
            predicted = 0 if false_negative and index == 1 else record["label"]
            predictions.append({"image_path": record["image_path"], "defect_type": record["defect_type"],
                                "label": record["label"], "score": 0.9 if predicted else 0.1,
                                "predicted_anomaly": predicted, "image_threshold": 0.5,
                                "pixel_threshold": 0.5, "dice_128": 0.6 if record["label"] else "",
                                "iou_128": 0.4 if record["label"] else "", "latency_ms": 1.2})
        write_csv(folder / "predictions.csv", PREDICTION_FIELDS, predictions)
        fn = int(false_negative)
        tp = self.n_test[category] - 1 - fn
        card_path = self.model_dir(category, method) / "model_card.json"
        write_json(folder / "metrics.json", {
            "model": f"{category}_{method}", "method": METHOD_IDENTITIES[method],
            "dataset_source": f"mvtec_ad/{category}", "manifest_sha256": digest(self.manifest(category)),
            "model_card_sha256": digest(card_path), "evaluation_code_sha256": digest(self.evaluation_code),
            "n_test": self.n_test[category], "n_normal": 1, "n_anomaly": self.n_test[category] - 1,
            "image_auroc": 0.7 + 0.1 * CATEGORIES.index(category) + (0.03 if method == "patch" else 0),
            "precision": 1.0 if tp else 0.0, "recall": tp / (tp + fn),
            "f1": 2 * tp / (2 * tp + fn), "confusion": {"tn": 1, "fp": 0, "fn": fn, "tp": tp},
            "pixel_auroc_128": 0.8, "mean_anomaly_iou_128": 0.4, "inference_median_ms": 1.2,
        })

    def add_all_results(self) -> None:
        for category in CATEGORIES:
            for method in METHODS:
                self.add_result(category, method)

    def summarize(self) -> dict:
        return runner.write_summary(self.root, self.output)


@pytest.fixture
def experiment(tmp_path):
    return Experiment(tmp_path)


def test_one_category_is_partial_and_does_not_publish_three_category_macro(experiment):
    for method in METHODS:
        experiment.add_result("bottle", method)
    summary = experiment.summarize()
    assert summary["complete"] is False
    assert summary["completed_model_count"] == 2
    assert summary["macro_category_count"] == {"pixel": 1, "patch": 1}
    assert summary["macro_image_auroc"] == {"pixel": None, "patch": None}
    assert read_json(experiment.output / "summary.json")["complete"] is False


def test_all_six_current_results_complete_with_distinct_support_counts(experiment):
    experiment.add_all_results()
    summary = experiment.summarize()
    assert summary["complete"] is True
    assert summary["completed_model_count"] == 6
    assert summary["macro_category_count"] == {"pixel": 3, "patch": 3}
    assert summary["macro_image_auroc"] == pytest.approx({"pixel": 0.8, "patch": 0.83})
    assert summary["distinct_test_images"] == 9
    assert summary["total_real_images"] == 15
    assert summary["display_samples"] == {"bottle": 1, "metal_nut": 2, "tile": 3}
    assert len({row["model"] for row in summary["models"]}) == 6
    assert sum(row["n_test"] for row in summary["models"]) == 18  # Both methods see the same nine images.
    assert len(csv_rows(experiment.output / "comparison.csv")) == 6


@pytest.mark.parametrize("changed, expected_models", [("card", 5), ("artifact", 5), ("manifest", 4), ("evaluator", 0)])
def test_changed_inputs_exclude_old_metrics_and_replace_current_comparison(experiment, changed, expected_models):
    experiment.add_all_results()
    assert experiment.summarize()["complete"] is True
    if changed == "card":
        path = experiment.model_dir("bottle", "patch") / "model_card.json"
        card = read_json(path)
        card["revision_note"] = "changed after evaluation"
        write_json(path, card)
    elif changed == "artifact":
        path = experiment.model_dir("bottle", "patch") / "model.npz"
        path.write_bytes(path.read_bytes() + b"changed after evaluation")
    elif changed == "manifest":
        path = experiment.manifest("bottle")
        path.write_text(path.read_text(encoding="utf-8").replace("train.png", "a_different_train.png"), encoding="utf-8")
        data_path = path.with_suffix(".summary.json")
        data = read_json(data_path)
        data["manifest_sha256"] = digest(path)
        write_json(data_path, data)
    else:
        experiment.evaluation_code.write_text("# evaluator changed after evaluation\n", encoding="utf-8")
    summary = experiment.summarize()
    assert summary["complete"] is False
    assert summary["completed_model_count"] == expected_models
    assert len(summary["stale_records"]) == 6 - expected_models
    for method in METHODS:
        if summary["macro_category_count"][method] < 3:
            assert summary["macro_image_auroc"][method] is None
    # The comparison is a current report too; it cannot retain formerly valid rows.
    assert len(csv_rows(experiment.output / "comparison.csv")) == expected_models


@pytest.mark.parametrize("changed", ["method", "source", "model_name"])
def test_self_consistent_but_wrong_model_identity_is_excluded(experiment, changed):
    experiment.add_all_results()
    card_path = experiment.model_dir("bottle", "patch") / "model_card.json"
    metrics_path = experiment.result_dir("bottle", "patch") / "metrics.json"
    card, metrics = read_json(card_path), read_json(metrics_path)
    if changed == "method":
        card["method"] = metrics["method"] = "pixel_reference"
    elif changed == "source":
        card["dataset_source"] = metrics["dataset_source"] = "mvtec_ad/tile"
    else:
        metrics["model"] = "bottle_pixel"
    write_json(card_path, card)
    metrics["model_card_sha256"] = digest(card_path)
    write_json(metrics_path, metrics)
    summary = experiment.summarize()
    assert summary["complete"] is False
    assert summary["completed_model_count"] == 5
    assert not any(row["category"] == "bottle" and row["method"] == "patch" for row in summary["models"])
    assert summary["macro_image_auroc"]["patch"] is None


@pytest.mark.parametrize("corruption", ["duplicate_missing", "foreign_path", "missing_row"])
def test_predictions_must_cover_each_held_out_path_exactly_once(experiment, corruption):
    experiment.add_all_results()
    path = experiment.result_dir("bottle", "patch") / "predictions.csv"
    predictions = csv_rows(path)
    if corruption == "duplicate_missing":
        predictions[1] = dict(predictions[0])
    elif corruption == "foreign_path":
        predictions[1]["image_path"] = "bottle/test/defect/not_in_the_manifest.png"
    else:
        predictions.pop()
    write_csv(path, PREDICTION_FIELDS, predictions)
    with pytest.raises(ValueError, match="exact held-out test set"):
        experiment.summarize()
    assert not (experiment.output / "summary.json").exists()


def test_regenerating_error_export_removes_old_false_negative(experiment):
    experiment.add_all_results()
    experiment.add_result("bottle", "patch", false_negative=True)
    experiment.summarize()
    errors = csv_rows(experiment.output / "errors.csv")
    assert len(errors) == 1
    assert errors[0]["error"] == "FN" and errors[0]["model"] == "bottle_patch"
    experiment.add_result("bottle", "patch", false_negative=False)
    assert experiment.summarize()["complete"] is True
    assert csv_rows(experiment.output / "errors.csv") == []
    assert "image_path" in (experiment.output / "errors.csv").read_text(encoding="utf-8-sig")


def test_summary_script_import_does_not_require_torch_or_gradio():
    code = """
import importlib.abc
import runpy
import sys
class NoOptionalPackages(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'torchvision', 'gradio'}:
            raise ImportError('Optional package deliberately unavailable: ' + fullname)
sys.meta_path.insert(0, NoOptionalPackages())
runpy.run_path(sys.argv[1], run_name='summary_import_only')
assert not any(name.split('.')[0] in {'torch', 'torchvision', 'gradio'} for name in sys.modules)
print('SUMMARY_IMPORT_OK')
"""
    result = subprocess.run([sys.executable, "-B", "-c", code, str(SCRIPT)], check=True, capture_output=True, text=True)
    assert "SUMMARY_IMPORT_OK" in result.stdout
