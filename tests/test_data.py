import csv
import json
import shutil

import numpy as np
from PIL import Image
import pytest

from industrial_inspection.data import build_manifest, generate_synthetic, read_manifest, resolve_root, training_paths, validate_dataset


@pytest.fixture
def synthetic(tmp_path):
    root = generate_synthetic(tmp_path / "data", seed=13)
    manifest = tmp_path / "manifests" / "synthetic.csv"
    return root, manifest


def test_synthetic_manifest_has_disjoint_content_groups_and_masks(synthetic):
    root, manifest = synthetic
    summary = build_manifest(root, "synthetic", manifest, seed=42)
    rows = read_manifest(manifest)
    assert summary["split_counts"] == {"train": 40, "calibration": 10, "test": 30}
    assert summary["split_label_counts"]["test"] == {"0": 10, "1": 20}
    assert summary["dataset_source"] == "synthetic"
    assert len({row["sha256"] for row in rows}) == 80
    hashes = {split: {row["sha256"] for row in rows if row["split"] == split} for split in ("train", "calibration", "test")}
    assert not hashes["train"] & hashes["calibration"]
    assert not (hashes["train"] | hashes["calibration"]) & hashes["test"]
    assert all(row["label"] == 0 for row in rows if row["split"] != "test")
    for row in rows:
        assert isinstance(row["label"], int)
        if row["label"]:
            assert Image.open(root / row["mask_path"]).size == Image.open(root / row["image_path"]).size
    train, calibration = training_paths(manifest)
    assert len(train) == 40 and len(calibration) == 10
    assert all(path.is_file() for path in train + calibration)


def test_split_is_reproducible_and_original_test_is_unchanged(synthetic):
    root, manifest = synthetic
    before = sorted(path.relative_to(root).as_posix() for path in (root / "synthetic" / "test").rglob("*.png"))
    build_manifest(root, "synthetic", manifest, seed=7)
    first = manifest.read_bytes()
    build_manifest(root, "synthetic", manifest, seed=7)
    assert manifest.read_bytes() == first
    build_manifest(root, "synthetic", manifest, seed=8)
    assert manifest.read_bytes() != first
    after = sorted(path.relative_to(root).as_posix() for path in (root / "synthetic" / "test").rglob("*.png"))
    assert before == after


def test_duplicate_training_content_stays_in_one_split(synthetic):
    root, manifest = synthetic
    original = root / "synthetic" / "train" / "good" / "normal_000.png"
    duplicate = original.with_name("duplicate_reencoded.png")
    # Different PNG compression still has the same decoded pixel content.
    Image.open(original).save(duplicate, compress_level=0)
    build_manifest(root, "synthetic", manifest)
    rows = read_manifest(manifest)
    matches = [row for row in rows if row["image_path"].endswith(("normal_000.png", "duplicate_reencoded.png")) and "/train/" in row["image_path"]]
    assert len(matches) == 2
    assert matches[0]["sha256"] == matches[1]["sha256"]
    assert matches[0]["split"] == matches[1]["split"]


def test_train_test_duplicate_is_rejected_before_manifest_is_written(synthetic):
    root, manifest = synthetic
    shutil.copyfile(root / "synthetic" / "train" / "good" / "normal_000.png", root / "synthetic" / "test" / "good" / "leak.png")
    with pytest.raises(ValueError, match="leakage"):
        build_manifest(root, "synthetic", manifest)
    assert not manifest.exists()


@pytest.mark.parametrize("failure", ["missing", "shape", "empty"])
def test_anomaly_masks_are_required_and_aligned(synthetic, failure):
    root, manifest = synthetic
    mask = root / "synthetic" / "ground_truth" / "scratch" / "scratch_000_mask.png"
    if failure == "missing":
        mask.unlink()
    elif failure == "shape":
        Image.new("L", (12, 12), 255).save(mask)
    else:
        Image.new("L", (128, 128), 0).save(mask)
    with pytest.raises(ValueError, match={"missing": "no ground-truth mask", "shape": "dimensions differ", "empty": "empty"}[failure]):
        build_manifest(root, "synthetic", manifest)


def test_manifest_follows_a_moved_project_using_relative_dataset_path(synthetic, tmp_path):
    root, manifest = synthetic
    build_manifest(root, "synthetic", manifest)
    summary_path = manifest.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert "\\" not in summary["dataset_root_relative_to_manifest"]
    summary["dataset_root_relative_to_manifest"] = summary["dataset_root_relative_to_manifest"].replace("/", "\\")
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    moved = tmp_path / "moved"
    moved.mkdir()
    shutil.move(str(root), str(moved / "data"))
    shutil.move(str(manifest.parent), str(moved / "manifests"))
    moved_manifest = moved / "manifests" / manifest.name
    assert resolve_root(moved_manifest) == (moved / "data").resolve()
    assert len(training_paths(moved_manifest)[0]) == 40


def test_changed_image_after_prepare_is_rejected(synthetic):
    root, manifest = synthetic
    build_manifest(root, "synthetic", manifest)
    assert validate_dataset(manifest) == {"verified_images": 80, "verified_masks": 20}
    row = next(row for row in read_manifest(manifest) if row["split"] == "train")
    path = root / row["image_path"]
    Image.new("RGB", (128, 128), (255, 0, 0)).save(path)
    with pytest.raises(ValueError, match="content changed after prepare"):
        training_paths(manifest)
    with pytest.raises(ValueError, match="content changed after prepare"):
        validate_dataset(manifest)


def test_manifest_integrity_and_path_escape_are_rejected(synthetic):
    root, manifest = synthetic
    build_manifest(root, "synthetic", manifest)
    with manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows[0]["image_path"] = "../outside.png"
    with manifest.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="relative"):
        read_manifest(manifest)
    with pytest.raises(ValueError, match="checksum"):
        resolve_root(manifest)


def test_generator_is_deterministic_and_does_not_replace_unmarked_data(tmp_path):
    root = generate_synthetic(tmp_path / "data", seed=17)
    sample = root / "synthetic" / "train" / "good" / "normal_000.png"
    original = sample.read_bytes()
    generate_synthetic(root, seed=17)
    assert sample.read_bytes() == original
    generate_synthetic(root, seed=18)
    assert sample.read_bytes() != original
    other = tmp_path / "other" / "synthetic" / "train" / "good"
    other.mkdir(parents=True)
    Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)).save(other / "user.png")
    with pytest.raises(ValueError, match="unmarked"):
        generate_synthetic(tmp_path / "other")


def test_missing_data_and_invalid_split_parameters_are_actionable(tmp_path):
    with pytest.raises(ValueError, match="No normal"):
        build_manifest(tmp_path, "bottle", tmp_path / "manifest.csv")
    with pytest.raises(ValueError, match="directory name"):
        build_manifest(tmp_path, "../bottle", tmp_path / "manifest.csv")
    with pytest.raises(ValueError, match="calibration_fraction"):
        build_manifest(tmp_path, "bottle", tmp_path / "manifest.csv", calibration_fraction=1)
