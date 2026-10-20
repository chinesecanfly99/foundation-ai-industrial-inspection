"""Reproducible normal-only splits and explicitly synthetic smoke-test data."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageOps


_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
_FIELDS = ["image_path", "mask_path", "split", "label", "defect_type", "sha256", "group_id"]


def _image_files(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted((path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in _IMAGE_SUFFIXES), key=lambda path: path.as_posix())


def _content_identity(path: Path) -> tuple[str, tuple[int, int]]:
    """Hash decoded pixels, so PNG metadata cannot conceal exact duplicates."""
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
        image.load()
        digest = hashlib.sha256()
        digest.update(f"RGB:{image.width}:{image.height}:".encode("ascii"))
        digest.update(image.tobytes())
        return digest.hexdigest(), image.size


def _read_source(root: Path, category_root: Path) -> dict[str, Any]:
    for path in (category_root / "source.json", root / "source.json"):
        if path.is_file():
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError(f"Dataset source metadata must be a JSON object: {path}")
            return value
    return {"kind": "local_user_supplied", "provenance": "not independently verified"}


def build_manifest(
    dataset_root: Path,
    category: str,
    output: Path,
    seed: int = 42,
    calibration_fraction: float = 0.2,
) -> dict[str, Any]:
    """Build an MVTec-layout CSV without touching the original test split.

    Training and normal calibration are split by exact RGB content groups. This
    detects duplicate images, but cannot infer physical product identities.
    """
    if not isinstance(category, str) or not category or "/" in category or "\\" in category or category in {".", ".."}:
        raise ValueError("category must be one directory name.")
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must be between 0 and 1.")
    root = Path(dataset_root).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    category_root = root / category
    train_files = _image_files(category_root / "train" / "good")
    test_files = _image_files(category_root / "test")
    if not train_files:
        raise ValueError(f"No normal training images found in {category_root / 'train' / 'good'}")
    if not test_files:
        raise ValueError(f"No test images found in {category_root / 'test'}")
    other_train = set(_image_files(category_root / "train")) - set(train_files)
    if other_train:
        raise ValueError("Only train/good is supported: unexpected non-normal training images were found.")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    records: list[dict[str, Any]] = []
    for path in train_files:
        content_hash, _ = _content_identity(path)
        record = {
            "image_path": path.relative_to(root).as_posix(),
            "mask_path": "",
            "split": "train",
            "label": 0,
            "defect_type": "good",
            "sha256": content_hash,
            "group_id": content_hash,
        }
        records.append(record)
        groups[content_hash].append(record)
    if len(groups) < 2:
        raise ValueError("At least two distinct normal image-content groups are required for independent train/calibration splits.")
    group_hashes = sorted(groups)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(group_hashes))
    calibration_count = max(1, min(len(groups) - 1, int(round(len(groups) * calibration_fraction))))
    calibration_groups = {group_hashes[index] for index in order[:calibration_count]}
    for content_hash, grouped_records in groups.items():
        if content_hash in calibration_groups:
            for record in grouped_records:
                record["split"] = "calibration"

    for path in test_files:
        test_relative = path.relative_to(category_root / "test")
        if len(test_relative.parts) < 2:
            raise ValueError(f"Test images must be inside test/<defect_type>/: {path}")
        defect_type = test_relative.parts[0]
        label = int(defect_type != "good")
        content_hash, size = _content_identity(path)
        if content_hash in groups:
            raise ValueError(f"Image-content leakage between train/calibration and test: {path}")
        mask_path = ""
        if label:
            mask_relative = Path(*test_relative.parts[1:])
            expected_mask = category_root / "ground_truth" / defect_type / mask_relative.parent / f"{mask_relative.stem}_mask.png"
            if not expected_mask.is_file():
                raise ValueError(f"Anomalous test image has no ground-truth mask: {expected_mask}")
            with Image.open(expected_mask) as mask:
                mask.load()
                if mask.size != size:
                    raise ValueError(f"Image and ground-truth mask dimensions differ: {path}")
                if not np.asarray(mask.convert("L")).any():
                    raise ValueError(f"Anomalous test mask is empty: {expected_mask}")
            mask_path = expected_mask.relative_to(root).as_posix()
        records.append({
            "image_path": path.relative_to(root).as_posix(),
            "mask_path": mask_path,
            "split": "test",
            "label": label,
            "defect_type": defect_type,
            "sha256": content_hash,
            "group_id": content_hash,
        })

    identity_splits: dict[str, set[str]] = defaultdict(set)
    identity_labels: dict[str, set[int]] = defaultdict(set)
    for record in records:
        identity_splits[record["sha256"]].add(record["split"])
        identity_labels[record["sha256"]].add(record["label"])
    if any(len(splits) > 1 for splits in identity_splits.values()):
        raise ValueError("Image-content leakage detected across manifest splits.")
    if any(len(labels) > 1 for labels in identity_labels.values()):
        raise ValueError("The same image content has contradictory normal/anomaly labels.")

    records.sort(key=lambda record: (record["split"], record["image_path"]))
    source = _read_source(root, category_root)
    source_text = json.dumps(source, ensure_ascii=False).lower()
    dataset_source = "synthetic" if category == "synthetic" or source.get("kind") == "synthetic" else (f"mvtec_ad/{category}" if "mvtec" in source_text else f"local/{category}")
    counts = Counter(record["split"] for record in records)
    label_counts: dict[str, dict[str, int]] = {}
    for split in ("train", "calibration", "test"):
        labels = Counter(str(record["label"]) for record in records if record["split"] == split)
        label_counts[split] = {"0": labels["0"], "1": labels["1"]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    try:
        relative_root: str | None = os.path.relpath(root, output.parent).replace("\\", "/")
    except ValueError:  # On Windows, a relative path cannot span drive letters.
        relative_root = None
    summary: dict[str, Any] = {
        "schema_version": 1,
        "dataset_root": str(root),
        "dataset_root_relative_to_manifest": relative_root,
        "category": category,
        "dataset_source": dataset_source,
        "source": source,
        "seed": int(seed),
        "calibration_fraction": float(calibration_fraction),
        "split_counts": {split: counts[split] for split in ("train", "calibration", "test")},
        "split_label_counts": label_counts,
        "image_count": len(records),
        "unique_content_groups": len(identity_splits),
        "duplicate_files_within_splits": len(records) - len(identity_splits),
        "hash_definition": "SHA256 of UTF-8 RGB:width:height: prefix followed by decoded, EXIF-oriented RGB pixel bytes",
        "manifest_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "split_policy": "Seeded exact-content-group split of original train/good; original test remains untouched.",
        "limitations": [
            "No physical object/product IDs are supplied; different photographs of the same physical product may still cross splits.",
            "Content grouping detects exact decoded-pixel duplicates, not near duplicates or adjacent captures.",
            "Anomaly test labels and masks must never be used to choose model thresholds or quality rules.",
        ],
    }
    if dataset_source == "synthetic":
        summary["limitations"].append("Synthetic images are only for pipeline smoke tests; metrics do not represent industrial performance.")
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def _safe_relative(value: str, field: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or PureWindowsPath(value).drive or ".." in path.parts:
        raise ValueError(f"Manifest {field} must stay relative to dataset_root: {value}")
    return normalized


def read_manifest(manifest: Path) -> list[dict[str, Any]]:
    with Path(manifest).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"Manifest is missing required columns: {_FIELDS}")
        records: list[dict[str, Any]] = []
        for row in reader:
            record: dict[str, Any] = dict(row)
            try:
                record["label"] = int(record["label"])
            except (TypeError, ValueError) as exc:
                raise ValueError("Manifest labels must be integers 0 or 1.") from exc
            if record["label"] not in {0, 1} or record["split"] not in {"train", "calibration", "test"}:
                raise ValueError("Manifest contains an unsupported label or split.")
            if record["split"] in {"train", "calibration"} and record["label"] != 0:
                raise ValueError("Only normal images are allowed in train/calibration.")
            if not record["image_path"]:
                raise ValueError("Manifest image_path cannot be empty.")
            record["image_path"] = _safe_relative(record["image_path"], "image_path")
            record["mask_path"] = _safe_relative(record["mask_path"], "mask_path") if record["mask_path"] else ""
            if record["label"] and not record["mask_path"]:
                raise ValueError("Anomalous records require mask_path.")
            records.append(record)
    if not records:
        raise ValueError("Manifest contains no image records.")
    return records


def resolve_root(manifest: Path) -> Path:
    manifest = Path(manifest).expanduser().resolve()
    summary = json.loads(manifest.with_suffix(".summary.json").read_text(encoding="utf-8"))
    expected_hash = summary.get("manifest_sha256")
    if expected_hash and hashlib.sha256(manifest.read_bytes()).hexdigest() != expected_hash:
        raise ValueError("Manifest checksum differs from its summary; regenerate the manifest after changing its contents.")
    candidates = []
    relative = summary.get("dataset_root_relative_to_manifest")
    if relative:
        # Accept both portable new summaries and older Windows summaries.
        candidates.append((manifest.parent / str(relative).replace("\\", "/")).resolve())
    if summary.get("dataset_root"):
        candidates.append(Path(summary["dataset_root"]).expanduser().resolve())
    for candidate in candidates:
        if candidate.is_dir() and (candidate / str(summary.get("category", ""))).is_dir():
            return candidate
    raise FileNotFoundError("Cannot locate the dataset after moving the project. Run prepare again with the current dataset_root.")


def _validate_records(root: Path, records: list[dict[str, Any]]) -> dict[str, int]:
    identities: dict[str, tuple[str, int]] = {}
    mask_count = 0
    for record in records:
        path = root / record["image_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image cannot be found: {path}. Run prepare using the current dataset location.")
        content_hash, size = _content_identity(path)
        if content_hash != record["sha256"]:
            raise ValueError(f"Image content changed after prepare: {path}. Restore the recorded data or regenerate the manifest.")
        identity = (record["split"], record["label"])
        if content_hash in identities and identities[content_hash] != identity:
            raise ValueError(f"Image-content leakage or contradictory labels found during dataset validation: {path}")
        identities[content_hash] = identity
        if record["label"]:
            mask_path = root / record["mask_path"]
            if not mask_path.is_file():
                raise FileNotFoundError(f"Ground-truth mask cannot be found: {mask_path}")
            with Image.open(mask_path) as mask:
                mask.load()
                if mask.size != size:
                    raise ValueError(f"Image and ground-truth mask dimensions differ: {path}")
                if not np.asarray(mask.convert("L")).any():
                    raise ValueError(f"Anomalous test mask is empty: {mask_path}")
            mask_count += 1
    return {"verified_images": len(records), "verified_masks": mask_count}


def validate_dataset(manifest: Path) -> dict[str, int]:
    """Recheck all recorded image content and required masks before evaluation.

    Raises on altered images, missing files, cross-split duplicate content, or
    masks that are missing, empty, or differently sized. Mask locations are not
    hashed by this manifest schema; this checks their availability and geometry.
    """
    root = resolve_root(manifest)
    return _validate_records(root, read_manifest(manifest))


def training_paths(manifest: Path) -> tuple[list[Path], list[Path]]:
    root = resolve_root(manifest)
    records = [record for record in read_manifest(manifest) if record["split"] in {"train", "calibration"}]
    train = [root / record["image_path"] for record in records if record["split"] == "train"]
    calibration = [root / record["image_path"] for record in records if record["split"] == "calibration"]
    if not train or not calibration:
        raise ValueError("Manifest needs non-empty normal train and calibration splits.")
    _validate_records(root, records)
    return train, calibration


def generate_synthetic(root: Path, seed: int = 42) -> Path:
    """Generate 80 distinct images and 20 masks for software smoke tests only."""
    root = Path(root).expanduser().resolve()
    category = root / "synthetic"
    marker = category / "source.json"
    existing_images = _image_files(category)
    if existing_images and not marker.is_file():
        raise ValueError(f"Refusing to overwrite an existing unmarked dataset: {category}")
    if marker.is_file():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("kind") != "synthetic" or previous.get("generator") != "industrial_inspection.data.generate_synthetic":
            raise ValueError(f"Dataset source marker does not belong to this synthetic generator: {marker}")
    for relative in ("train/good", "test/good", "test/scratch", "ground_truth/scratch"):
        (category / relative).mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    side = 128
    y, x = np.mgrid[:side, :side]
    radius = np.sqrt((x - 64) ** 2 + (y - 64) ** 2)
    ring = (radius >= 22) & (radius <= 45)

    def make_normal() -> np.ndarray:
        base = np.where(ring, 180 + (x - 64) * 0.18, 48).astype(np.float32)
        base += rng.normal(0, 1.1) + rng.normal(0, 1.2, (side, side))
        return np.clip(np.stack([base, base + 1, base + 2], axis=-1), 0, 255).astype(np.uint8)

    for index in range(50):
        Image.fromarray(make_normal()).save(category / "train" / "good" / f"normal_{index:03}.png")
    for index in range(10):
        Image.fromarray(make_normal()).save(category / "test" / "good" / f"normal_{index:03}.png")
    positions = [((88, 54), (100, 68)), ((57, 58), (71, 70)), ((10, 22), (24, 36))]
    locations = []
    for index in range(20):
        image = Image.fromarray(make_normal())
        mask = Image.new("L", (side, side), 0)
        start, end = positions[index % 3]
        offset = int(rng.integers(-2, 3))
        line = [(start[0] + offset, start[1]), (end[0] + offset, end[1])]
        ImageDraw.Draw(image).line(line, fill=(215, 30, 35), width=4)
        ImageDraw.Draw(mask).line(line, fill=255, width=4)
        image.save(category / "test" / "scratch" / f"scratch_{index:03}.png")
        mask.save(category / "ground_truth" / "scratch" / f"scratch_{index:03}_mask.png")
        locations.append({"image": f"test/scratch/scratch_{index:03}.png", "location": ["ring", "center", "outside"][index % 3]})
    source = {
        "kind": "synthetic",
        "dataset": "Synthetic aligned washer smoke-test images",
        "generator": "industrial_inspection.data.generate_synthetic",
        "seed": int(seed),
        "image_size": [side, side],
        "counts": {"train_good": 50, "test_good": 10, "test_anomaly": 20},
        "purpose": "Pipeline smoke tests only; never report these metrics as industrial detection performance.",
        "defect_locations": locations,
    }
    marker.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return root
