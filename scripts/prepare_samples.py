"""Copy deterministic first-N test examples for the three product categories.

Selection uses test/<defect>/ file-name order only, never model predictions.
Existing files are retained; a destination with different content is rejected.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

CATEGORIES = ("bottle", "metal_nut", "tile")
SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def prepare_samples(root: Path, output: Path, categories: list[str] | tuple[str, ...] = CATEGORIES, per_type: int = 5) -> dict:
    if not categories or any(category not in CATEGORIES for category in categories):
        raise ValueError(f"Supported categories: {CATEGORIES}")
    if not 1 <= per_type <= 100:
        raise ValueError("per_type must be between 1 and 100")
    root, output = Path(root).expanduser().resolve(), Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    all_records = []
    summaries = {}
    for category in dict.fromkeys(categories):
        test = (root / category / "test").resolve()
        if not test.is_relative_to(root) or not test.is_dir():
            raise FileNotFoundError(f"Missing safe category test directory: {test}")
        target_root = (output / category).resolve()
        if not target_root.is_relative_to(output):
            raise ValueError("Unsafe sample category directory")
        target_root.mkdir(parents=True, exist_ok=True)
        records = []
        for defect in sorted(path for path in test.iterdir() if path.is_dir()):
            candidates = sorted((path for path in defect.iterdir() if path.is_file() and path.suffix.lower() in SUFFIXES), key=lambda path: path.name)
            for source in candidates[:per_type]:
                if not source.resolve().is_relative_to(test):
                    raise ValueError("Sample source escapes test directory")
                target = target_root / f"{defect.name}_{source.stem}{source.suffix.lower()}"
                digest = hashlib.sha256(source.read_bytes()).hexdigest()
                if target.exists():
                    if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                        raise ValueError(f"Existing sample content differs; refusing overwrite: {target}")
                else:
                    shutil.copy2(source, target)
                records.append({"sample_path": target.relative_to(output).as_posix(),
                                "source_path": source.relative_to(root).as_posix(),
                                "category": category, "defect_type": defect.name,
                                "label": int(defect.name != "good"), "file_sha256": digest})
        if not records or not any(record["label"] == 0 for record in records):
            raise ValueError(f"No normal test examples found for {category}")
        summary = {"category": category, "selection": "First N filenames in each original test defect directory; no model-based selection",
                   "per_type": per_type, "sample_count": len(records),
                   "defect_counts": dict(Counter(record["defect_type"] for record in records)), "samples": records}
        (target_root / "samples.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summaries[category] = {key: value for key, value in summary.items() if key != "samples"}
        all_records.extend(records)
    combined = {"schema_version": 1, "selection": "File name ordering only; no model result selection", "categories": summaries,
                "sample_count": len(all_records), "samples": all_records,
                "note": "Original test images remain test-only; samples are copies for UI demonstration, not extra evaluation records."}
    (output / "samples.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"sample_count": len(all_records), "categories": summaries}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/raw/mvtec_ad"))
    parser.add_argument("--output", type=Path, default=Path("samples"))
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES, default=list(CATEGORIES))
    parser.add_argument("--per-type", type=int, default=5)
    args = parser.parse_args()
    print(json.dumps(prepare_samples(args.root, args.output, args.categories, args.per_type), ensure_ascii=False, indent=2))
