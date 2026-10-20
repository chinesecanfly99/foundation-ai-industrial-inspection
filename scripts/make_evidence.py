"""Create fixed-order demo samples and visual evidence from already frozen models."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from industrial_inspection.pipeline import inspect_image


def make(root: Path):
    root = root.resolve()
    data_root = root / "data/raw/mvtec_ad"
    output = root / "results/examples"
    output.mkdir(parents=True, exist_ok=True)
    samples = root / "samples/bottle"
    samples.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default(size=18)
    small = ImageFont.load_default(size=14)
    width, top, row_height = 1050, 75, 395
    canvas = Image.new("RGB", (width, top + row_height * 4), "#f4f7fa")
    draw = ImageDraw.Draw(canvas)
    for x, title in zip((15, 365, 715), ("Input image", "Ground-truth defect mask", "Model heatmap + demo ROI")):
        draw.text((x, 16), title, font=font, fill="#123a4d")
    reports = []
    for row, kind in enumerate(("good", "broken_large", "broken_small", "contamination")):
        path = data_root / "bottle/test" / kind / "000.png"
        shutil.copy2(path, samples / f"{kind}_000.png")
        rgb = Image.open(path).convert("RGB")
        overlay, report = inspect_image(rgb, root / "models/bottle_patch", root / "configs/quality_rules.json")
        overlay.save(output / f"{kind}_overlay.png")
        (output / f"{kind}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        ground_truth = rgb.copy()
        if kind != "good":
            mask = np.asarray(Image.open(data_root / f"bottle/ground_truth/{kind}/000_mask.png").convert("L")) > 0
            image_arr = np.asarray(rgb).copy()
            image_arr[mask] = (image_arr[mask] * 0.35 + np.array([255, 40, 40]) * .65).astype(np.uint8)
            ground_truth = Image.fromarray(image_arr)
        for x, picture in zip((0, 350, 700), (rgb, ground_truth, overlay)):
            picture.thumbnail((340, 340))
            canvas.paste(picture, (x + 5, top + row * row_height))
        y = top + row * row_height + 345
        draw.text((10, y), kind + "/000", font=font, fill="#123a4d")
        draw.text((360, y), "Normal" if kind == "good" else "Dataset pixel annotation", font=small, fill="#4f6472")
        draw.text((710, y), f"score {report['score']:.4f} / threshold {report['image_threshold']:.4f}", font=small, fill="#123a4d")
        draw.text((710, y + 20), f"Teaching policy: {report['decision']}", font=small, fill="#4f6472")
        reports.append({"image": f"bottle/test/{kind}/000.png", "decision": report["decision"], "score": report["score"], "threshold": report["image_threshold"]})
    canvas.save(output / "bottle_examples.png")
    (output / "example_index.json").write_text(json.dumps({"selection": "First filename of each class, fixed before inspecting scores; not representative performance statistics", "examples": reports}, indent=2), encoding="utf-8")
    with (root / "results/bottle_patch/predictions.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    failures = [r for r in rows if r["label"] == "1" and r["predicted_anomaly"] == "0"]
    if failures:
        with (root / "results/bottle_patch/false_negatives.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(failures)
    print(json.dumps({"examples": len(reports), "false_negatives": len(failures), "figure": str(output / "bottle_examples.png")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    make(parser.parse_args().root)
