"""Reproducible entry points for the course project."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

from PIL import Image


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def check_environment(output):
    packages = {}
    for package in ["numpy", "pillow", "opencv-python-headless", "scikit-learn", "torch", "torchvision", "gradio", "pytest"]:
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    result = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "packages": packages,
              "default_device": "cpu", "cpu_count": os.cpu_count(), "cuda_available": False}
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda_version"] = torch.version.cuda
        if result["cuda_available"]:
            result["gpu_name"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    save_json(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description="工业异常检测与可解释质量决策（课程原型）")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("prepare-demo", help="生成合成联调数据、清单、像素模型和演示样例")
    demo.add_argument("--root", type=Path, default=Path("."))
    prep = sub.add_parser("prepare", help="从原始MVTec结构建立训练/校准/test清单")
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--category", default="bottle")
    prep.add_argument("--output", type=Path, default=Path("data/manifests/bottle.csv"))
    prep.add_argument("--seed", type=int, default=42)
    train = sub.add_parser("train", help="只用正常训练图建立模型，正常校准图确定阈值")
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--method", choices=["pixel", "patch"], default="pixel")
    train.add_argument("--model-dir", type=Path, required=True)
    train.add_argument("--image-size", type=int, default=128)
    train.add_argument("--memory-size", type=int, default=2048)
    train.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    train.add_argument("--seed", type=int, default=42)
    ev = sub.add_parser("evaluate", help="锁定模型和阈值后评估test")
    ev.add_argument("--manifest", type=Path, required=True)
    ev.add_argument("--model-dir", type=Path, required=True)
    ev.add_argument("--output", type=Path, required=True)
    ev.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    predict = sub.add_parser("predict")
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--model-dir", type=Path, required=True)
    predict.add_argument("--rules", type=Path, default=Path("configs/quality_rules.json"))
    predict.add_argument("--output", type=Path, default=Path("results/prediction"))
    env = sub.add_parser("check-env")
    env.add_argument("--output", type=Path, default=Path("results/environment.json"))
    args = parser.parse_args()
    if args.command in ("train", "evaluate"):
        try:
            import torch
            torch.set_num_threads(min(4, os.cpu_count() or 1))
        except ImportError:
            pass
    if args.command == "check-env":
        result = check_environment(args.output)
    elif args.command == "prepare-demo":
        from .data import generate_synthetic, build_manifest
        root = args.root.resolve()
        data_root = generate_synthetic(root / "data/synthetic")
        manifest = root / "data/manifests/synthetic.csv"
        build_manifest(data_root, "synthetic", manifest)
        result = fit_model(manifest, "pixel", root / "models/synthetic_pixel")
        sample_dir = root / "samples/demo"
        sample_dir.mkdir(parents=True, exist_ok=True)
        for folder in ("good", "scratch"):
            for i, image in enumerate(sorted((data_root / "synthetic/test" / folder).glob("*.png"))[:3]):
                shutil.copy2(image, sample_dir / f"synthetic_{folder}_{i}.png")
    elif args.command == "prepare":
        from .data import build_manifest
        result = build_manifest(args.data_root, args.category, args.output, seed=args.seed)
    elif args.command == "train":
        result = fit_model(args.manifest, args.method, args.model_dir, args.image_size, args.memory_size, args.device, args.seed)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        result = evaluate(args.manifest, args.model_dir, args.output, args.device)
    elif args.command == "predict":
        from .pipeline import inspect_image
        image, result = inspect_image(Image.open(args.image), args.model_dir, args.rules)
        args.output.mkdir(parents=True, exist_ok=True)
        image.save(args.output / "overlay.png")
        save_json(args.output / "report.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def fit_model(manifest, method, model_dir, image_size=128, memory_size=2048, device="cpu", seed=42):
    from .data import training_paths
    from .detectors import PixelReferenceDetector, PatchMemoryDetector
    start = time.perf_counter()
    train, calibration = training_paths(manifest)
    detector = PixelReferenceDetector(image_size=image_size, seed=seed) if method == "pixel" else PatchMemoryDetector(image_size=image_size, memory_size=memory_size, device=device, seed=seed)
    detector.fit(train, calibration)
    detector.save(model_dir)
    card_path = model_dir / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    summary = json.loads(manifest.with_suffix(".summary.json").read_text(encoding="utf-8"))
    source = summary.get("dataset_source", "unknown")
    source_hashes = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() for name in ("detectors.py", "data.py", "cli.py")}
    card.update({"dataset_source": source, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
                 "manifest": str(manifest).replace("\\", "/"), "fit_seconds": round(time.perf_counter() - start, 3),
                 "seed": seed, "device": device, "source_sha256": source_hashes,
                 "scope": "Single category, aligned inspection images. Quality policy is illustrative and not independently validated."})
    save_json(card_path, card)
    return card


if __name__ == "__main__":
    main()
