"""Build deterministic, allowlisted v0.2.0 assets; verify CRC/SHA256 and optionally smoke-test offline.

Run from the project environment:
    python scripts/package_demo_assets.py --smoke-test --check-reproducible
This contains no virtual environment, raw dataset, IDE files, or credentials.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VERSION = "v0.2.0"
MODELS = ("bottle_patch", "bottle_pixel", "metal_nut_patch", "metal_nut_pixel",
          "tile_patch", "tile_pixel", "synthetic_pixel")
SAMPLE_COUNTS = {"bottle": 20, "metal_nut": 25, "tile": 30, "demo": 6}
POLICIES = {"bottle": "quality_rules.json", "metal_nut": "quality_rules_metal_nut.json",
            "tile": "quality_rules_tile.json", "synthetic": "quality_rules_synthetic.json"}
WEIGHTS = ".cache/torch/hub/checkpoints/resnet18-f37072fd.pth"
WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
LEGACY_BOTTLE_POLICY_SHA256 = "cf18c0bfbbc13c4153a03746cf7f390d2e709ab0ea19e07e327c42c3b5f1990b"
MANIFEST_NAME = "DEMO_ASSETS_MANIFEST.json"
README_NAME = "DEMO_ASSETS_README.md"
FIXED_ZIP_TIME = (1980, 1, 1, 0, 0, 0)

README = """# 三类演示资产 v0.2.0

将本 ZIP 的内容直接解压合并到**对应版本的项目根目录**，与 app/、src/、pyproject.toml 同级；不要额外套一层文件夹。请使用包含三类产品选择器、catalog.py 与独立质量规则的 v0.2.0 源码。模型包本身不含启动源码与 Python 依赖。

包含 7 个模型：瓶口、金属螺母、瓷砖各一个特征记忆主模型和一个像素基线，另一个 synthetic_pixel 供联调；官方 ResNet18 权重缓存；75 张真实样例、6 张合成联调图片；样例元数据和来源说明；4 个类别各自的教学质量规则。

首次安装依赖需要联网。依赖装好后，在项目根目录运行 `python -m app.app --host 127.0.0.1 --port 7860`，可使用包内图片和模型离线推理，无需下载完整原始数据或重新训练。若电脑设置过 TORCH_HOME，请将其指向本项目 `.cache/torch`，或取消该环境变量，使程序找到包内骨干缓存。每个产品只使用对应模型和规则；高级选项中的合成模型不代表工业效果。

完整训练/独立评估仍需另备原始数据。模型卡中的历史 manifest 路径只用于训练来源记录，离线推理不会读取这些路径。不要把原电脑的 .venv 复制到其他电脑。详细安装与演示步骤见源码内 README_组员运行.md、README_演示.md。

DEMO_ASSETS_MANIFEST.json 逐一记录本包文件长度和 SHA256（清单自身除外，以免自引用），并记录配套源码哈希。发布的 ZIP 另有整体 SHA256，可验证下载完整性。ZIP 使用固定顺序、时间戳和权限，相同源文件在相同 Python/zlib 下重复构建得到相同字节。

真实样例来自 MVTec AD，作者 MVTec Software GmbH，许可 CC BY-NC-SA 4.0，仅用于非商业课程研究和演示；保留来源并遵守相同方式共享要求。来源与镜像校验范围见 samples/README.md。ResNet18 来自 torchvision 的 IMAGENET1K_V1 官方预训练权重：https://download.pytorch.org/models/resnet18-f37072fd.pth 。项目的特征记忆方法为简化教学实现，不是官方 PatchCore 基准复现。

本包不包含 data/raw、.venv、.idea、凭证、用户配置或完整实验输出。异常分数不是概率，规则未经实际工厂业务独立验证。
"""


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_bytes(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def source_bytes(root: Path, relative: str) -> bytes:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError(f"Missing or unsafe required asset: {relative}")
    return path.read_bytes()


def payloads_for(root: Path) -> tuple[dict[str, bytes], dict]:
    root = root.resolve()
    payloads = {}
    for model in MODELS:
        artifact_path = f"models/{model}/model.npz"
        card_path = f"models/{model}/model_card.json"
        artifact, card_bytes = source_bytes(root, artifact_path), source_bytes(root, card_path)
        card = json.loads(card_bytes)
        category, _, method = model.rpartition("_")
        expected_source = "synthetic" if category == "synthetic" else f"mvtec_ad/{category}"
        expected_method = "patch_memory" if method == "patch" else "pixel_reference"
        if card.get("artifact_sha256") != digest_bytes(artifact):
            raise ValueError(f"Artifact checksum differs from its card: {model}")
        if card.get("dataset_source") != expected_source or card.get("method") != expected_method:
            raise ValueError(f"Model category/method mismatch: {model}")
        payloads[artifact_path], payloads[card_path] = artifact, card_bytes
    payloads[WEIGHTS] = source_bytes(root, WEIGHTS)
    if digest_bytes(payloads[WEIGHTS]) != WEIGHTS_SHA256:
        raise ValueError("ResNet18 official cached weight checksum differs")
    for category, count in SAMPLE_COUNTS.items():
        sample_dir = (root / "samples" / category).resolve()
        if not sample_dir.is_relative_to(root):
            raise ValueError("Sample directory escapes the project")
        pictures = sorted(sample_dir.glob("*.png"))
        if len(pictures) != count:
            raise ValueError(f"Expected {count} samples in {category}, found {len(pictures)}")
        for picture in pictures:
            relative = picture.relative_to(root).as_posix()
            payloads[relative] = source_bytes(root, relative)
    for relative in ("samples/README.md", "samples/samples.json",
                     "samples/bottle/samples.json", "samples/metal_nut/samples.json", "samples/tile/samples.json"):
        payloads[relative] = source_bytes(root, relative)
    sample_metadata = json.loads(payloads["samples/samples.json"])
    if sample_metadata.get("sample_count") != 75 or len(sample_metadata.get("samples", [])) != 75:
        raise ValueError("Sample metadata does not cover the 75 real examples")
    described = set()
    for record in sample_metadata["samples"]:
        name = "samples/" + record["sample_path"]
        if name in described or name not in payloads or digest_bytes(payloads[name]) != record["file_sha256"]:
            raise ValueError(f"Sample metadata/hash mismatch: {name}")
        described.add(name)
    real_names = {name for name in payloads if name.endswith(".png") and not name.startswith("samples/demo/")}
    if described != real_names:
        raise ValueError("Real sample files and metadata differ")
    for category, filename in POLICIES.items():
        relative = f"configs/{filename}"
        value = source_bytes(root, relative)
        declared_category = json.loads(value).get("product_category")
        # Keep the frozen v0.1 bottle policy byte-for-byte; its registered path
        # binds it to bottle even though this older JSON has no category field.
        legacy_bottle = (category == "bottle" and declared_category is None
                         and digest_bytes(value) == LEGACY_BOTTLE_POLICY_SHA256)
        if declared_category != category and not legacy_bottle:
            raise ValueError(f"Built-in policy does not match its category: {relative}")
        payloads[relative] = value
    payloads[README_NAME] = README.encode("utf-8")
    compatible_source = ["pyproject.toml"]
    compatible_source += [path.relative_to(root).as_posix() for path in sorted((root / "app").glob("*.py"))]
    compatible_source += [path.relative_to(root).as_posix() for path in sorted((root / "src/industrial_inspection").glob("*.py"))]
    manifest = {
        "schema_version": 1, "version": VERSION, "model_count": len(MODELS), "models": list(MODELS),
        "sample_counts": SAMPLE_COUNTS, "real_sample_count": 75, "synthetic_sample_count": 6,
        "file_count_excluding_manifest": len(payloads), "manifest_self_excluded": True,
        "compatible_source_sha256": {name: digest_bytes(source_bytes(root, name)) for name in compatible_source},
        "files": [{"path": name, "size": len(payloads[name]), "sha256": digest_bytes(payloads[name])} for name in sorted(payloads)],
        "layout": "Merge ZIP contents directly into the matching repository root.",
        "offline": "Requires locally installed Python dependencies; does not require raw data, retraining, or network for inference.",
    }
    payloads[MANIFEST_NAME] = json_bytes(manifest)
    return payloads, manifest


def build_zip(path: Path, payloads: dict[str, bytes]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    try:
        with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for name in sorted(payloads):
                item = zipfile.ZipInfo(name, date_time=FIXED_ZIP_TIME)
                item.create_system = 3
                item.external_attr = 0o100644 << 16
                archive.writestr(item, payloads[name], compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)
        verify_zip(partial)
        partial.replace(path)
    finally:
        if partial.exists():
            partial.unlink()


def verify_zip(path: Path) -> dict:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive members")
        for name in names:
            relative = PurePosixPath(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
                raise ValueError(f"Unsafe archive member: {name}")
        if archive.testzip() is not None:
            raise ValueError("ZIP CRC check failed")
        manifest = json.loads(archive.read(MANIFEST_NAME))
        expected = {entry["path"] for entry in manifest["files"]}
        if len(expected) != len(manifest["files"]) or set(names) != expected | {MANIFEST_NAME}:
            raise ValueError("Archive membership differs from its SHA256 manifest")
        for record in manifest["files"]:
            value = archive.read(record["path"])
            if len(value) != record["size"] or digest_bytes(value) != record["sha256"]:
                raise ValueError(f"Manifest verification failed: {record['path']}")
    return {"zip_crc_ok": True, "manifest_sha256_ok": True, "archive_entries": len(names),
            "manifest_files_verified": len(expected), "manifest_sha256": digest_bytes(json_bytes(manifest))}


OFFLINE_SMOKE = r'''
from pathlib import Path
import json, os, socket, sys, urllib.request
isolated, original = map(lambda x: Path(x).resolve(), sys.argv[1:3])
sys.path.insert(0, str(isolated / "src"))
os.environ["TORCH_HOME"] = str(isolated / ".cache/torch")
os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
attempts, forbidden = [], []
def deny_network(*args, **kwargs):
    attempts.append(str(args[0]) if args else "network")
    raise AssertionError("Network is disabled during offline smoke validation")
socket.socket.connect = deny_network
urllib.request.urlopen = deny_network
blocked = [(original / name).resolve() for name in ("data", "models", "samples", "configs", ".cache", "src", "app")]
def audit(event, args):
    if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        path = Path(os.fsdecode(args[0])).resolve()
        if any(path.is_relative_to(folder) for folder in blocked) and not path.is_relative_to(isolated):
            forbidden.append(str(path))
            raise AssertionError("Original project asset/data access is forbidden: " + str(path))
sys.addaudithook(audit)
import requests
requests.sessions.Session.request = deny_network
import torch
torch.set_num_threads(4)
torch.hub.download_url_to_file = deny_network
from PIL import Image
from industrial_inspection import pipeline, detectors
assert Path(pipeline.__file__).resolve().is_relative_to(isolated / "src")
assert Path(detectors.__file__).resolve().is_relative_to(isolated / "src")
assert not (isolated / "data").exists()
cases = []
for product in ("bottle", "metal_nut", "tile", "synthetic"):
    for method in (("pixel",) if product == "synthetic" else ("patch", "pixel")):
        model_id = f"{product}_{method}"
        sample = isolated / "samples" / ("demo/synthetic_good_0.png" if product == "synthetic" else f"{product}/good_000.png")
        with Image.open(sample) as opened:
            overlay, report = pipeline.inspect_image(opened.copy(), isolated / "models" / model_id)
        assert report["product_category"] == product and report["model_id"] == model_id
        assert overlay.size[0] > 0 and overlay.size[1] > 0
        cases.append({key: report[key] for key in ("product_category", "model_id", "policy_name", "decision", "score", "image_sha256", "latency_ms")})
assert not attempts and not forbidden
print(json.dumps({"passed": True, "cases": cases, "case_count": len(cases),
                  "network_disabled": True, "network_attempts": attempts,
                  "original_project_asset_reads": forbidden, "raw_data_present": False,
                  "source_root": str(isolated / "src"), "torch_home": os.environ["TORCH_HOME"]}, ensure_ascii=False))
'''


def smoke_test(path: Path, root: Path) -> dict:
    # The temp folder deliberately has a different root and no raw dataset.
    with TemporaryDirectory(prefix="inspection_assets_offline_") as folder:
        isolated = Path(folder).resolve()
        verify_zip(path)
        with zipfile.ZipFile(path) as archive:
            archive.extractall(isolated)
        package = root / "src/industrial_inspection"
        for source in package.rglob("*.py"):
            relative = source.relative_to(root)
            destination = isolated / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        result = subprocess.run([sys.executable, "-B", "-c", OFFLINE_SMOKE, str(isolated), str(root)],
                                cwd=isolated, capture_output=True, text=True, encoding="utf-8", timeout=180)
        if result.returncode != 0:
            raise RuntimeError(f"Isolated offline inference failed:\n{result.stdout}\n{result.stderr}")
        value = json.loads(result.stdout.strip().splitlines()[-1])
        # Record isolation assertions without publishing machine/user-specific
        # temporary paths. The child verified real import and asset locations.
        value["source_root"] = "<isolated>/src"
        value["torch_home"] = "<isolated>/.cache/torch"
        value["source_import_verified_in_isolated_directory"] = True
        value["warnings"] = result.stderr.strip()
        return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--verify-only", type=Path, help="Verify an existing asset ZIP without rebuilding")
    parser.add_argument("--smoke-test", action="store_true", help="Use installed dependencies for isolated offline inference of all seven models")
    parser.add_argument("--check-reproducible", action="store_true", help="Build twice and require identical ZIP SHA256")
    args = parser.parse_args()
    root = args.root.resolve()
    path = args.verify_only.resolve() if args.verify_only else (args.output.resolve() if args.output else root / f"deliverables/demo-assets-{VERSION}.zip")
    if not args.verify_only:
        payloads, _ = payloads_for(root)
        build_zip(path, payloads)
        if args.check_reproducible:
            with TemporaryDirectory(prefix="inspection_assets_repro_") as folder:
                second = Path(folder) / path.name
                build_zip(second, payloads)
                if digest_bytes(path.read_bytes()) != digest_bytes(second.read_bytes()):
                    raise ValueError("Repeated build did not produce identical bytes")
    validation = {"version": VERSION, "archive": str(path), "bytes": path.stat().st_size,
                  "sha256": digest_bytes(path.read_bytes()), "verified_utc": datetime.now(timezone.utc).isoformat(),
                  **verify_zip(path), "reproducible_double_build": bool(args.check_reproducible and not args.verify_only)}
    if args.smoke_test:
        validation["offline_smoke"] = smoke_test(path, root)
    record = path.with_suffix(".validation.json")
    record.write_bytes(json_bytes(validation))
    evidence = root / "results/three_products/package_verification.json"
    evidence.parent.mkdir(parents=True, exist_ok=True)
    evidence.write_bytes(json_bytes(validation))
    path.with_suffix(path.suffix + ".sha256").write_text(f"{validation['sha256']}  {path.name}\n", encoding="ascii")
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
