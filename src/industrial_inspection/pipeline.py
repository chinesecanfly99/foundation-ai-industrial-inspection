"""Single inference path shared by CLI and browser UI."""
import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from .calibration import input_consistency, resolve_policy
from .detectors import _validate_rgb, load_detector
from .rules import decide, validate_policy


@lru_cache(maxsize=4)
def _load_model(path: str, card_modified: int, artifact_modified: int, artifact_size: int):
    # The artifact identity is part of the cache key: replacing model.npz
    # without touching the card must force a fresh checksum-verified load.
    del card_modified, artifact_modified, artifact_size
    return load_detector(Path(path), device="cpu")


def decision_facts(rgb: np.ndarray, prediction, policy: dict, calibration: dict | None = None,
                   source_size: tuple[int, int] | None = None) -> dict:
    """Measure policy facts in the detector's calibrated coordinate grid.

    When the policy enables the input gate, alignment and illumination are
    compared against the calibration reference. Aspect and resolution facts use
    the uploaded source size, not the working-resolution copy, so a large
    upload cannot slip through the resolution gate after the 1024px thumbnail.
    """
    rgb = _validate_rgb(rgb)
    x0, y0, x1, y1 = policy["critical_roi"]
    if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
        raise ValueError("关键区域必须为0到1的合法归一化矩形")
    model_heat = np.asarray(prediction.model_heatmap, dtype=np.float32)
    mask = model_heat > prediction.pixel_threshold
    h, w = model_heat.shape
    roi = np.zeros_like(mask)
    roi[int(y0*h):int(y1*h), int(x0*w):int(x1*w)] = True
    roi_pixels = max(int(roi.sum()), 1)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    facts = {
        "score": float(prediction.score), "image_threshold": float(prediction.image_threshold),
        "pixel_threshold": float(prediction.pixel_threshold),
        "anomaly_area_ratio": float(mask.mean()),
        "critical_overlap_ratio": float((mask & roi).sum() / max(int(mask.sum()), 1)),
        "critical_area_ratio": float((mask & roi).sum() / roi_pixels),
        "brightness": float(gray.mean()), "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "model_heatmap_size": [int(h), int(w)],
    }
    gate = policy.get("input_gate") or {}
    if gate.get("enabled"):
        if calibration is None or "_reference" not in calibration:
            raise ValueError("输入门控已启用，但缺少校准参考数据")
        consistency = input_consistency(np.asarray(rgb, dtype=np.float32) / 255.0, calibration["_reference"])
        width, height = source_size or (rgb.shape[1], rgb.shape[0])
        facts.update(consistency)
        facts["aspect_deviation"] = abs(float(width) / float(height) - 1.0)
        facts["resolution_ratio"] = max(float(width), float(height)) / float(policy["expected_long_edge"])
    return facts


def prepare_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    """Apply the serving-path preprocessing shared by the UI and evaluation."""
    if not isinstance(image, Image.Image):
        # Reuse the detector's numeric validation so float [0, 1], float RGB
        # [0, 255], and uint8 inputs all mean the same pixels here too.
        image = Image.fromarray(_validate_rgb(image))
    image = ImageOps.exif_transpose(image).convert("RGB")
    # Limit working resolution, preserve aspect ratio; the model has its own resize.
    image.thumbnail((1024, 1024))
    return np.asarray(image)


def inspect_image(image: Image.Image | np.ndarray, model_dir: Path, config_path: Path) -> tuple[Image.Image, dict]:
    start = time.perf_counter()
    if image is None:
        raise ValueError("请先上传图片")
    if not isinstance(image, Image.Image):
        # Reuse the detector's numeric validation so float [0, 1], float RGB
        # [0, 255], and uint8 inputs all mean the same pixels here too.
        image = Image.fromarray(_validate_rgb(image))
    if min(image.size) < 32 or image.width * image.height > 20_000_000:
        raise ValueError("图片至少32×32像素且不超过2000万像素")
    oriented = ImageOps.exif_transpose(image).convert("RGB")
    # Hash the EXIF-oriented upload before the working-resolution resize,
    # while retaining image_sha256 for the exact processed pixels.
    upload_sha256 = hashlib.sha256(np.asarray(oriented).tobytes()).hexdigest()
    upload_size = list(oriented.size)
    rgb = prepare_rgb(oriented)
    model_dir = Path(model_dir).resolve()
    card_path = model_dir / "model_card.json"
    artifact_path = model_dir / "model.npz"
    artifact_stat = artifact_path.stat()
    model = _load_model(str(model_dir), card_path.stat().st_mtime_ns, artifact_stat.st_mtime_ns, artifact_stat.st_size)
    card_bytes = card_path.read_bytes()
    card = json.loads(card_bytes)
    policy_bytes = Path(config_path).read_bytes()
    policy = json.loads(policy_bytes)
    category = str(card.get("dataset_source", "")).rstrip("/").split("/")[-1] or None
    effective_policy, calibration = resolve_policy(policy, Path(config_path), model_dir.name, category,
                                                   model_card=card,
                                                   model_card_sha256=hashlib.sha256(card_bytes).hexdigest())
    validate_policy(effective_policy)
    if calibration and calibration.get("manifest_sha256") != card.get("manifest_sha256"):
        raise ValueError("校准文件与模型训练清单不一致；请用当前清单和模型重新运行 calibrate-policy")
    prediction = model.predict(rgb)
    heat = np.asarray(prediction.heatmap, dtype=np.float32)
    facts = decision_facts(rgb, prediction, effective_policy, calibration, source_size=oriented.size)
    x0, y0, x1, y1 = effective_policy["critical_roi"]
    outcome = decide(facts, effective_policy)
    report = {**facts, **outcome, "model_id": model_dir.name,
              "score_ratio": float(prediction.score) / float(prediction.image_threshold),
              "effective_policy": effective_policy,
              "model_method": card.get("method", type(model).__name__),
              "dataset_source": card.get("dataset_source", "unknown"),
              "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
              "calibration_sha256": (calibration or {}).get("_sha256"),
              "model_card_sha256": hashlib.sha256(card_bytes).hexdigest(),
              "model_artifact_sha256": card.get("artifact_sha256"),
              "upload_sha256": upload_sha256, "upload_size": upload_size,
              "image_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
              "image_size": upload_size,
              "prepared_size": [int(rgb.shape[1]), int(rgb.shape[0])],
              "notice": "分数不是概率；热图表示视觉差异，规则解释质量建议，不解释缺陷物理成因。仅适用于模型对应类别和拍摄条件。"}
    # Color scale is fixed relative to calibration threshold, not min/max per image.
    normalized = np.clip(heat / max(prediction.pixel_threshold * 2, 1e-8), 0, 1)
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    overlay = (0.65 * rgb + 0.35 * color).astype(np.uint8)
    # Contours are a display aid, so draw them on the full-resolution heatmap;
    # the numeric area and ROI facts above stay in model coordinates.
    display_mask = heat > prediction.pixel_threshold
    contours, _ = cv2.findContours(display_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    full_h, full_w = overlay.shape[:2]
    cv2.rectangle(overlay, (int(x0*full_w), int(y0*full_h)), (int(x1*full_w), int(y1*full_h)), (255, 220, 60), 1)
    report["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return Image.fromarray(overlay), report
