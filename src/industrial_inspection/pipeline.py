"""Single inference path shared by CLI and browser UI."""
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageOps

from .catalog import PRODUCTS, policy_path_for_product, product_id_from_card
from .detectors import load_detector
from .rules import decide


@lru_cache(maxsize=8)
def _load_model(path: str, modified: int):
    return load_detector(Path(path), device="cpu")


def inspect_image(image: Image.Image | np.ndarray, model_dir: Path, config_path: Path | None = None) -> tuple[Image.Image, dict]:
    start = time.perf_counter()
    if image is None:
        raise ValueError("请先上传图片")
    if not isinstance(image, Image.Image):
        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    if min(image.size) < 32 or image.width * image.height > 20_000_000:
        raise ValueError("图片至少32×32像素且不超过2000万像素")
    image = ImageOps.exif_transpose(image).convert("RGB")
    # Limit working resolution, preserve aspect ratio; model has its own resize.
    image.thumbnail((1024, 1024))
    rgb = np.asarray(image)
    model_dir = Path(model_dir).resolve()
    card_path = model_dir / "model_card.json"
    card = json.loads(card_path.read_text(encoding="utf-8"))
    product_id = product_id_from_card(card)
    project_root = Path(__file__).resolve().parents[2]
    if config_path is None:
        if product_id is None:
            raise ValueError("模型卡的 dataset_source 不是已支持的产品类别，无法自动选择质量策略；请显式指定 --rules 配置文件。")
        config_path = policy_path_for_product(project_root, product_id)
    config_path = Path(config_path).resolve()
    # Legacy bottle policy predates product_category metadata. Its registered
    # location still binds it to bottle; only external copies are custom rules.
    registered_category = next((category for category in PRODUCTS
                                if config_path == policy_path_for_product(project_root, category).resolve()), None)
    if registered_category is not None and registered_category != product_id:
        raise ValueError(f"内置质量策略类别 {registered_category} 与模型类别 {product_id or '未知'} 不一致，请选择该产品对应的模型和策略。")
    policy_bytes = Path(config_path).read_bytes()
    policy = json.loads(policy_bytes)
    policy_category = policy.get("product_category")
    if policy_category is not None and policy_category != product_id:
        raise ValueError(f"质量策略类别 {policy_category} 与模型类别 {product_id or '未知'} 不一致，请选择该产品对应的模型和策略。")
    x0, y0, x1, y1 = policy["critical_roi"]
    if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
        raise ValueError("关键区域必须为0到1的合法归一化矩形")
    model = _load_model(str(model_dir), card_path.stat().st_mtime_ns)
    prediction = model.predict(rgb)
    heat = np.asarray(prediction.heatmap, dtype=np.float32)
    mask = heat > prediction.pixel_threshold
    h, w = mask.shape
    roi = np.zeros_like(mask)
    roi[int(y0*h):int(y1*h), int(x0*w):int(x1*w)] = True
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    facts = {
        "score": float(prediction.score), "image_threshold": float(prediction.image_threshold),
        "pixel_threshold": float(prediction.pixel_threshold),
        "anomaly_area_ratio": float(mask.mean()),
        "critical_overlap_ratio": float((mask & roi).sum() / max(int(mask.sum()), 1)),
        "brightness": float(gray.mean()), "sharpness": float(cv2.Laplacian(gray, cv2.CV_64F).var()),
    }
    report = {**facts, **decide(facts, policy), "model_id": model_dir.name,
              "model_method": card.get("method", type(model).__name__),
              "dataset_source": card.get("dataset_source", "unknown"),
              "product_category": product_id or "unknown",
              "product_label": PRODUCTS[product_id]["label"] if product_id else "自定义类别",
              "policy_name": Path(config_path).name,
              "policy_description": policy.get("description", "自定义质量策略"),
              "policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
              "image_sha256": hashlib.sha256(rgb.tobytes()).hexdigest(),
              "image_size": list(image.size),
              "notice": "分数不是概率；热图表示视觉差异，规则解释质量建议，不解释缺陷物理成因。仅适用于模型对应类别和拍摄条件。"}
    # Color scale is fixed relative to calibration threshold, not min/max per image.
    normalized = np.clip(heat / max(prediction.pixel_threshold * 2, 1e-8), 0, 1)
    color = cv2.applyColorMap((normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    overlay = (0.65 * rgb + 0.35 * color).astype(np.uint8)
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 255, 255), 1)
    cv2.rectangle(overlay, (int(x0*w), int(y0*h)), (int(x1*w), int(y1*h)), (255, 220, 60), 1)
    report["latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    return Image.fromarray(overlay), report
