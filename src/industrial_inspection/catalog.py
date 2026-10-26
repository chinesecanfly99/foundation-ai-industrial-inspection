"""Fixed product registry for model, demonstration and teaching-policy routing.

The registry is deliberately local: a model card identifies its training data,
but never supplies a filesystem path to a quality policy.
"""
from pathlib import Path


PRODUCTS = {
    "bottle": {
        "label": "瓶口",
        "sample_dir": "bottle",
        "model_ids": ["bottle_patch", "bottle_pixel"],
        "default_model": "bottle_patch",
        "policy_file": "configs/quality_rules.json",
        "defect_labels": {"good": "正常", "broken_large": "大块破损", "broken_small": "小缺口", "contamination": "污染或异物"},
        "description": "固定视角瓶口：比较正常瓶口、破损及污染的视觉差异。",
    },
    "metal_nut": {
        "label": "金属螺母",
        "sample_dir": "metal_nut",
        "model_ids": ["metal_nut_patch", "metal_nut_pixel"],
        "default_model": "metal_nut_patch",
        "policy_file": "configs/quality_rules_metal_nut.json",
        "defect_labels": {"good": "正常", "bent": "变形", "color": "颜色异常", "flip": "翻面", "scratch": "划痕"},
        "description": "金属螺母：比较形状、方向、颜色和局部表面的视觉差异。",
    },
    "tile": {
        "label": "瓷砖表面",
        "sample_dir": "tile",
        "model_ids": ["tile_patch", "tile_pixel"],
        "default_model": "tile_patch",
        "policy_file": "configs/quality_rules_tile.json",
        "defect_labels": {"good": "正常", "crack": "裂纹", "glue_strip": "胶条残留", "gray_stroke": "灰色划痕", "oil": "油污", "rough": "粗糙表面"},
        "description": "瓷砖纹理表面：比较裂纹、污染和局部纹理变化。",
    },
    "synthetic": {
        "label": "合成联调",
        "sample_dir": "demo",
        "model_ids": ["synthetic_pixel"],
        "default_model": "synthetic_pixel",
        "policy_file": "configs/quality_rules_synthetic.json",
        "defect_labels": {"good": "合成正常", "scratch": "合成划痕"},
        "description": "人工生成的垫圈，仅用于软件流程联调，不代表真实工业检测性能。",
    },
}

REAL_PRODUCTS = ("bottle", "metal_nut", "tile")


def product_id_from_card(card: dict) -> str | None:
    """Identify supported training categories without guessing from filenames."""
    source = card.get("dataset_source")
    if source == "synthetic":
        return "synthetic"
    for product_id in REAL_PRODUCTS:
        if source == f"mvtec_ad/{product_id}":
            return product_id
    return None


def policy_path_for_product(project_root: Path, product_id: str) -> Path:
    """Return only a policy explicitly registered for a supported product."""
    if product_id not in PRODUCTS:
        raise ValueError(f"未知产品类别：{product_id}；请为自定义模型显式指定 --rules。")
    return Path(project_root) / PRODUCTS[product_id]["policy_file"]
