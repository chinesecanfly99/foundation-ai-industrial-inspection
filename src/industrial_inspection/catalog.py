"""Fixed product registry for model, demonstration and teaching-policy routing.

The registry is deliberately local: a model card identifies its training data,
but never supplies a filesystem path to a quality policy.
"""
from pathlib import Path


PRODUCTS = {
    "bottle": {
        "label": "Bottle",
        "sample_dir": "bottle",
        "model_ids": ["bottle_patch", "bottle_pixel"],
        "default_model": "bottle_patch",
        "policy_file": "configs/quality_rules.json",
        "defect_labels": {"good": "Normal", "broken_large": "Large break", "broken_small": "Small break", "contamination": "Contamination"},
        "description": "Fixed-view bottle inspection: compare normal appearance, breaks and contamination.",
    },
    "metal_nut": {
        "label": "Metal nut",
        "sample_dir": "metal_nut",
        "model_ids": ["metal_nut_patch", "metal_nut_pixel"],
        "default_model": "metal_nut_patch",
        "policy_file": "configs/quality_rules_metal_nut.json",
        "defect_labels": {"good": "Normal", "bent": "Bent", "color": "Color anomaly", "flip": "Flipped", "scratch": "Scratch"},
        "description": "Metal nut inspection: compare shape, orientation, color and local surface appearance.",
    },
    "tile": {
        "label": "Tile surface",
        "sample_dir": "tile",
        "model_ids": ["tile_patch", "tile_pixel"],
        "default_model": "tile_patch",
        "policy_file": "configs/quality_rules_tile.json",
        "defect_labels": {"good": "Normal", "crack": "Crack", "glue_strip": "Glue residue", "gray_stroke": "Gray stroke", "oil": "Oil stain", "rough": "Rough surface"},
        "description": "Tile surface inspection: compare cracks, contamination and local texture changes.",
    },
    "synthetic": {
        "label": "Synthetic check",
        "sample_dir": "demo",
        "model_ids": ["synthetic_pixel"],
        "default_model": "synthetic_pixel",
        "policy_file": "configs/quality_rules_synthetic.json",
        "defect_labels": {"good": "Synthetic normal", "scratch": "Synthetic scratch"},
        "description": "Artificial washer images for software checks only; they do not represent real industrial detection performance.",
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
        raise ValueError(f"Unknown product category: {product_id}; specify --rules explicitly for a custom model.")
    return Path(project_root) / PRODUCTS[product_id]["policy_file"]
