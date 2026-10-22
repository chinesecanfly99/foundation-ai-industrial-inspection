"""Calibrate input-consistency and pass-area thresholds from normal images only.

The camera/part pose reference and the input gate are derived from the training
normals, while gate thresholds and pass-area limits come from the held-out
calibration normals. No test image or defect label influences these numbers.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps


def _load_rgb(path: Path, image_size: int) -> np.ndarray:
    with Image.open(path) as image:
        oriented = ImageOps.exif_transpose(image).convert("RGB")
        return np.asarray(
            oriented.resize((image_size, image_size), Image.Resampling.BILINEAR), dtype=np.float32
        ) / 255.0


def build_reference(train_paths: Sequence[Path], image_size: int = 128) -> np.ndarray:
    """Average the normal training images into a camera/pose reference."""
    if not train_paths:
        raise ValueError("构建参考图至少需要一张正常训练图。")
    stack = [_load_rgb(Path(path), image_size) for path in train_paths]
    reference = np.mean(stack, axis=0).astype(np.float32)
    if not np.isfinite(reference).all():
        raise ValueError("参考图包含非有限数值。")
    return reference


def input_consistency(rgb: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    """Return pose agreement (NCC) and global illumination gain.

    NCC is invariant to affine intensity changes, so it measures structure and
    alignment. The gain is the median per-channel mean ratio, which flags a
    uniform exposure or lighting shift that the residual baseline cannot absorb.
    """
    image = np.asarray(rgb, dtype=np.float32)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("输入一致性检查需要 RGB 图像。")
    ref = np.asarray(reference, dtype=np.float32)
    if image.shape[:2] != ref.shape[:2]:
        image = np.asarray(
            Image.fromarray(np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)).resize(
                (ref.shape[1], ref.shape[0]), Image.Resampling.BILINEAR
            ), dtype=np.float32
        ) / 255.0
    else:
        image = np.clip(image, 0.0, 1.0)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    reference_gray = cv2.cvtColor(ref, cv2.COLOR_RGB2GRAY)
    if gray.std() < 1e-6 or reference_gray.std() < 1e-6:
        alignment = 0.0
    else:
        alignment = float(np.corrcoef(gray.ravel(), reference_gray.ravel())[0, 1])
    image_mean = image.mean(axis=(0, 1))
    reference_mean = np.maximum(ref.mean(axis=(0, 1)), 1e-6)
    gain = float(np.median(image_mean / reference_mean))
    return {"alignment_ncc": alignment, "illumination_gain": gain}


def _percentile(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q))


def _pass_area_cap(policy: dict, key: str) -> float | None:
    """Return the policy ceiling that a calibrated pass limit may only tighten."""
    if key in policy:
        return float(policy[key])
    if "min_anomaly_area" in policy:
        return float(policy["min_anomaly_area"])
    return None


def calibrate(
    manifest,
    model_dir: Path | None,
    output: Path,
    image_size: int | None = None,
    alignment_margin: float = 0.02,
    illumination_margin: float = 0.03,
    area_percentile: float = 90.0,
    area_multiplier: float = 1.2,
    rules: Path | None = None,
) -> dict[str, Any]:
    """Write a calibration artifact and return its summary."""
    from .data import read_manifest, resolve_root, training_paths
    from .detectors import load_detector
    from .pipeline import decision_facts

    if not math.isfinite(alignment_margin) or alignment_margin < 0:
        raise ValueError("alignment_margin 必须是有限非负数。")
    if not math.isfinite(illumination_margin) or illumination_margin < 0:
        raise ValueError("illumination_margin 必须是有限非负数。")
    if not 0 <= area_percentile <= 100:
        raise ValueError("area_percentile 必须在 0 到 100 之间。")
    if not math.isfinite(area_multiplier) or area_multiplier <= 0:
        raise ValueError("area_multiplier 必须是有限正数。")
    manifest = Path(manifest)
    root = resolve_root(manifest)
    rows = read_manifest(manifest)
    train, _calibration = training_paths(manifest)
    policy = json.loads(Path(rules).read_text(encoding="utf-8")) if rules else {}
    roi = [float(value) for value in policy.get("critical_roi", [0.2, 0.2, 0.8, 0.8])]
    model_card = None
    if model_dir is not None:
        model_card = json.loads((Path(model_dir) / "model_card.json").read_text(encoding="utf-8"))
    if image_size is None:
        image_size = int(model_card.get("image_size", 128)) if model_card else 128
    image_size = int(image_size)
    if not 32 <= image_size <= 1024:
        raise ValueError("image_size 必须在 32 到 1024 之间。")
    summary_path = manifest.with_suffix(".summary.json")
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
    category = str(summary.get("category") or manifest.stem)
    reference = build_reference(train, image_size)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    reference_path = output.parent / f"reference_{category.replace('/', '_')}_{image_size}.npz"
    np.savez_compressed(reference_path, reference_rgb=reference)

    train_long_edges = []
    for row in rows:
        if row["split"] != "train":
            continue
        with Image.open(root / row["image_path"]) as image:
            oriented = ImageOps.exif_transpose(image)
            train_long_edges.append(max(oriented.size))
    edge_values = np.array(train_long_edges, dtype=np.float64)
    normal_facts = []
    for row in rows:
        if row["split"] != "calibration":
            continue
        rgb = _load_rgb(root / row["image_path"], image_size)
        normal_facts.append(input_consistency(rgb, reference))
    ncc = np.array([item["alignment_ncc"] for item in normal_facts])
    gain = np.array([item["illumination_gain"] for item in normal_facts])
    if len(ncc) < 2:
        raise ValueError("至少需要两张正常校准图。")

    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    artifact: dict[str, Any] = {
        "schema_version": 1,
        "manifest_sha256": manifest_sha256,
        "category": category,
        "critical_roi": roi,
        "rules_sha256": hashlib.sha256(Path(rules).read_bytes()).hexdigest() if rules else None,
        "image_size": image_size,
        "reference": {
            "path": reference_path.name,
            "array": "reference_rgb",
            "sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "source": "mean of manifest train split normals",
        },
        "input_gate": {
            "long_edge_median": float(np.median(edge_values)),
            "long_edge_min": float(edge_values.min()),
            "long_edge_max": float(edge_values.max()),
            "long_edge_source": "long edge of manifest train split images",
            "calibration_count": len(ncc),
            "alignment_ncc_min": float(ncc.min()),
            "alignment_ncc_p01": _percentile(ncc, 1),
            "alignment_ncc_median": _percentile(ncc, 50),
            "illumination_gain_min": float(gain.min()),
            "illumination_gain_max": float(gain.max()),
            "alignment_ncc_threshold": float(np.floor((ncc.min() - alignment_margin) * 1000) / 1000),
            "illumination_gain_min_threshold": float(np.floor((gain.min() - illumination_margin) * 1000) / 1000),
            "illumination_gain_max_threshold": float(np.ceil((gain.max() + illumination_margin) * 1000) / 1000),
        },
        "models": {},
    }

    if model_dir is not None:
        model_dir = Path(model_dir)
        card = model_card or json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
        if card.get("manifest_sha256") != manifest_sha256:
            raise ValueError(
                f"模型 {model_dir.name} 不是用该 manifest 训练的；请先用同一划分重新训练，再校准通过面积门槛")
        if output.is_file():
            previous = json.loads(output.read_text(encoding="utf-8"))
            # Area thresholds depend on the split, ROI, percentile, and
            # multiplier; keep other models only when all four still match.
            if (previous.get("manifest_sha256") == manifest_sha256
                    and [float(value) for value in previous.get("critical_roi", [])] == roi):
                artifact["models"] = {
                    name: values for name, values in (previous.get("models") or {}).items()
                    if values.get("area_percentile") == area_percentile
                    and values.get("area_multiplier") == area_multiplier
                }
        model = load_detector(model_dir)
        areas, critical = [], []
        for row in rows:
            if row["split"] != "calibration":
                continue
            with Image.open(root / row["image_path"]) as image:
                from .pipeline import prepare_rgb

                rgb_full = prepare_rgb(image)
            prediction = model.predict(rgb_full)
            facts = decision_facts(rgb_full, prediction, {"critical_roi": roi})
            areas.append(facts["anomaly_area_ratio"])
            critical.append(facts["critical_area_ratio"])
        areas_arr, critical_arr = np.array(areas), np.array(critical)
        # Percentile-based limits target a normal-review budget. Using the
        # calibration maximum would let a single noisy normal image push the
        # automatic-pass area limit above a real defect (measured on bottle).
        pass_area = round(float(_percentile(areas_arr, area_percentile)) * area_multiplier, 4)
        pass_critical = round(float(_percentile(critical_arr, area_percentile)) * area_multiplier, 4)
        reject_area = float(policy.get("reject_area", 0.06))
        if not 0 <= pass_area <= 1 or not 0 <= pass_critical <= 1:
            raise ValueError("校准得到的通过面积门槛超出0到1；请降低 area_percentile 或 area_multiplier")
        # Calibration may only tighten the teaching policy: a noisy normal set
        # must never widen automatic release beyond configs/quality_rules.json.
        # The same ceiling is enforced again in resolve_policy, so an older
        # calibration file cannot relax the decision rules either.
        cap_area = _pass_area_cap(policy, "pass_area_ratio")
        cap_critical = _pass_area_cap(policy, "pass_critical_area_ratio")
        pass_area = pass_area if cap_area is None else min(pass_area, cap_area)
        pass_critical = pass_critical if cap_critical is None else min(pass_critical, cap_critical)
        if pass_area > reject_area or pass_critical > reject_area:
            raise ValueError(
                f"校准得到 pass_area_ratio={pass_area}, pass_critical_area_ratio={pass_critical}，"
                f"大于 reject_area={reject_area}；正常校准图噪声过大或策略上限设置不当，"
                "请检查数据或调整 area_percentile/area_multiplier")
        artifact["models"][model_dir.name] = {
            "model_artifact_sha256": card.get("artifact_sha256"),
            "model_card_sha256": hashlib.sha256((model_dir / "model_card.json").read_bytes()).hexdigest(),
            "calibration_count": len(areas_arr),
            "anomaly_area_ratio_max": float(areas_arr.max()),
            "anomaly_area_ratio_percentile": _percentile(areas_arr, area_percentile),
            "critical_area_ratio_max": float(critical_arr.max()),
            "critical_area_ratio_percentile": _percentile(critical_arr, area_percentile),
            "pass_area_ratio": pass_area,
            "pass_critical_area_ratio": pass_critical,
            "area_percentile": area_percentile,
            "area_multiplier": area_multiplier,
        }
    output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return artifact


def load_calibration(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("schema_version") != 1:
        raise ValueError(f"不支持的校准文件格式：{path}")
    gate = value.get("input_gate")
    if not isinstance(gate, dict):
        raise ValueError(f"校准文件缺少 input_gate 对象：{path}")  # noqa: TRY004 - callers handle ValueError
    gate_numbers = {}
    for key in ("alignment_ncc_threshold", "illumination_gain_min_threshold",
                "illumination_gain_max_threshold", "long_edge_median"):
        try:
            number = float(gate[key])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"校准文件缺少有效的 {key}：{path}") from exc
        if not math.isfinite(number):
            raise ValueError(f"校准文件的 {key} 不是有限数值：{path}")
        gate_numbers[key] = number
    if not -1 <= gate_numbers["alignment_ncc_threshold"] <= 1:
        raise ValueError(f"校准的对齐阈值必须在 [-1, 1] 之间：{path}")
    if not 0 < gate_numbers["illumination_gain_min_threshold"] <= gate_numbers["illumination_gain_max_threshold"]:
        raise ValueError(f"校准的光照阈值必须满足 0 < min <= max：{path}")
    if gate_numbers["long_edge_median"] <= 0:
        raise ValueError(f"校准的 long_edge_median 必须为正：{path}")
    reference = value.get("reference") or {}
    reference_path = Path(path).parent / str(reference.get("path", ""))
    if not reference_path.is_file():
        raise FileNotFoundError(f"校准参考图缺失：{reference_path}")
    actual = hashlib.sha256(reference_path.read_bytes()).hexdigest()
    if reference.get("sha256") and actual != reference["sha256"]:
        raise ValueError("校准参考图的校验和与校准文件不一致。")
    with np.load(reference_path, allow_pickle=False) as arrays:
        reference_array = np.asarray(arrays[str(reference.get("array", "reference_rgb"))], dtype=np.float32)
    if reference_array.ndim != 3 or reference_array.shape[2] != 3 or reference_array.size == 0:
        raise ValueError(f"校准参考图必须是非空的 HxWx3 RGB 数组：{reference_path}")
    if not np.isfinite(reference_array).all():
        raise ValueError(f"校准参考图包含非有限数值：{reference_path}")
    value["_reference"] = reference_array
    value["_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return value


def resolution_bounds(policy: dict, config_path: Path, category: str | None) -> tuple[int, int] | None:
    """Long-edge range (pixels) the input gate really accepts; None when it cannot be derived.

    The UI uses it instead of a hard-coded pixel cap so the hint always matches the
    calibration file and the policy tolerance; images outside the range are judged as
    undetermined by the gate.
    """
    gate = policy.get("input_gate") or {}
    if not gate.get("enabled"):
        return None
    calibration_rel = _calibration_entry(policy, category)
    if not calibration_rel:
        return None
    artifact_path = Path(config_path).parent / str(calibration_rel)
    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        center = float((artifact.get("input_gate") or {})["long_edge_median"])
        tolerance = float(gate.get("resolution_tolerance", 0.25))
    except (OSError, KeyError, TypeError, ValueError):
        # A missing or stale calibration file only affects the hint text, not the later gate errors.
        return None
    if not math.isfinite(center) or center <= 0 or not 0 <= tolerance < 1:
        return None
    return math.floor(center * (1 - tolerance) + 1e-9), math.ceil(center * (1 + tolerance) - 1e-9)


def _calibration_entry(policy: dict, category: str | None) -> str | None:
    spec = policy.get("calibration")
    if spec is None:
        return None
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict):
        for key in (category, "default"):
            if key and spec.get(key):
                return str(spec[key])
    return None


def resolve_policy(policy: dict, config_path: Path, model_id: str, category: str | None = None,
                   calibration: dict | None = None, model_card: dict | None = None,
                   model_card_sha256: str | None = None) -> tuple[dict, dict | None]:
    """Overlay calibrated thresholds on the teaching policy for one model.

    Explicit policy values still win when calibration is disabled, so reports
    remain reproducible from the stored policy hash. A calibration artifact may
    only tighten the policy's pass-area ceilings, never loosen them.
    """
    gate = policy.get("input_gate") or {}
    if not gate.get("enabled"):
        return policy, calibration
    if calibration is None:
        calibration_rel = _calibration_entry(policy, category)
        if not calibration_rel:
            raise ValueError(
                f"策略启用了输入门控，但类别 {category or 'unknown'} 没有配置校准文件；"
                "请先运行 calibrate-policy")
        calibration = load_calibration(Path(config_path).parent / str(calibration_rel))
    calibration_category = calibration.get("category")
    if calibration_category and category and str(calibration_category) != category:
        raise ValueError(
            f"校准文件类别为 {calibration_category}，与当前模型类别 {category} 不一致；请使用对应类别的校准文件")
    expected_roi = calibration.get("critical_roi")
    if expected_roi is None:
        raise ValueError("校准文件缺少 critical_roi，无法确认面积门槛是否仍适用；请重新运行 calibrate-policy")
    current_roi = [float(value) for value in policy.get("critical_roi", [])]
    if len(current_roi) != 4 or any(abs(a - float(b)) > 1e-9 for a, b in zip(current_roi, expected_roi)):
        raise ValueError("策略 critical_roi 与校准时使用的关键区不一致；请重新运行 calibrate-policy")
    # The calibration records the policy it was derived from; comparing the
    # digest stops an edited quality_rules.json from silently reusing stale,
    # possibly looser, pass-area limits.
    recorded_rules_sha256 = calibration.get("rules_sha256")
    if not recorded_rules_sha256:
        raise ValueError("校准文件未记录 rules_sha256，无法确认面积门槛对应当前策略；请重新运行 calibrate-policy")
    current_rules_sha256 = hashlib.sha256(Path(config_path).read_bytes()).hexdigest()
    if recorded_rules_sha256 != current_rules_sha256:
        raise ValueError(
            "quality_rules.json 在校准后被修改（rules_sha256 不一致），旧校准可能放宽通过面积门槛；"
            "请确认规则后重新运行 calibrate-policy")
    gate_values = calibration["input_gate"]
    effective = dict(policy)
    effective.update({
        "alignment_min": gate_values["alignment_ncc_threshold"],
        "illumination_min": gate_values["illumination_gain_min_threshold"],
        "illumination_max": gate_values["illumination_gain_max_threshold"],
        "max_aspect_deviation": float(gate.get("max_aspect_deviation", 0.15)),
        "expected_long_edge": float(gate_values["long_edge_median"]),
        "resolution_tolerance": float(gate.get("resolution_tolerance", 0.25)),
        "calibrated": True,
    })
    model_values = (calibration.get("models") or {}).get(model_id)
    if not isinstance(model_values, dict) or any(
            key not in model_values for key in ("pass_area_ratio", "pass_critical_area_ratio")):
        raise ValueError(
            f"校准文件未包含模型 {model_id} 的通过面积门槛；请用 --model-dir 指向该模型重新运行 calibrate-policy")
    if model_card is not None:
        expected = model_values.get("model_artifact_sha256")
        if not expected:
            raise ValueError(
                f"校准文件未记录模型 {model_id} 的指纹，无法确认面积门槛是否对应当前模型；请重新运行 calibrate-policy")
        if expected != model_card.get("artifact_sha256"):
            raise ValueError(
                f"校准文件中的面积门槛属于另一个 {model_id} 模型；重新训练后必须重新运行 calibrate-policy")
    if model_card_sha256 is not None:
        recorded_card_sha256 = model_values.get("model_card_sha256")
        if not recorded_card_sha256:
            raise ValueError(
                f"校准文件未记录模型 {model_id} 的模型卡指纹，无法确认阈值对应当前模型卡；"
                "请重新运行 calibrate-policy")
        if recorded_card_sha256 != model_card_sha256:
            raise ValueError(
                f"模型 {model_id} 的模型卡在校准后被修改，旧校准阈值可能不适用；请重新运行 calibrate-policy")
    for key in ("pass_area_ratio", "pass_critical_area_ratio"):
        calibrated = float(model_values[key])
        cap = _pass_area_cap(policy, key)
        # Calibration may only tighten the policy ceiling; a calibration file
        # recorded with a looser limit must not widen automatic release.
        effective[key] = calibrated if cap is None else min(calibrated, cap)
    return effective, calibration
