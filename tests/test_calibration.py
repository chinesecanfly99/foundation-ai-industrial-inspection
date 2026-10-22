import hashlib
from pathlib import Path

import pytest

from industrial_inspection.calibration import calibrate, resolve_policy
from industrial_inspection.cli import fit_model
from industrial_inspection.data import build_manifest, generate_synthetic

RULES = Path(__file__).resolve().parents[1] / "configs" / "quality_rules.json"


def _rules_sha256() -> str:
    return hashlib.sha256(RULES.read_bytes()).hexdigest()


def _policy(**overrides):
    policy = {
        "version": "test",
        "critical_roi": [0.2, 0.2, 0.8, 0.8],
        "input_gate": {"enabled": True, "max_aspect_deviation": 0.15, "resolution_tolerance": 0.25},
    }
    policy.update(overrides)
    return policy


def _calibration(**overrides):
    calibration = {
        "category": "bottle",
        "critical_roi": [0.2, 0.2, 0.8, 0.8],
        "rules_sha256": _rules_sha256(),
        "input_gate": {
            "alignment_ncc_threshold": 0.96,
            "illumination_gain_min_threshold": 0.95,
            "illumination_gain_max_threshold": 1.05,
            "long_edge_median": 900.0,
        },
        "models": {
            "bottle_pixel": {"pass_area_ratio": 0.011, "pass_critical_area_ratio": 0.008},
        },
    }
    calibration.update(overrides)
    return calibration


def test_resolve_policy_applies_gate_and_model_area_thresholds():
    effective, applied = resolve_policy(
        _policy(), RULES, "bottle_pixel", "bottle", calibration=_calibration())
    assert applied["category"] == "bottle"
    assert effective["alignment_min"] == pytest.approx(0.96)
    assert effective["expected_long_edge"] == pytest.approx(900.0)
    assert effective["pass_area_ratio"] == pytest.approx(0.011)
    assert effective["pass_critical_area_ratio"] == pytest.approx(0.008)
    assert effective["calibrated"] is True


def test_resolve_policy_only_tightens_the_policy_pass_ceiling():
    strict = _policy(min_anomaly_area=0.005, pass_area_ratio=0.005, pass_critical_area_ratio=0.005)
    effective, _ = resolve_policy(strict, RULES, "bottle_pixel", "bottle", calibration=_calibration())
    assert effective["pass_area_ratio"] == pytest.approx(0.005)
    assert effective["pass_critical_area_ratio"] == pytest.approx(0.005)
    # A larger policy ceiling must not widen a stricter calibrated limit.
    loose = _policy(min_anomaly_area=0.005, pass_area_ratio=0.02, pass_critical_area_ratio=0.02)
    effective, _ = resolve_policy(loose, RULES, "bottle_pixel", "bottle", calibration=_calibration())
    assert effective["pass_area_ratio"] == pytest.approx(0.011)
    assert effective["pass_critical_area_ratio"] == pytest.approx(0.008)


@pytest.mark.parametrize("kwargs,match", [
    ({"category": "synthetic"}, "类别"),
    ({"critical_roi": None}, "critical_roi"),
    ({"critical_roi": [0.1, 0.1, 0.9, 0.9]}, "critical_roi"),
    ({"models": {}}, "未包含模型"),
    ({"models": {"bottle_pixel": {"pass_area_ratio": 0.011}}}, "未包含模型"),
])
def test_resolve_policy_refuses_stale_or_incomplete_calibration(kwargs, match):
    with pytest.raises(ValueError, match=match):
        resolve_policy(_policy(), RULES, "bottle_pixel", "bottle",
                       calibration=_calibration(**kwargs))


def test_calibrate_rejects_a_model_trained_on_another_manifest(tmp_path):
    data_root = generate_synthetic(tmp_path / "data", seed=11)
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    build_manifest(data_root, "synthetic", first, seed=11)
    build_manifest(data_root, "synthetic", second, seed=12)
    model_dir = tmp_path / "model"
    fit_model(first, "pixel", model_dir, image_size=32, seed=11)
    with pytest.raises(ValueError, match="manifest"):
        calibrate(second, model_dir, tmp_path / "calibration.json", image_size=32, rules=RULES)


@pytest.mark.parametrize("kwargs,match", [
    ({"area_percentile": 101}, "area_percentile"),
    ({"area_percentile": -1}, "area_percentile"),
    ({"area_multiplier": 0}, "area_multiplier"),
    ({"alignment_margin": -0.1}, "alignment_margin"),
    ({"illumination_margin": float("nan")}, "illumination_margin"),
])
def test_calibrate_rejects_invalid_parameters(tmp_path, kwargs, match):
    with pytest.raises(ValueError, match=match):
        calibrate(tmp_path / "missing.csv", None, tmp_path / "calibration.json", **kwargs)


def test_calibration_uses_the_manifest_category_and_model_image_size(tmp_path):
    data_root = generate_synthetic(tmp_path / "data", seed=13)
    manifest = tmp_path / "manifests" / "custom-name.csv"
    build_manifest(data_root, "synthetic", manifest, seed=13)
    model_dir = tmp_path / "model"
    fit_model(manifest, "pixel", model_dir, image_size=32, seed=13)
    output = tmp_path / "calibration" / "synthetic.json"
    artifact = calibrate(manifest, model_dir, output, rules=RULES)
    assert artifact["category"] == "synthetic"
    assert artifact["image_size"] == 32
    assert artifact["critical_roi"] == [0.2, 0.2, 0.8, 0.8]
    assert (output.parent / artifact["reference"]["path"]).name == "reference_synthetic_32.npz"
    assert "model" in artifact["models"]
    values = artifact["models"]["model"]
    # The generated artifact records raw normal-image percentiles but writes
    # pass limits that never loosen configs/quality_rules.json.
    assert values["pass_area_ratio"] <= 0.005
    assert values["pass_critical_area_ratio"] <= 0.005


def test_calibrate_caps_pass_limits_at_the_policy_ceiling(tmp_path):
    data_root = generate_synthetic(tmp_path / "data", seed=17)
    manifest = tmp_path / "synthetic.csv"
    build_manifest(data_root, "synthetic", manifest, seed=17)
    model_dir = tmp_path / "model"
    fit_model(manifest, "pixel", model_dir, image_size=32, seed=17)
    artifact = calibrate(manifest, model_dir, tmp_path / "calibration.json",
                         area_multiplier=20.0, rules=RULES)
    values = artifact["models"]["model"]
    # Both raw percentiles now exceed the policy ceiling, so the cap must bind.
    assert values["anomaly_area_ratio_percentile"] * 20.0 > 0.005
    assert values["critical_area_ratio_percentile"] * 20.0 > 0.005
    assert values["pass_area_ratio"] == pytest.approx(0.005)
    assert values["pass_critical_area_ratio"] == pytest.approx(0.005)


def test_load_calibration_rejects_incomplete_or_bad_references(tmp_path):
    import hashlib

    import numpy as np

    from industrial_inspection.calibration import load_calibration

    reference = tmp_path / "reference.npz"
    np.savez(reference, reference_rgb=np.zeros((8, 8, 3), dtype=np.float32))
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    artifact = {
        "schema_version": 1, "category": "bottle", "critical_roi": [0.2, 0.2, 0.8, 0.8],
        "reference": {"path": "reference.npz", "array": "reference_rgb", "sha256": digest},
        "input_gate": {"alignment_ncc_threshold": 0.9,
                       "illumination_gain_min_threshold": 0.9,
                       "illumination_gain_max_threshold": 1.1},
        "models": {},
    }
    path = tmp_path / "calibration.json"
    path.write_text(__import__("json").dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match="long_edge_median"):
        load_calibration(path)


def test_resolve_policy_refuses_a_calibration_from_another_model():
    calibration = _calibration()
    calibration["models"]["bottle_pixel"]["model_artifact_sha256"] = "a" * 64
    with pytest.raises(ValueError, match="另一个"):
        resolve_policy(_policy(), RULES, "bottle_pixel", "bottle", calibration=calibration,
                       model_card={"artifact_sha256": "b" * 64})
    calibration["models"]["bottle_pixel"].pop("model_artifact_sha256")
    with pytest.raises(ValueError, match="指纹"):
        resolve_policy(_policy(), RULES, "bottle_pixel", "bottle", calibration=calibration,
                       model_card={"artifact_sha256": "b" * 64})


def test_resolve_policy_refuses_a_calibration_from_edited_rules():
    with pytest.raises(ValueError, match="rules_sha256"):
        resolve_policy(_policy(), RULES, "bottle_pixel", "bottle",
                       calibration=_calibration(rules_sha256="0" * 64))
    stale = _calibration()
    stale.pop("rules_sha256")
    with pytest.raises(ValueError, match="rules_sha256"):
        resolve_policy(_policy(), RULES, "bottle_pixel", "bottle", calibration=stale)


def test_resolve_policy_refuses_a_model_card_changed_after_calibration():
    calibration = _calibration()
    model = calibration["models"]["bottle_pixel"]
    model["model_artifact_sha256"] = "a" * 64
    model["model_card_sha256"] = "c" * 64
    with pytest.raises(ValueError, match="模型卡"):
        resolve_policy(_policy(), RULES, "bottle_pixel", "bottle", calibration=calibration,
                       model_card={"artifact_sha256": "a" * 64}, model_card_sha256="d" * 64)
    model.pop("model_card_sha256")
    with pytest.raises(ValueError, match="模型卡"):
        resolve_policy(_policy(), RULES, "bottle_pixel", "bottle", calibration=calibration,
                       model_card={"artifact_sha256": "a" * 64}, model_card_sha256="d" * 64)


def test_calibrate_refuses_an_area_limit_that_would_auto_pass_everything(tmp_path):
    data_root = generate_synthetic(tmp_path / "data", seed=15)
    manifest = tmp_path / "synthetic.csv"
    build_manifest(data_root, "synthetic", manifest, seed=15)
    model_dir = tmp_path / "model"
    fit_model(manifest, "pixel", model_dir, image_size=32, seed=15)
    with pytest.raises(ValueError, match="0到1|reject_area"):
        calibrate(manifest, model_dir, tmp_path / "calibration.json",
                  area_multiplier=1000.0, rules=RULES)


@pytest.mark.parametrize("override,match", [
    ({"alignment_ncc_threshold": 1.5}, "对齐阈值"),
    ({"illumination_gain_min_threshold": 1.2, "illumination_gain_max_threshold": 1.1}, "光照阈值"),
    ({"long_edge_median": 0}, "long_edge_median"),
])
def test_load_calibration_rejects_out_of_range_gate_numbers(tmp_path, override, match):
    import hashlib

    import numpy as np

    from industrial_inspection.calibration import load_calibration

    reference = tmp_path / "reference.npz"
    np.savez(reference, reference_rgb=np.zeros((8, 8, 3), dtype=np.float32))
    artifact = {
        "schema_version": 1, "category": "bottle", "critical_roi": [0.2, 0.2, 0.8, 0.8],
        "reference": {"path": "reference.npz", "array": "reference_rgb",
                       "sha256": hashlib.sha256(reference.read_bytes()).hexdigest()},
        "input_gate": {"alignment_ncc_threshold": 0.9,
                       "illumination_gain_min_threshold": 0.9,
                       "illumination_gain_max_threshold": 1.1,
                       "long_edge_median": 900.0} | override,
        "models": {},
    }
    path = tmp_path / "calibration.json"
    path.write_text(__import__("json").dumps(artifact), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_calibration(path)


def test_resolution_bounds_follow_calibration_median_and_tolerance(tmp_path):
    import json

    from industrial_inspection.calibration import resolution_bounds

    (tmp_path / "calibration").mkdir()
    (tmp_path / "calibration" / "bottle.json").write_text(
        json.dumps({"input_gate": {"long_edge_median": 900.0}}), encoding="utf-8")
    config = tmp_path / "quality_rules.json"
    config.write_text(json.dumps({
        "input_gate": {"enabled": True, "resolution_tolerance": 0.25},
        "calibration": {"bottle": "calibration/bottle.json"},
    }), encoding="utf-8")
    policy = json.loads(config.read_text(encoding="utf-8"))
    assert resolution_bounds(policy, config, "bottle") == (675, 1125)
    assert resolution_bounds(policy, config, "unknown") is None


def test_resolution_bounds_are_absent_when_the_gate_or_calibration_is_missing(tmp_path):
    import json

    from industrial_inspection.calibration import resolution_bounds

    config = tmp_path / "quality_rules.json"
    config.write_text(json.dumps({
        "input_gate": {"enabled": False, "resolution_tolerance": 0.25},
        "calibration": {"bottle": "calibration/bottle.json"},
    }), encoding="utf-8")
    policy = json.loads(config.read_text(encoding="utf-8"))
    assert resolution_bounds(policy, config, "bottle") is None
    policy["input_gate"]["enabled"] = True
    # A missing calibration file only removes the range hint; it must not raise and keep the UI from opening.
    assert resolution_bounds(policy, config, "bottle") is None
