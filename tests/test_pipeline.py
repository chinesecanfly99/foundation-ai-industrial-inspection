import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from industrial_inspection.detectors import PixelReferenceDetector
from industrial_inspection.pipeline import inspect_image, prepare_rgb


def test_no_image_or_tiny_input_fails_before_loading_model(tmp_path):
    with pytest.raises(ValueError, match="上传"):
        inspect_image(None, tmp_path, tmp_path / "none")
    with pytest.raises(ValueError, match="32"):
        inspect_image(Image.new("RGB", (20, 20)), tmp_path, tmp_path / "none")


def test_policy_area_is_measured_on_the_model_grid(tmp_path):
    rng = np.random.default_rng(23)
    paths = []
    for index in range(10):
        rgb = np.clip(128 + rng.normal(0, 1.2, (48, 64, 3)), 0, 255).astype(np.uint8)
        path = tmp_path / f"normal_{index:02}.png"
        Image.fromarray(rgb).save(path)
        paths.append(path)
    detector = PixelReferenceDetector(image_size=32, seed=5).fit(paths[:8], paths[8:])
    model_dir = tmp_path / "model"
    detector.save(model_dir)
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({
        "version": "test", "critical_roi": [0.2, 0.2, 0.8, 0.8], "review_margin": 0.1,
        "min_anomaly_area": 0.005, "reject_area": 0.06, "critical_overlap": 0.2,
        "min_brightness": 15, "max_brightness": 245, "min_sharpness": 0,
    }), encoding="utf-8")
    image = Image.fromarray(np.full((64, 80, 3), 128, dtype=np.uint8))
    _, report = inspect_image(image, model_dir, rules)
    prediction = detector.predict(np.asarray(image))
    expected = float((prediction.model_heatmap > prediction.pixel_threshold).mean())
    assert report["model_heatmap_size"] == list(prediction.model_heatmap.shape)
    assert report["anomaly_area_ratio"] == pytest.approx(expected)
    assert 0 <= report["critical_area_ratio"] <= 1
    assert report["upload_sha256"] == hashlib.sha256(np.asarray(image).tobytes()).hexdigest()
    assert report["model_card_sha256"] == hashlib.sha256((model_dir / "model_card.json").read_bytes()).hexdigest()
    assert report["model_artifact_sha256"] == json.loads((model_dir / "model_card.json").read_text())["artifact_sha256"]
    assert report["image_size"] == [80, 64]
    assert report["prepared_size"] == [80, 64]
    assert report["effective_policy"]["version"] == "test"
    assert report["score_ratio"] == pytest.approx(report["score"] / report["image_threshold"])
    broken = json.loads(rules.read_text(encoding="utf-8"))
    del broken["critical_roi"]
    broken_path = tmp_path / "broken_rules.json"
    broken_path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(ValueError, match="critical_roi"):
        inspect_image(image, model_dir, broken_path)


def test_prepare_rgb_applies_exif_and_working_resolution_limit():
    image = Image.new("RGB", (2400, 1200), (10, 20, 30))
    prepared = prepare_rgb(image)
    assert prepared.shape == (512, 1024, 3)


def test_float_and_uint8_arrays_are_processed_equally(tmp_path):
    rng = np.random.default_rng(31)
    paths = []
    for index in range(10):
        rgb = np.clip(128 + rng.normal(0, 1.2, (48, 64, 3)), 0, 255).astype(np.uint8)
        path = tmp_path / f"normal_{index:02}.png"
        Image.fromarray(rgb).save(path)
        paths.append(path)
    detector = PixelReferenceDetector(image_size=32, seed=7).fit(paths[:8], paths[8:])
    model_dir = tmp_path / "model"
    detector.save(model_dir)
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({
        "version": "test", "critical_roi": [0.2, 0.2, 0.8, 0.8], "review_margin": 0.1,
        "min_anomaly_area": 0.005, "pass_area_ratio": 0.015, "pass_critical_area_ratio": 0.005,
        "reject_area": 0.06, "critical_overlap": 0.2,
        "min_brightness": 15, "max_brightness": 245, "min_sharpness": 0,
    }), encoding="utf-8")
    rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
    _, from_uint8 = inspect_image(rgb, model_dir, rules)
    _, from_float = inspect_image(rgb.astype(np.float32) / 255, model_dir, rules)
    assert from_float["brightness"] == from_uint8["brightness"] == pytest.approx(128)
    assert from_float["score"] == pytest.approx(from_uint8["score"])


def test_replaced_artifact_bypasses_the_cached_detector(tmp_path):
    rng = np.random.default_rng(41)
    paths = []
    for index in range(10):
        rgb = np.clip(128 + rng.normal(0, 1.2, (48, 64, 3)), 0, 255).astype(np.uint8)
        path = tmp_path / f"normal_{index:02}.png"
        Image.fromarray(rgb).save(path)
        paths.append(path)
    detector = PixelReferenceDetector(image_size=32, seed=9).fit(paths[:8], paths[8:])
    model_dir = tmp_path / "model"
    detector.save(model_dir)
    rules = tmp_path / "rules.json"
    rules.write_text(json.dumps({
        "version": "test", "critical_roi": [0.2, 0.2, 0.8, 0.8], "review_margin": 0.1,
        "min_anomaly_area": 0.005, "pass_area_ratio": 0.015, "pass_critical_area_ratio": 0.005,
        "reject_area": 0.06, "critical_overlap": 0.2,
        "min_brightness": 15, "max_brightness": 245, "min_sharpness": 0,
    }), encoding="utf-8")
    image = Image.new("RGB", (64, 64), (128, 128, 128))
    inspect_image(image, model_dir, rules)
    artifact = model_dir / "model.npz"
    artifact.write_bytes(artifact.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="checksum"):
        inspect_image(image, model_dir, rules)


def test_gate_uses_source_resolution_not_the_working_copy():
    from industrial_inspection.detectors import AnomalyPrediction
    from industrial_inspection.pipeline import decision_facts
    from industrial_inspection.rules import decide

    rows = np.linspace(0, 255, 64, dtype=np.uint8)[:, None, None]
    rgb = np.repeat(np.repeat(rows, 64, axis=1), 3, axis=2)
    reference = np.asarray(
        Image.fromarray(rgb).resize((32, 32), Image.Resampling.BILINEAR), dtype=np.float32
    ) / 255
    prediction = AnomalyPrediction(
        heatmap=np.zeros((32, 32), dtype=np.float32),
        model_heatmap=np.zeros((32, 32), dtype=np.float32),
        score=0.01, image_threshold=1.0, pixel_threshold=0.5, model_id="test", metadata={},
    )
    policy = {
        "version": "test", "critical_roi": [0.2, 0.2, 0.8, 0.8], "review_margin": 0.1,
        "min_anomaly_area": 0.005, "pass_area_ratio": 0.015, "pass_critical_area_ratio": 0.005,
        "reject_area": 0.06, "critical_overlap": 0.2, "min_brightness": 0, "max_brightness": 255,
        "min_sharpness": 0, "input_gate": {"enabled": True, "max_aspect_deviation": 0.15,
                                           "resolution_tolerance": 0.25},
        "alignment_min": 0.90, "illumination_min": 0.9, "illumination_max": 1.1,
        "expected_long_edge": 64.0,
    }
    calibration = {"_reference": reference}
    native = decision_facts(rgb, prediction, policy, calibration, source_size=(64, 64))
    enlarged = decision_facts(rgb, prediction, policy, calibration, source_size=(256, 256))
    assert native["resolution_ratio"] == pytest.approx(1.0)
    assert enlarged["resolution_ratio"] == pytest.approx(4.0)
    from_float = decision_facts(rgb.astype(np.float32) / 255, prediction, policy, calibration,
                                source_size=(64, 64))
    assert from_float["brightness"] == pytest.approx(native["brightness"])
    assert decide(native, policy)["decision"] != "undetermined"
    assert decide(enlarged, policy)["decision"] == "undetermined"
