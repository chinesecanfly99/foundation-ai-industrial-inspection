import json

import numpy as np
from PIL import Image
import pytest

from industrial_inspection.detectors import PatchMemoryDetector, PixelReferenceDetector, load_detector


@pytest.fixture
def normal_dataset(tmp_path):
    rng = np.random.default_rng(17)
    paths = []
    for index in range(12):
        # Distinct images from one normal distribution, no anomaly labels used.
        rgb = np.clip(128 + rng.normal(0, 1.5, (48, 64, 3)), 0, 255).astype(np.uint8)
        path = tmp_path / f"normal_{index:02}.png"
        Image.fromarray(rgb).save(path)
        paths.append(path)
    return paths[:8], paths[8:]


def test_pixel_detector_localizes_unseen_defect_and_preserves_shape(normal_dataset):
    train, calibration = normal_dataset
    detector = PixelReferenceDetector(image_size=32).fit(train, calibration)
    normal = np.full((48, 64, 3), 128, dtype=np.uint8)
    defective = normal.copy()
    defective[16:32, 20:44] = [250, 20, 20]
    good, bad = detector.predict(normal), detector.predict(defective)
    assert good.score < good.image_threshold
    assert bad.score > bad.image_threshold
    assert bad.heatmap.shape == normal.shape[:2]
    assert np.isfinite(bad.heatmap).all()
    assert bad.heatmap[20:28, 24:40].mean() > 10 * bad.heatmap[:8, :8].mean()
    assert bad.metadata["threshold_source"] == "held_out_normal_calibration_only"


def test_normal_calibration_scores_stay_below_image_threshold(normal_dataset):
    train, calibration = normal_dataset
    detector = PixelReferenceDetector(image_size=32).fit(train, calibration)
    for path in calibration:
        prediction = detector.predict(np.asarray(Image.open(path).convert("RGB")))
        assert prediction.score <= prediction.image_threshold


def test_numeric_artifact_roundtrip_and_card(normal_dataset, tmp_path):
    train, calibration = normal_dataset
    original = PixelReferenceDetector(image_size=32).fit(train, calibration)
    model_dir = tmp_path / "model"
    original.save(model_dir)
    reloaded = load_detector(model_dir)
    rgb = np.asarray(Image.open(calibration[0]).convert("RGB"))
    before, after = original.predict(rgb), reloaded.predict(rgb)
    np.testing.assert_array_equal(before.heatmap, after.heatmap)
    assert before.score == after.score
    assert before.image_threshold == after.image_threshold
    assert before.pixel_threshold == after.pixel_threshold
    card = json.loads((model_dir / "model_card.json").read_text(encoding="utf-8"))
    assert card["calibration_count"] == len(calibration)
    assert card["training_count"] == len(train)
    assert card["calibration"]["source"] == "held_out_normal_images_only"
    with np.load(model_dir / "model.npz", allow_pickle=False) as arrays:
        assert set(arrays.files) == {"mean", "std"}
        assert all(arrays[name].dtype.kind == "f" for name in arrays.files)


def test_artifact_corruption_is_detected(normal_dataset, tmp_path):
    train, calibration = normal_dataset
    model_dir = tmp_path / "corrupt_model"
    PixelReferenceDetector(image_size=32).fit(train, calibration).save(model_dir)
    with (model_dir / "model.npz").open("ab") as artifact:
        artifact.write(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        load_detector(model_dir)


@pytest.mark.parametrize("rgb", [np.zeros((32, 32)), np.zeros((32, 32, 4)), np.zeros((1, 32, 3)), np.full((32, 32, 3), np.nan), np.full((32, 32, 3), -1), np.full((32, 32, 3), 256)])
def test_invalid_images_are_rejected(normal_dataset, rgb):
    detector = PixelReferenceDetector(image_size=32).fit(*normal_dataset)
    with pytest.raises(ValueError):
        detector.predict(rgb)


def test_float_rgb_and_uint8_inputs_agree(normal_dataset):
    detector = PixelReferenceDetector(image_size=32).fit(*normal_dataset)
    rgb = np.full((48, 64, 3), 128, dtype=np.uint8)
    np.testing.assert_array_equal(detector.predict(rgb).heatmap, detector.predict(rgb.astype(np.float32) / 255).heatmap)


def test_fit_requires_disjoint_nonempty_calibration(normal_dataset):
    train, calibration = normal_dataset
    with pytest.raises(ValueError, match="training"):
        PixelReferenceDetector().fit([], calibration)
    with pytest.raises(ValueError, match="calibration"):
        PixelReferenceDetector().fit(train, [])
    with pytest.raises(ValueError, match="disjoint"):
        PixelReferenceDetector().fit(train, train[:2])
    with pytest.raises(FileNotFoundError):
        PixelReferenceDetector().fit([train[0].parent / "missing.png"], calibration)


def test_unfitted_and_invalid_configuration_are_actionable(tmp_path):
    detector = PixelReferenceDetector()
    with pytest.raises(RuntimeError, match="not fitted"):
        detector.predict(np.zeros((32, 32, 3), dtype=np.uint8))
    with pytest.raises(RuntimeError, match="not fitted"):
        detector.save(tmp_path)
    with pytest.raises(ValueError, match="image_size"):
        PixelReferenceDetector(image_size=8)
    with pytest.raises(ValueError, match="memory_size"):
        PatchMemoryDetector(memory_size=0)
    with pytest.raises(ValueError, match="device"):
        PatchMemoryDetector(device="gpu0")
