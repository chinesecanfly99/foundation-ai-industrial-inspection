"""Product routing must follow model provenance and never reuse another policy."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from industrial_inspection.catalog import PRODUCTS, REAL_PRODUCTS, policy_path_for_product, product_id_from_card
from industrial_inspection import pipeline


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("product_id", [*REAL_PRODUCTS, "synthetic"])
def test_registered_categories_have_separate_models_and_policies(product_id):
    product = PRODUCTS[product_id]
    assert product["default_model"] in product["model_ids"]
    assert all(model.startswith(product_id + "_") for model in product["model_ids"])
    path = policy_path_for_product(PROJECT_ROOT, product_id)
    assert path.is_file()
    policy = json.loads(path.read_text(encoding="utf-8"))
    assert policy.get("product_category", "bottle") == product_id
    assert policy["review_margin"] == 0.1
    assert policy["reject_area"] == 0.06


@pytest.mark.parametrize("product_id", REAL_PRODUCTS)
def test_card_source_identifies_real_product(product_id):
    assert product_id_from_card({"dataset_source": f"mvtec_ad/{product_id}"}) == product_id


def test_synthetic_source_remains_separate_from_real_products():
    assert "synthetic" not in REAL_PRODUCTS
    assert product_id_from_card({"dataset_source": "synthetic"}) == "synthetic"


@pytest.mark.parametrize("source", [None, "unknown", "local/tile", "mvtec_ad/unknown", "mvtec_ad/tile/../../custom", {"product": "tile"}])
def test_unregistered_sources_are_not_guessed_or_used_as_paths(source):
    assert product_id_from_card({"dataset_source": source, "model_id": "tile_patch"}) is None


def test_unknown_product_cannot_supply_policy_path():
    with pytest.raises(ValueError, match="未知产品类别"):
        policy_path_for_product(PROJECT_ROOT, "../../outside")


def model_fixture(tmp_path, monkeypatch, source):
    model_dir = tmp_path / "a_custom_directory_name"
    model_dir.mkdir()
    (model_dir / "model_card.json").write_text(json.dumps({"dataset_source": source}), encoding="utf-8")
    # A sharp, mid-brightness input passes the independent quality gate.
    image = np.indices((64, 64)).sum(axis=0) % 2 * 160 + 40
    image = Image.fromarray(np.repeat(image[:, :, None], 3, axis=2).astype(np.uint8))
    dummy = SimpleNamespace(predict=lambda rgb: SimpleNamespace(
        heatmap=np.zeros(rgb.shape[:2], dtype=np.float32), score=0.1,
        image_threshold=1.0, pixel_threshold=0.5,
    ))
    monkeypatch.setattr(pipeline, "_load_model", lambda *args: dummy)
    return image, model_dir


@pytest.mark.parametrize("product_id", [*REAL_PRODUCTS, "synthetic"])
def test_inference_automatically_routes_policy_and_exports_portable_identity(tmp_path, monkeypatch, product_id):
    source = "synthetic" if product_id == "synthetic" else f"mvtec_ad/{product_id}"
    image, model_dir = model_fixture(tmp_path, monkeypatch, source)
    overlay, report = pipeline.inspect_image(image, model_dir)
    assert overlay.size == image.size
    assert report["product_category"] == product_id
    assert report["product_label"] == PRODUCTS[product_id]["label"]
    assert report["policy_name"] == Path(PRODUCTS[product_id]["policy_file"]).name
    assert "/" not in report["policy_name"] and "\\" not in report["policy_name"]
    assert report["decision"] == "pass"


def test_unknown_model_requires_explicit_rules_before_loading_weights(tmp_path, monkeypatch):
    image, model_dir = model_fixture(tmp_path, monkeypatch, "local/my_product")
    monkeypatch.setattr(pipeline, "_load_model", lambda *args: pytest.fail("Should validate category first"))
    with pytest.raises(ValueError, match="显式指定 --rules"):
        pipeline.inspect_image(image, model_dir)


def test_explicit_policy_for_another_product_is_rejected(tmp_path, monkeypatch):
    image, model_dir = model_fixture(tmp_path, monkeypatch, "mvtec_ad/metal_nut")
    monkeypatch.setattr(pipeline, "_load_model", lambda *args: pytest.fail("Should reject mismatched policy first"))
    with pytest.raises(ValueError, match="不一致"):
        pipeline.inspect_image(image, model_dir, policy_path_for_product(PROJECT_ROOT, "tile"))


def test_builtin_legacy_bottle_policy_is_not_a_custom_policy_for_another_product(tmp_path, monkeypatch):
    image, model_dir = model_fixture(tmp_path, monkeypatch, "mvtec_ad/metal_nut")
    monkeypatch.setattr(pipeline, "_load_model", lambda *args: pytest.fail("Must reject builtin category before loading weights"))
    with pytest.raises(ValueError, match="内置质量策略类别 bottle.*不一致"):
        pipeline.inspect_image(image, model_dir, PROJECT_ROOT / "configs" / ".." / "configs" / "quality_rules.json")


def test_bottle_can_explicitly_use_its_legacy_builtin_policy(tmp_path, monkeypatch):
    image, model_dir = model_fixture(tmp_path, monkeypatch, "mvtec_ad/bottle")
    _, report = pipeline.inspect_image(image, model_dir, PROJECT_ROOT / "configs" / "quality_rules.json")
    assert report["product_category"] == "bottle"
    assert report["policy_name"] == "quality_rules.json"
    assert report["decision"] == "pass"


def test_category_policy_cannot_validate_unknown_model(tmp_path, monkeypatch):
    image, model_dir = model_fixture(tmp_path, monkeypatch, "local/unknown")
    with pytest.raises(ValueError, match="不一致"):
        pipeline.inspect_image(image, model_dir, policy_path_for_product(PROJECT_ROOT, "tile"))


@pytest.mark.parametrize("source", ["mvtec_ad/metal_nut", "local/my_product"])
def test_explicit_legacy_custom_policy_without_category_remains_supported(tmp_path, monkeypatch, source):
    image, model_dir = model_fixture(tmp_path, monkeypatch, source)
    policy = json.loads((PROJECT_ROOT / "configs/quality_rules.json").read_text(encoding="utf-8"))
    custom_path = tmp_path / "custom-rules.json"
    custom_path.write_text(json.dumps(policy), encoding="utf-8")
    _, report = pipeline.inspect_image(image, model_dir, custom_path)
    assert report["policy_name"] == "custom-rules.json"
    assert report["product_category"] == ("metal_nut" if source.startswith("mvtec") else "unknown")


def test_tile_uses_the_entire_texture_surface_as_demo_roi():
    policy = json.loads(policy_path_for_product(PROJECT_ROOT, "tile").read_text(encoding="utf-8"))
    assert policy["critical_roi"] == [0, 0, 1, 1]


def test_cli_predict_defaults_to_automatic_policy(tmp_path, monkeypatch, capsys):
    from industrial_inspection import cli

    source_path = tmp_path / "input.png"
    Image.new("RGB", (64, 64), "gray").save(source_path)
    output = tmp_path / "out"
    calls = []

    def fake_inspect(image, model_dir, config_path):
        calls.append(config_path)
        return image, {"product_category": "tile", "policy_name": "quality_rules_tile.json"}

    monkeypatch.setattr(pipeline, "inspect_image", fake_inspect)
    monkeypatch.setattr("sys.argv", ["industrial_inspection", "predict", "--image", str(source_path),
                                     "--model-dir", str(tmp_path / "tile_patch"), "--output", str(output)])
    cli.main()
    assert calls == [None]
    assert json.loads((output / "report.json").read_text(encoding="utf-8"))["product_category"] == "tile"


def test_cli_invalid_policy_shows_a_concise_actionable_error(tmp_path, monkeypatch, capsys):
    from industrial_inspection import cli

    source_path = tmp_path / "input.png"
    Image.new("RGB", (64, 64), "gray").save(source_path)

    def fail_inspect(*args):
        raise ValueError("未知产品类别，请显式指定 --rules 配置文件。")

    monkeypatch.setattr(pipeline, "inspect_image", fail_inspect)
    monkeypatch.setattr("sys.argv", ["industrial_inspection", "predict", "--image", str(source_path),
                                     "--model-dir", str(tmp_path / "unknown")])
    with pytest.raises(SystemExit) as result:
        cli.main()
    assert result.value.code == 2
    stderr = capsys.readouterr().err
    assert "--rules" in stderr and "Traceback" not in stderr
