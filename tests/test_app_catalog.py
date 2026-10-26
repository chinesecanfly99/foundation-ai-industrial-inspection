"""UI category boundaries and reports must agree with the chosen artifact."""
import json

import pytest
from PIL import Image

pytest.importorskip("gradio")

import app.app as app_module
from app.app import _clear_results, _load_sample_for_ui, _model_cards, _model_choices, _run_for_ui, inspect_for_ui
from industrial_inspection.catalog import PRODUCTS, REAL_PRODUCTS


def write_card(root, model_id, source, *, artifact=True):
    directory = root / "models" / model_id
    directory.mkdir(parents=True)
    (directory / "model_card.json").write_text(json.dumps({"dataset_source": source}), encoding="utf-8")
    if artifact:
        (directory / "model.npz").write_bytes(b"test artifact; pipeline mocked")


def test_model_choices_hide_other_products_and_incomplete_artifacts(tmp_path):
    write_card(tmp_path, "bottle_patch", "mvtec_ad/bottle")
    write_card(tmp_path, "bottle_pixel", "mvtec_ad/metal_nut")  # mislabeled directory
    write_card(tmp_path, "metal_nut_patch", "mvtec_ad/metal_nut")
    write_card(tmp_path, "tile_patch", "mvtec_ad/tile", artifact=False)
    write_card(tmp_path, "synthetic_pixel", "synthetic")
    cards = _model_cards(tmp_path)
    assert [value for _, value in _model_choices(cards, "bottle")] == ["bottle_patch"]
    assert [value for _, value in _model_choices(cards, "metal_nut")] == ["metal_nut_patch"]
    assert _model_choices(cards, "tile") == []
    assert "tile_patch" not in cards


@pytest.mark.parametrize("product_id", REAL_PRODUCTS)
def test_ui_uses_selected_products_policy_and_exports_same_report(tmp_path, monkeypatch, product_id):
    model_id = f"{product_id}_patch"
    write_card(tmp_path, model_id, f"mvtec_ad/{product_id}")
    image = Image.new("RGB", (64, 64), "gray")

    def fake_inspect(picture, model_dir, config_path):
        assert model_dir == tmp_path / "models" / model_id
        assert config_path == tmp_path / PRODUCTS[product_id]["policy_file"]
        return picture, {"product_category": product_id, "model_id": model_id,
                         "decision": "review", "trace": [], "reasons": ["test"]}

    monkeypatch.setattr("industrial_inspection.pipeline.inspect_image", fake_inspect)
    output = inspect_for_ui(image, model_id, tmp_path)
    assert output[5]["product_category"] == product_id
    json_paths = list((tmp_path / "results" / "ui_exports").glob("*/inspection_report.json"))
    assert len(json_paths) == 1
    assert json.loads(json_paths[0].read_text(encoding="utf-8"))["product_category"] == product_id
    assert json_paths[0].with_name("inspection_summary.csv").is_file()


def test_changing_inputs_clears_report_and_both_download_values():
    cleared = _clear_results()
    assert cleared[0] is None
    assert cleared[4] == []
    assert cleared[5] is None
    for download in cleared[6:]:
        assert download["value"] is None
        assert download["interactive"] is False


def test_expected_detection_error_finishes_with_no_stale_report(tmp_path, monkeypatch):
    warnings = []
    monkeypatch.setattr(app_module.gr, "Warning", warnings.append)
    output = _run_for_ui(None, "bottle_patch", tmp_path)
    assert "检测未完成" in output[1]
    assert "请先上传图片" in output[1]
    assert "正在检测" not in output[1]
    assert output[0] is None and output[5] is None
    assert all(item["value"] is None and not item["interactive"] for item in output[6:])
    assert warnings == ["请先上传图片，或从演示样例中选择一张。"]
    # The original inference API still raises the expected Gradio error.
    with pytest.raises(app_module.gr.Error):
        inspect_for_ui(None, "bottle_patch", tmp_path)


def test_unexpected_detection_exception_is_not_hidden(tmp_path, monkeypatch):
    def broken_inference(*args):
        raise TypeError("unexpected programming failure")

    monkeypatch.setattr(app_module, "inspect_for_ui", broken_inference)
    with pytest.raises(TypeError, match="unexpected programming failure"):
        _run_for_ui(None, "bottle_patch", tmp_path)


@pytest.mark.parametrize("corrupt", [False, True])
def test_unreadable_sample_clears_image_and_downloads(tmp_path, monkeypatch, corrupt):
    warnings = []
    monkeypatch.setattr(app_module.gr, "Warning", warnings.append)
    if corrupt:
        folder = tmp_path / "samples" / "bottle"
        folder.mkdir(parents=True)
        (folder / "good_000.png").write_bytes(b"not an image")
    output = _load_sample_for_ui("good_000", "bottle", None, tmp_path)
    assert output[0] is None
    assert output[3] is None  # old heatmap
    assert output[8] is None  # old JSON report
    assert all(item["value"] is None and not item["interactive"] for item in output[9:])
    assert warnings == ["这个样例暂时无法读取，已清除旧图片和报告；请刷新样例或上传一张图片。"]


def test_half_loaded_product_cannot_create_wrong_category_report(tmp_path, monkeypatch):
    write_card(tmp_path, "bottle_patch", "mvtec_ad/bottle")
    warnings = []
    monkeypatch.setattr(app_module.gr, "Warning", warnings.append)
    def must_not_inspect(*args):
        pytest.fail("Inference must not run with a model from the previous product")
    monkeypatch.setattr(app_module, "inspect_for_ui", must_not_inspect)
    output = app_module._run_selected_for_ui(Image.new("RGB", (64, 64)), "bottle_patch",
                                              "metal_nut", "good_000", tmp_path)
    assert "产品和模型仍在切换" in output[1]
    assert output[5] is None
    assert all(not item["interactive"] and item["value"] is None for item in output[6:])
    assert len(warnings) == 1


def test_half_loaded_sample_cannot_export_previous_image(tmp_path, monkeypatch):
    write_card(tmp_path, "metal_nut_patch", "mvtec_ad/metal_nut")
    folder = tmp_path / "samples/metal_nut"
    folder.mkdir(parents=True)
    Image.new("RGB", (64, 64), "white").save(folder / "good_000.png")
    monkeypatch.setattr(app_module.gr, "Warning", lambda message: None)
    monkeypatch.setattr(app_module, "inspect_for_ui", lambda *args: pytest.fail("Old image reached inference"))
    output = app_module._run_selected_for_ui(Image.new("RGB", (64, 64), "black"),
                                              "metal_nut_patch", "metal_nut", "good_000", tmp_path)
    assert "图片与所选样例仍未同步" in output[1]
    assert output[5] is None and not output[6]["interactive"]


def test_loaded_sample_and_uploaded_image_both_reach_inference(tmp_path, monkeypatch):
    write_card(tmp_path, "tile_patch", "mvtec_ad/tile")
    folder = tmp_path / "samples/tile"
    folder.mkdir(parents=True)
    image = Image.new("RGB", (64, 64), "gray")
    image.save(folder / "good_000.png")
    called = []
    marker = object()
    def inspect(picture, model_id, root):
        called.append((picture, model_id, root))
        return marker
    monkeypatch.setattr(app_module, "inspect_for_ui", inspect)
    assert app_module._run_selected_for_ui(image, "tile_patch", "tile", "good_000", tmp_path) is marker
    uploaded = Image.new("RGB", (80, 80), "red")
    assert app_module._run_selected_for_ui(uploaded, "tile_patch", "tile", None, tmp_path) is marker
    assert len(called) == 2 and called[1][0] is uploaded


def test_removed_sample_cannot_reuse_old_picture(tmp_path, monkeypatch):
    write_card(tmp_path, "tile_patch", "mvtec_ad/tile")
    monkeypatch.setattr(app_module.gr, "Warning", lambda message: None)
    monkeypatch.setattr(app_module, "inspect_for_ui", lambda *args: pytest.fail("Missing sample reached inference"))
    output = app_module._run_selected_for_ui(Image.new("RGB", (64, 64)),
                                              "tile_patch", "tile", "good_000", tmp_path)
    assert "所选样例尚未加载" in output[1]
    assert output[5] is None
