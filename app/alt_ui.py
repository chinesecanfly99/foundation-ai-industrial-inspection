"""A local-only Gradio interface; model inference is loaded on the first click."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
os.environ["TORCH_HOME"] = str(ROOT / ".cache" / "torch")
os.environ["GRADIO_TEMP_DIR"] = str(ROOT / ".cache" / "gradio")
_scratch = ROOT / ".cache" / "upload_tmp"
_scratch.mkdir(parents=True, exist_ok=True)
for _name in ("TMP", "TEMP", "TMPDIR"):
    os.environ[_name] = str(_scratch)
import tempfile
tempfile.tempdir = str(_scratch)

# Keep local presentation independent of online analytics and downloaded fonts.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from industrial_inspection.catalog import PRODUCTS, REAL_PRODUCTS, policy_path_for_product, product_id_from_card
from app.upload_cache import install_windows_upload_cache_fix
from app.localization import locale, localize_report, localized_products, locale_startup_js
from app.workflow_ui import build_workflow_tabs
from app.workbench import CSS, build_header, build_evidence, reset_evidence, attach_evidence
from industrial_inspection.workflow import InspectionStore
from industrial_inspection.workspace import results_dir

LANGUAGE = 'zh'
PRODUCTS = localized_products(LANGUAGE)
TEXT = locale(LANGUAGE)["ui"]


MAX_PIXELS = 16_000_000
MAX_DIMENSION = 6000
DECISIONS = {
    "pass": ("通过", "pass"),
    "review": ("人工复核", "review"),
    "reject": ("拒收", "reject"),
    "undetermined": ("无法判定", "undetermined"),
}



def _model_cards(project_root: Path) -> dict[str, dict[str, Any]]:
    cards = {}
    for card_path in sorted((project_root / "models").glob("*/model_card.json")):
        if not (card_path.parent / "model.npz").is_file():
            continue
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if isinstance(card, dict):
                cards[card_path.parent.name] = card
        except (OSError, ValueError):
            # An incomplete artifact must not prevent the rest of the UI loading.
            continue
    return cards


def _product_models(cards: dict[str, dict[str, Any]], product_id: str) -> list[str]:
    """Only offer artifacts whose model card agrees with the selected product."""
    return [name for name in PRODUCTS[product_id]["model_ids"]
            if name in cards and product_id_from_card(cards[name]) == product_id]


def _default_model(cards: dict[str, dict[str, Any]], product_id: str = "bottle") -> str | None:
    return next(iter(_product_models(cards, product_id)), None)


def _model_choices(cards: dict[str, dict[str, Any]], product_id: str) -> list[tuple[str, str]]:
    label = PRODUCTS[product_id]["label"]
    return [(("合成垫圈流程测试模型" if product_id == "synthetic" else
              f"特征记忆主模型 · {label}（ResNet18）" if name.endswith("_patch") else
              f"像素对比基线 · {label}"), name)
            for name in _product_models(cards, product_id)]


def _model_info(model_id: str | None, project_root: Path) -> str:
    cards = _model_cards(project_root)
    if not model_id or model_id not in cards:
        return "尚未找到模型。请先按 README 完成样例准备与模型训练，然后点击刷新模型。"
    card = cards[model_id]
    product_id = product_id_from_card(card)
    if product_id is None:
        return "模型卡未注明受支持的产品类别，暂不在页面中提供此模型。"
    if product_id == "synthetic":
        method = "人工合成垫圈的像素对比，仅检查上传、热图、规则和下载流程。"
    elif card.get("method") == "patch_memory":
        method = "主模型：用预训练 ResNet18 提取局部特征，与正常样本的特征库比较，定位视觉异常。"
    else:
        method = "像素基线：比较正常图片各位置的颜色统计，便于与主模型对比；对位置和光照更敏感。"
    counts = ""
    if isinstance(card.get("training_count"), int) and isinstance(card.get("calibration_count"), int):
        counts = f" 正常建模 {card['training_count']} 张，正常校准 {card['calibration_count']} 张。"
    return f"**当前类别：{PRODUCTS[product_id]['label']}** · `{model_id}`  \n{method}{counts}  \n请使用该类别和相近拍摄条件的图片；模型输出异常位置与分数，不自动判断缺陷名称。"


def _sample_paths(project_root: Path, product_id: str = "bottle") -> dict[str, Path]:
    folder = project_root / "samples" / PRODUCTS[product_id]["sample_dir"]
    return {path.stem: path for path in sorted(folder.glob("*.png"))}


def _sample_choices(project_root: Path, product_id: str) -> list[tuple[str, str]]:
    paths = _sample_paths(project_root, product_id)
    labels = PRODUCTS[product_id]["defect_labels"]
    choices = []
    for stem in paths:
        defect, _, number = stem.rpartition("_")
        if product_id == "synthetic":
            defect = defect.removeprefix("synthetic_")
        choices.append((f"{labels.get(defect, defect)} · 样例 {number}", stem))
    return sorted(choices, key=lambda pair: (not pair[1].startswith(("good_", "synthetic_good_")), pair[1]))


def _load_picture(path: Path | None) -> Image.Image | None:
    if path is None:
        return None
    try:
        with Image.open(path) as picture:
            return picture.convert("RGB").copy()
    except (OSError, ValueError):
        return None


def _product_info(product_id: str, project_root: Path) -> str:
    product = PRODUCTS[product_id]
    count = len(_sample_paths(project_root, product_id))
    note = "仅用于软件流程检查。" if product_id == "synthetic" else "样例名称来自数据集标注，不是模型识别出的缺陷类别。"
    return f"{product['description']} 当前可选 **{count} 张样例**。{note}"


def _policy_info(product_id: str, project_root: Path) -> str:
    try:
        policy = json.loads(policy_path_for_product(project_root, product_id).read_text(encoding="utf-8"))
        description = str(policy.get("description", "课程演示规则"))
        description = locale(LANGUAGE)["policies"].get(product_id, description)
    except (OSError, ValueError):
        description = "尚未找到对应规则，请先完成该类别的准备。"
    return f"**当前规则：{PRODUCTS[product_id]['label']}。** {description}  \n黄色框表示本类别配置的关键区域；阈值与区域尚需业务人员独立验证。"


def _clear_results():
    return (None, '<div class="decision-card"><h2>等待检测</h2><p>图片或模型已更新，请点击“开始检测”。</p></div>',
            "", "", [], None, gr.update(value=None, interactive=False), gr.update(value=None, interactive=False))


def _failed_results(message: str):
    cleared = list(_clear_results())
    cleared[1] = ('<div class="decision-card decision-undetermined"><h2>检测未完成</h2>'
                  f'<p>{html.escape(message)}</p></div>')
    return tuple(cleared)


def _load_sample_for_ui(chosen_sample: str | None, chosen_product: str, current_model: str | None, project_root: Path):
    paths = _sample_paths(project_root, chosen_product)
    picture = _load_picture(paths.get(chosen_sample))
    if chosen_sample and picture is None:
        gr.Warning("这个样例暂时无法读取，已清除旧图片和报告；请刷新样例或上传一张图片。")
    available = _model_cards(project_root)
    chosen_model = current_model if current_model in _product_models(available, chosen_product) else _default_model(available, chosen_product)
    return (picture, gr.update(choices=_model_choices(available, chosen_product), value=chosen_model),
            _model_info(chosen_model, project_root), *_clear_results())


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items() if key not in {"mask", "heatmap", "overlay"}}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return _safe_json(value.item())
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def _number(report: dict[str, Any], field: str, *, percent: bool = False) -> str:
    value = report.get(field)
    if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
        return "—"
    return f"{100 * value:.2f}%" if percent else f"{value:.4f}"


def _decision_html(report: dict[str, Any]) -> str:
    label, css_name = DECISIONS.get(str(report.get("decision")), ("无法判定", "undetermined"))
    if report.get("label_zh"):
        label = str(report["label_zh"])
    return (
        f'<div class="decision-card decision-{css_name}"><h2>{html.escape(label)}</h2>'
        '<p>依据异常位置、面积与质量规则给出本次演示决策。请结合原图和规则记录复核。</p></div>'
    )


def _metrics_html(report: dict[str, Any]) -> str:
    values = (
        ("异常分数", _number(report, "score")),
        ("图像阈值", _number(report, "image_threshold")),
        ("异常面积占比", _number(report, "anomaly_area_ratio", percent=True)),
        ("异常落在关键区", _number(report, "critical_overlap_ratio", percent=True)),
    )
    return '<div class="metrics-grid">' + "".join(
        f'<div class="metric"><span class="label">{label}</span><span class="value">{value}</span></div>'
        for label, value in values
    ) + "</div>"


def _trace_rows(trace: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for index, step in enumerate(trace, 1):
        name = step.get("rule_id", step.get("rule", step.get("id", step.get("name", f"规则 {index}"))))
        outcome = step.get("triggered", step.get("matched", step.get("passed", step.get("result", step.get("condition", "已检查")))))
        if isinstance(outcome, bool):
            outcome = ("命中并采用" if step.get("selected") else "命中，未采用") if outcome else "未命中"
        detail = step.get("reason", step.get("message", step.get("description", step.get("condition", ""))))
        if not detail:
            detail = json.dumps({k: v for k, v in step.items() if k not in {"rule_id", "rule", "id", "name", "triggered", "matched", "passed", "result"}}, ensure_ascii=False)
        rows.append([str(name), str(outcome), str(detail)])
    return rows


def inspect_for_ui(image: Image.Image | None, model_id: str | None, project_root: Path | None = None):
    """Run one inspection and prepare view/download values without changing a model."""
    project_root = (project_root or ROOT).resolve()
    if image is None:
        raise gr.Error("请先上传图片，或从演示样例中选择一张。")
    if not isinstance(image, Image.Image):
        raise gr.Error("图片读取失败，请上传 PNG 或 JPEG 图片。")
    if image.width * image.height > MAX_PIXELS or max(image.size) > MAX_DIMENSION:
        raise gr.Error("图片过大。请将长边缩至 6000 像素以内，且总像素不超过 1600 万。")
    cards = _model_cards(project_root)
    if not model_id or model_id not in cards:
        raise gr.Error("没有可用模型。请先完成模型训练，再刷新模型列表。")
    product_id = product_id_from_card(cards[model_id])
    if product_id is None or model_id not in _product_models(cards, product_id):
        raise gr.Error("模型卡未匹配已支持的产品类别，请使用对应产品的模型。")
    model_dir = project_root / "models" / model_id
    try:
        from industrial_inspection.pipeline import inspect_image

        overlay, report = inspect_image(
            ImageOps.exif_transpose(image).convert("RGB"),
            model_dir,
            policy_path_for_product(project_root, product_id),
        )
        safe_report = localize_report(_safe_json(report), LANGUAGE)
        safe_report = InspectionStore(project_root).save(safe_report, image, overlay, source="single",
            filename=image.info.get("inspection_filename", "uploaded_image.png"))
        export_dir = results_dir(project_root) / "ui_exports" / uuid.uuid4().hex
        export_dir.mkdir(parents=True, exist_ok=False)
        json_path = export_dir / "inspection_report.json"
        csv_path = export_dir / "inspection_summary.csv"
        json_path.write_text(json.dumps(safe_report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        scalar_report = {key: value for key, value in safe_report.items() if not isinstance(value, (dict, list))}
        scalar_report["reasons"] = "；".join(str(item) for item in safe_report.get("reasons", []))
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scalar_report))
            writer.writeheader()
            writer.writerow(scalar_report)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, OSError) as exc:
        raise gr.Error(f"本次检测未完成：{exc}") from exc
    reasons = safe_report.get("reasons", [])
    reasons_html = '<div class="rule-reasons"><strong>决策依据</strong><ul>' + "".join(
        f"<li>{html.escape(str(reason))}</li>" for reason in reasons
    ) + "</ul></div>"
    return (
        overlay,
        _decision_html(safe_report),
        _metrics_html(safe_report),
        reasons_html,
        _trace_rows(safe_report.get("trace", [])),
        safe_report,
        gr.DownloadButton(value=str(json_path), interactive=True),
        gr.DownloadButton(value=str(csv_path), interactive=True),
    )


def _run_for_ui(image: Image.Image | None, model_id: str | None, project_root: Path):
    """Recover expected user errors while preserving inspect_for_ui's API."""
    try:
        return inspect_for_ui(image, model_id, project_root)
    except gr.Error as exc:
        message = str(exc.message)
        gr.Warning(message)
        return _failed_results(message)



def _run_selected_for_ui(image: Image.Image | None, model_id: str | None,
                         chosen_product: str, chosen_sample: str | None, project_root: Path):
    """Reject a half-loaded selection before generating a report for its image."""
    try:
        cards = _model_cards(project_root)
        if chosen_product not in PRODUCTS or (model_id in cards and
                product_id_from_card(cards[model_id]) != chosen_product):
            raise gr.Error("产品和模型仍在切换，请等待加载完成后重新检测。")
        if chosen_sample and isinstance(image, Image.Image):
            expected = _load_picture(_sample_paths(project_root, chosen_product).get(chosen_sample))
            if expected is None:
                raise gr.Error("所选样例尚未加载，请刷新样例后重新检测。")
            actual = ImageOps.exif_transpose(image).convert("RGB")
            expected.thumbnail((1024, 1024))
            actual.thumbnail((1024, 1024))
            if actual.size != expected.size or actual.tobytes() != expected.tobytes():
                raise gr.Error("图片与所选样例仍未同步，请等待加载完成后重新检测。")
        if isinstance(image, Image.Image):
            image.info["inspection_filename"] = f"{chosen_sample}.png" if chosen_sample else "uploaded_image.png"
        return _run_for_ui(image, model_id, project_root)
    except gr.Error as exc:
        message = str(exc.message)
        gr.Warning(message)
        return _failed_results(message)


def create_app(project_root: Path | None = None) -> gr.Blocks:
    install_windows_upload_cache_fix()
    project_root = (project_root or ROOT).resolve()
    cards = _model_cards(project_root)
    product_id = "bottle"
    selected = _default_model(cards, product_id)
    initial_choices = _sample_choices(project_root, product_id)
    initial_sample = initial_choices[0][1] if initial_choices else None
    initial_image = _load_picture(_sample_paths(project_root, product_id).get(initial_sample))
    # System fonts keep startup usable offline.
    theme = gr.themes.Default(primary_hue="blue", secondary_hue="teal", neutral_hue="slate", font=["Arial", "Microsoft YaHei", "sans-serif"], font_mono=["Consolas", "Courier New", "monospace"])
    with gr.Blocks(title="工业异常定位 · 质量决策", theme=theme, css=CSS, js=locale_startup_js(LANGUAGE), analytics_enabled=False, delete_cache=(3600, 86400)) as demo:
        build_header(LANGUAGE, TEXT)
        with gr.Tabs(elem_id="inspection-tabs"):
            with gr.Tab(TEXT["single"], elem_id="single-tab"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=320):
                        gr.Markdown("### 1 · 选择产品、图片与模型")
                        with gr.Row():
                            product = gr.Dropdown(label="产品类别", choices=[(PRODUCTS[name]["label"], name) for name in REAL_PRODUCTS], value=product_id, elem_id="product-choice", interactive=True)
                            sample = gr.Dropdown(label="演示样例", choices=initial_choices, value=initial_sample, elem_id="sample-choice", interactive=True)
                        input_image = gr.Image(value=initial_image, label="待检图片", type="pil", sources=["upload"], height=294, format="png", elem_id="inspection-input", placeholder="上传对应产品图片，或在上方选择演示样例")
                        detect = gr.Button("开始检测", variant="primary", size="lg", elem_id="inspect-button")
                        with gr.Accordion("模型与样例设置", open=False):
                            with gr.Row():
                                model = gr.Dropdown(label="检测模型", choices=_model_choices(cards, product_id), value=selected, scale=4, elem_id="model-choice", interactive=True)
                                refresh = gr.Button("刷新模型与样例", scale=1, min_width=110, elem_id="refresh-models")
                            model_info = gr.Markdown(_model_info(selected, project_root), elem_id="model-description")
                            product_info = gr.Markdown(_product_info(product_id, project_root), elem_id="product-description")
                            show_synthetic = gr.Checkbox(label="显示合成垫圈流程测试", value=False, info="开启后在产品列表中增加合成联调。它不属于三类真实工业数据。", elem_id="show-synthetic")
                    with gr.Column(scale=6, min_width=320):
                        gr.Markdown("### 2 · 查看异常与决策")
                        decision = gr.HTML('<div class="decision-card"><h2>等待检测</h2><p>选择一张图片，点击“开始检测”。</p></div>', elem_id="quality-decision")
                        output_image = gr.Image(label="异常热图", type="pil", interactive=False, height=294, format="png", elem_id="inspection-overlay")
                        metrics = gr.HTML("", elem_id="inspection-metrics")
                        reasons = gr.HTML("", elem_id="inspection-reasons")
                evidence = build_evidence(LANGUAGE)
                with gr.Row():
                    json_download = gr.DownloadButton("下载 JSON 报告", interactive=False, elem_id="download-json")
                    csv_download = gr.DownloadButton("下载 CSV 摘要", interactive=False, elem_id="download-csv")
                with gr.Accordion("3 · 查看逐条规则与完整记录", open=False):
                    policy_info = gr.Markdown(_policy_info(product_id, project_root), elem_id="policy-description")
                    trace = gr.Dataframe(headers=["规则", "触发 / 检查结果", "说明"], datatype=["str", "str", "str"], type="array", interactive=False, wrap=True, label="规则检查记录", elem_id="rule-trace")
                    full_report = gr.JSON(label="本次检测报告", elem_id="inspection-json")

            build_workflow_tabs(project_root, LANGUAGE)

        results = [output_image, decision, metrics, reasons, trace, full_report, json_download, csv_download]
        ui_results = [*results, *evidence]
        controls = [input_image, product, sample, model, refresh, show_synthetic, detect]

        def run(image, model_id, chosen_product, chosen_sample):
            values = list(_run_selected_for_ui(image, model_id, chosen_product, chosen_sample, project_root))
            evidence_values, combined_paths = attach_evidence(values[5], image, chosen_product, project_root, LANGUAGE)
            if combined_paths:
                values[6:8] = [gr.DownloadButton(value=path, interactive=True) for path in combined_paths]
            return (*values, *evidence_values)

        def select_product(chosen_product, current_model=None):
            if chosen_product not in PRODUCTS:
                chosen_product = "bottle"
            available = _model_cards(project_root)
            valid_models = _product_models(available, chosen_product)
            chosen_model = current_model if current_model in valid_models else _default_model(available, chosen_product)
            sample_choices = _sample_choices(project_root, chosen_product)
            chosen_sample = sample_choices[0][1] if sample_choices else None
            picture = _load_picture(_sample_paths(project_root, chosen_product).get(chosen_sample))
            return (picture, gr.update(choices=sample_choices, value=chosen_sample),
                    gr.update(choices=_model_choices(available, chosen_product), value=chosen_model),
                    _model_info(chosen_model, project_root), _product_info(chosen_product, project_root),
                    _policy_info(chosen_product, project_root), *_clear_results(), *reset_evidence())

        selection_outputs = [input_image, sample, model, model_info, product_info, policy_info, *ui_results]

        def load_sample(chosen_sample, chosen_product, current_model):
            return (*_load_sample_for_ui(chosen_sample, chosen_product, current_model, project_root), *reset_evidence())

        def toggle_synthetic(enabled, current_product):
            choices = [*REAL_PRODUCTS, *(["synthetic"] if enabled else [])]
            chosen_product = current_product if current_product in choices else "bottle"
            return (gr.update(choices=[(PRODUCTS[name]["label"], name) for name in choices], value=chosen_product),
                    *select_product(chosen_product))

        def begin_run():
            cleared = [*_clear_results(), *reset_evidence()]
            cleared[1] = '<div class="decision-card"><h2>正在检测</h2><p>完成后可查看热图、规则和本次报告。</p></div>'
            return (*[gr.update(interactive=False) for _ in controls], *cleared)

        def begin_selection():
            cleared = [*_clear_results(), *reset_evidence()]
            cleared[1] = '<div class="decision-card"><h2>正在加载</h2><p>正在更新产品、样例和模型，请稍候。</p></div>'
            return (*[gr.update(interactive=False) for _ in controls], *cleared)

        def unlock_controls():
            return [gr.update(interactive=True) for _ in controls]

        # Product/sample loads and inference share one queue. A late load cannot
        # overwrite an in-flight inspection, and a half-loaded click is checked
        # against the product and sample content before any report is produced.
        queue_options = {"concurrency_id": "inspection-selection", "concurrency_limit": 1}
        start = detect.click(begin_run, outputs=[*controls, *ui_results], queue=False, api_name=False)
        inference = start.then(run, inputs=[input_image, model, product, sample], outputs=ui_results,
                               api_name=False, **queue_options)
        inference.then(unlock_controls, outputs=controls, queue=False, api_name=False)

        def bind_selection(event, callback, inputs, outputs):
            loading = event(begin_selection, outputs=[*controls, *ui_results], queue=False,
                            api_name=False, trigger_mode="always_last")
            loaded = loading.then(callback, inputs=inputs, outputs=outputs, api_name=False, **queue_options)
            loaded.then(unlock_controls, outputs=controls, queue=False, api_name=False)

        # Programmatic updates already load their own image and must not start
        # another selection callback. Only a human input enters these chains.
        bind_selection(product.input, select_product, product, selection_outputs)
        bind_selection(sample.input, load_sample, [sample, product, model],
                       [input_image, model, model_info, *ui_results])
        bind_selection(refresh.click, select_product, [product, model], selection_outputs)
        bind_selection(show_synthetic.input, toggle_synthetic, [show_synthetic, product],
                       [product, *selection_outputs])
        bind_selection(model.input, lambda name: (_model_info(name, project_root), *_clear_results(), *reset_evidence()),
                       model, [model_info, *ui_results])
        input_image.change(lambda: (*_clear_results(), *reset_evidence()), outputs=ui_results, queue=False, api_name=False)
        input_image.upload(lambda: (gr.update(value=None), *_clear_results(), *reset_evidence()), outputs=[sample, *ui_results], queue=False, api_name=False)
        input_image.clear(lambda: (gr.update(value=None), *_clear_results(), *reset_evidence()), outputs=[sample, *ui_results], queue=False, api_name=False)

        # Preserve the existing two-input API for integrations. Visible UI runs
        # use the additional product/sample consistency check above.
        api_inspect = gr.Button(visible=False)
        api_inspect.click(lambda image, model_id: _run_for_ui(image, model_id, project_root),
                          inputs=[input_image, model], outputs=results, api_name="inspect", **queue_options)
    return demo.queue(max_size=8, default_concurrency_limit=1)


