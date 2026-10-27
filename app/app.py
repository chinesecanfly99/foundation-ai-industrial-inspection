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
from industrial_inspection.workflow import InspectionStore

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

CSS = """
.gradio-container { width: 100% !important; min-width: 0 !important;
  max-width: 1280px !important; margin: auto; box-sizing: border-box; }
.gradio-container main.app, .gradio-container main.app > .wrap { min-width: 0; }
#rule-trace, #inspection-json { max-width: 100%; min-width: 0 !important;
  overflow: auto !important; }
.project-hero { padding: 26px 30px; border-radius: 20px; background: #102d3a;
  color: #f5fbff; margin-bottom: 18px; }
.project-eyebrow { color: #8de3cd; font-size: 12px; font-weight: 700;
  letter-spacing: .14em; margin: 0 0 10px; }
.project-hero h1 { color: #f5fbff !important; font-size: 29px !important;
  line-height: 1.35; margin: 0 0 10px !important; }
.project-hero p { color: #c6dce5; margin: 0; line-height: 1.8; }
.project-note { padding: 13px 17px; border-radius: 12px; color: #33505e;
  background: #eaf5f5; border: 1px solid #cce4e4; line-height: 1.7; }
.decision-card { border-radius: 14px; border: 1px solid #dce6ed;
  background: #f6f9fb; padding: 20px; min-height: 126px; }
.decision-card h2 { font-size: 24px !important; margin: 0 0 9px !important; }
.decision-card p { margin: 0; line-height: 1.75; }
.decision-pass { background: #ecf8f0; border-color: #a8d6b7; color: #1c5632; }
.decision-review { background: #fff8e8; border-color: #edcc83; color: #805712; }
.decision-reject { background: #fff0ee; border-color: #eab7b0; color: #8a3027; }
.decision-undetermined { background: #f1f3f6; border-color: #cbd5df; color: #47596c; }
.metrics-grid { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 10px; margin-top: 10px; }
.metric { background: #fff; border: 1px solid #dce6ed; border-radius: 12px;
  padding: 12px 14px; }
.metric .label { display: block; color: #546d7c; font-size: 12px; margin-bottom: 5px; }
.metric .value { color: #153b4c; font-size: 20px; font-weight: 700; }
.metric .unit { font-size: 11px; font-weight: 400; color: #657d8a; }
.rule-reasons { color: #365464; line-height: 1.8; padding: 3px 1px; }
.rule-reasons ul { margin: 8px 0; padding-left: 22px; }
@media (max-width: 700px) {
  .gradio-container .row { flex-direction: column; }
  .gradio-container .row > .column, .gradio-container .row > .form {
    width: 100%; min-width: 0 !important; flex: 1 1 auto !important;
  }
  .project-hero { padding: 22px; }
  .project-hero h1 { font-size: 24px !important; }
  .metrics-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
}
"""


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
        export_dir = project_root / "results" / "ui_exports" / uuid.uuid4().hex
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
    theme = gr.themes.Default(primary_hue="teal", secondary_hue="blue", neutral_hue="slate", font=["Arial", "Microsoft YaHei", "sans-serif"], font_mono=["Consolas", "Courier New", "monospace"])
    with gr.Blocks(title="工业异常定位 · 质量决策", theme=theme, css=CSS, js=locale_startup_js(LANGUAGE), analytics_enabled=False, delete_cache=(3600, 86400)) as demo:
        language_toggle = gr.Button(TEXT["switch"], elem_id="language-toggle", size="sm")
        language_toggle.click(fn=None, js="() => { window.location.assign('/en/'); return []; }", queue=False, api_name=False)
        gr.Markdown(TEXT["switch_note"], elem_id="language-note")
        if (project_root / "EN_LOCAL_ONLY.txt").exists():
            gr.Markdown('此处为独立的本地项目副本，记录与原项目分开保存。')
        gr.HTML(
            '<div class="project-hero"><div class="project-eyebrow">FOUNDATION OF AI · GROUP PROJECT</div>'
            '<h1>工业异常定位与可解释质量决策</h1>'
            '<p>瓶口、金属螺母、瓷砖表面：查看异常位置，理解质量规则，再决定通过、复核或拒收。</p></div>'
        )
        gr.HTML(
            '<div class="project-note">课程原型 · 每类产品使用各自模型和规则。'
            '异常分数不是概率，热图显示视觉差异；检测结果不推断缺陷的物理原因。真实性能以独立测试集评估为准。</div>'
        )
        with gr.Tabs(elem_id="inspection-tabs"):
            with gr.Tab(TEXT["single"], elem_id="single-tab"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=320):
                        gr.Markdown("### 1 · 选择产品、图片与模型")
                        product = gr.Dropdown(label="产品类别", choices=[(PRODUCTS[name]["label"], name) for name in REAL_PRODUCTS], value=product_id, elem_id="product-choice", interactive=True)
                        product_info = gr.Markdown(_product_info(product_id, project_root), elem_id="product-description")
                        input_image = gr.Image(value=initial_image, label="待检图片", type="pil", sources=["upload"], height=350, format="png", elem_id="inspection-input", placeholder="上传对应产品图片，或在下方选择演示样例")
                        sample = gr.Dropdown(label="演示样例", choices=initial_choices, value=initial_sample, info="按数据集类别显示中文名称；编号不同代表不同图片。上传自定义图片后会清除样例选择。", elem_id="sample-choice", interactive=True)
                        with gr.Row():
                            model = gr.Dropdown(label="检测模型", choices=_model_choices(cards, product_id), value=selected, scale=4, elem_id="model-choice", interactive=True)
                            refresh = gr.Button("刷新模型与样例", scale=1, min_width=110, elem_id="refresh-models")
                        model_info = gr.Markdown(_model_info(selected, project_root), elem_id="model-description")
                        with gr.Accordion("高级 · 开发流程检查", open=False):
                            show_synthetic = gr.Checkbox(label="显示合成垫圈流程测试", value=False, info="开启后在产品列表中增加合成联调。它不属于三类真实工业数据。", elem_id="show-synthetic")
                        detect = gr.Button("开始检测", variant="primary", size="lg", elem_id="inspect-button")
                    with gr.Column(scale=6, min_width=320):
                        gr.Markdown("### 2 · 查看异常与决策")
                        output_image = gr.Image(label="异常热图叠加 · 颜色越暖，异常响应越高", type="pil", interactive=False, height=350, format="png", elem_id="inspection-overlay")
                        decision = gr.HTML('<div class="decision-card"><h2>等待检测</h2><p>选择一张图片，点击“开始检测”。</p></div>', elem_id="quality-decision")
                        metrics = gr.HTML("", elem_id="inspection-metrics")
                        reasons = gr.HTML("", elem_id="inspection-reasons")
                with gr.Accordion("3 · 查看逐条规则与完整记录", open=False):
                    policy_info = gr.Markdown(_policy_info(product_id, project_root), elem_id="policy-description")
                    trace = gr.Dataframe(headers=["规则", "触发 / 检查结果", "说明"], datatype=["str", "str", "str"], type="array", interactive=False, wrap=True, label="规则检查记录", elem_id="rule-trace")
                    full_report = gr.JSON(label="本次检测报告", elem_id="inspection-json")
                with gr.Row():
                    json_download = gr.DownloadButton("下载 JSON 报告", interactive=False, elem_id="download-json")
                    csv_download = gr.DownloadButton("下载 CSV 摘要", interactive=False, elem_id="download-csv")
                gr.Markdown("**报告用途：** CSV 是当前这一张图片的一行摘要，便于用 Excel 汇总实验；JSON 保存完整指标、产品和模型信息及逐条规则，便于复核。热图可在图片区域单独下载。")
                gr.Markdown("**演示方法：** 每类先看正常样例，再看各类异常，比较热图与规则；可切换主模型和像素基线对比。低质量图片应进入复核或无法判定。切换产品、图片或模型后需重新检测。")
            build_workflow_tabs(project_root, LANGUAGE)

        results = [output_image, decision, metrics, reasons, trace, full_report, json_download, csv_download]
        controls = [input_image, product, sample, model, refresh, show_synthetic, detect]

        def run(image, model_id, chosen_product, chosen_sample):
            return _run_selected_for_ui(image, model_id, chosen_product, chosen_sample, project_root)

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
                    _policy_info(chosen_product, project_root), *_clear_results())

        selection_outputs = [input_image, sample, model, model_info, product_info, policy_info, *results]

        def load_sample(chosen_sample, chosen_product, current_model):
            return _load_sample_for_ui(chosen_sample, chosen_product, current_model, project_root)

        def toggle_synthetic(enabled, current_product):
            choices = [*REAL_PRODUCTS, *(["synthetic"] if enabled else [])]
            chosen_product = current_product if current_product in choices else "bottle"
            return (gr.update(choices=[(PRODUCTS[name]["label"], name) for name in choices], value=chosen_product),
                    *select_product(chosen_product))

        def begin_run():
            cleared = list(_clear_results())
            cleared[1] = '<div class="decision-card"><h2>正在检测</h2><p>完成后可查看热图、规则和本次报告。</p></div>'
            return (*[gr.update(interactive=False) for _ in controls], *cleared)

        def begin_selection():
            cleared = list(_clear_results())
            cleared[1] = '<div class="decision-card"><h2>正在加载</h2><p>正在更新产品、样例和模型，请稍候。</p></div>'
            return (*[gr.update(interactive=False) for _ in controls], *cleared)

        def unlock_controls():
            return [gr.update(interactive=True) for _ in controls]

        # Product/sample loads and inference share one queue. A late load cannot
        # overwrite an in-flight inspection, and a half-loaded click is checked
        # against the product and sample content before any report is produced.
        queue_options = {"concurrency_id": "inspection-selection", "concurrency_limit": 1}
        start = detect.click(begin_run, outputs=[*controls, *results], queue=False, api_name=False)
        inference = start.then(run, inputs=[input_image, model, product, sample], outputs=results,
                               api_name=False, **queue_options)
        inference.then(unlock_controls, outputs=controls, queue=False, api_name=False)

        def bind_selection(event, callback, inputs, outputs):
            loading = event(begin_selection, outputs=[*controls, *results], queue=False,
                            api_name=False, trigger_mode="always_last")
            loaded = loading.then(callback, inputs=inputs, outputs=outputs, api_name=False, **queue_options)
            loaded.then(unlock_controls, outputs=controls, queue=False, api_name=False)

        # Programmatic updates already load their own image and must not start
        # another selection callback. Only a human input enters these chains.
        bind_selection(product.input, select_product, product, selection_outputs)
        bind_selection(sample.input, load_sample, [sample, product, model],
                       [input_image, model, model_info, *results])
        bind_selection(refresh.click, select_product, [product, model], selection_outputs)
        bind_selection(show_synthetic.input, toggle_synthetic, [show_synthetic, product],
                       [product, *selection_outputs])
        bind_selection(model.input, lambda name: (_model_info(name, project_root), *_clear_results()),
                       model, [model_info, *results])
        input_image.change(_clear_results, outputs=results, queue=False, api_name=False)
        input_image.upload(lambda: (gr.update(value=None), *_clear_results()), outputs=[sample, *results], queue=False, api_name=False)
        input_image.clear(lambda: (gr.update(value=None), *_clear_results()), outputs=[sample, *results], queue=False, api_name=False)

        # Preserve the existing two-input API for integrations. Visible UI runs
        # use the additional product/sample consistency check above.
        api_inspect = gr.Button(visible=False)
        api_inspect.click(lambda image, model_id: _run_for_ui(image, model_id, project_root),
                          inputs=[input_image, model], outputs=results, api_name="inspect", **queue_options)
    return demo.queue(max_size=8, default_concurrency_limit=1)


def main() -> None:
    parser = argparse.ArgumentParser(description="启动工业异常定位课程演示")
    parser.add_argument("--host", default="127.0.0.1", choices=["127.0.0.1", "localhost"], help="仅监听本机")
    parser.add_argument("--port", default=7860, type=int)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("端口必须在 1 到 65535 之间。")
    # Gradio checks its own loopback endpoint during launch. Corporate/system
    # HTTP proxies must not intercept that request; retain all other exclusions.
    exclusions = list(dict.fromkeys(item.strip() for key in ("NO_PROXY", "no_proxy") for item in os.environ.get(key, "").split(",") if item.strip()))
    for host in ("127.0.0.1", "localhost", "::1"):
        if host not in exclusions:
            exclusions.append(host)
    os.environ["NO_PROXY"] = os.environ["no_proxy"] = ",".join(exclusions)
    from app.alt_ui import create_app as create_alternate_app
    from app.server import launch
    launch(create_app(), create_alternate_app(), default_language=LANGUAGE, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
