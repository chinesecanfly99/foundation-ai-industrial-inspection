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

# Keep local presentation independent of online analytics and downloaded fonts.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

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
        try:
            card = json.loads(card_path.read_text(encoding="utf-8"))
            if isinstance(card, dict):
                cards[card_path.parent.name] = card
        except (OSError, ValueError):
            # An incomplete artifact must not prevent the rest of the UI loading.
            continue
    return cards


def _default_model(cards: dict[str, dict[str, Any]]) -> str | None:
    return next((name for name in ("bottle_patch", "synthetic_pixel") if name in cards), next(iter(cards), None))


def _model_info(model_id: str | None, project_root: Path) -> str:
    cards = _model_cards(project_root)
    if not model_id or model_id not in cards:
        return "尚未找到模型。请先按 README 完成样例准备与模型训练，然后点击刷新模型。"
    card = cards[model_id]
    source = card.get("dataset_source", card.get("data_source", card.get("dataset", "来源见模型卡")))
    method = card.get("method", card.get("model_method", card.get("detector_type", "方法见模型卡")))
    is_synthetic = "synthetic" in str(source).lower() or model_id.startswith("synthetic")
    note = "合成样例模型：只验证流程，不能代表真实工业性能。" if is_synthetic else "请使用对应产品、相近拍摄条件的图片；真实测试结果见独立评估记录。"
    return f"**当前模型：** `{model_id}` · **数据：** {html.escape(str(source))} · **方法：** {html.escape(str(method))}  \n{note}"


def _sample_paths(project_root: Path) -> dict[str, Path]:
    paths = {}
    for folder, title in (("demo", "合成联调"), ("bottle", "MVTec bottle")):
        for path in sorted((project_root / "samples" / folder).glob("*.png")):
            paths[f"{title} / {path.stem}"] = path
    return paths


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
    if not model_id or model_id not in _model_cards(project_root):
        raise gr.Error("没有可用模型。请先完成模型训练，再刷新模型列表。")
    model_dir = project_root / "models" / model_id
    try:
        from industrial_inspection.pipeline import inspect_image

        overlay, report = inspect_image(
            ImageOps.exif_transpose(image).convert("RGB"),
            model_dir,
            project_root / "configs" / "quality_rules.json",
        )
        safe_report = _safe_json(report)
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


def create_app(project_root: Path | None = None) -> gr.Blocks:
    project_root = (project_root or ROOT).resolve()
    cards = _model_cards(project_root)
    selected = _default_model(cards)
    samples = _sample_paths(project_root)
    # System fonts keep startup usable offline.
    theme = gr.themes.Default(primary_hue="teal", secondary_hue="blue", neutral_hue="slate", font=["Arial", "Microsoft YaHei", "sans-serif"], font_mono=["Consolas", "Courier New", "monospace"])
    with gr.Blocks(title="工业异常定位 · 质量决策", theme=theme, css=CSS, analytics_enabled=False, delete_cache=(3600, 86400)) as demo:
        gr.HTML(
            '<div class="project-hero"><div class="project-eyebrow">FOUNDATION OF AI · GROUP PROJECT</div>'
            '<h1>工业异常定位与可解释质量决策</h1>'
            '<p>查看异常在哪里，理解质量规则为什么触发，再决定通过、复核或拒收。</p></div>'
        )
        gr.HTML(
            '<div class="project-note">课程原型 · 合成样例仅用于流程联调；真实工业性能以独立测试集评估为准。'
            '异常分数不是概率，热图显示模型关注的异常位置，不推断缺陷的物理原因。</div>'
        )
        with gr.Row(equal_height=False):
            with gr.Column(scale=5, min_width=320):
                gr.Markdown("### 1 · 选择图片与模型")
                input_image = gr.Image(label="待检图片", type="pil", sources=["upload"], height=350, format="png", elem_id="inspection-input", placeholder="上传产品图片，或在下方选择演示样例")
                sample = gr.Dropdown(label="演示样例", choices=list(samples), value=None, info="选择样例后自动载入图片，并匹配可用模型。", elem_id="sample-choice")
                with gr.Row():
                    model = gr.Dropdown(label="检测模型", choices=list(cards), value=selected, scale=4, elem_id="model-choice")
                    refresh = gr.Button("刷新模型", scale=1, min_width=90)
                model_info = gr.Markdown(_model_info(selected, project_root), elem_id="model-description")
                detect = gr.Button("开始检测", variant="primary", size="lg", elem_id="inspect-button")
            with gr.Column(scale=6, min_width=320):
                gr.Markdown("### 2 · 查看异常与决策")
                output_image = gr.Image(label="异常热图叠加 · 颜色越暖，异常响应越高", type="pil", interactive=False, height=350, format="png", elem_id="inspection-overlay")
                decision = gr.HTML('<div class="decision-card"><h2>等待检测</h2><p>选择一张图片，点击“开始检测”。</p></div>', elem_id="quality-decision")
                metrics = gr.HTML("", elem_id="inspection-metrics")
                reasons = gr.HTML("", elem_id="inspection-reasons")
        with gr.Accordion("3 · 查看逐条规则与完整记录", open=False):
            gr.Markdown("规则阈值和关键区域是课程演示配置，需要业务人员独立标注后才能评估决策正确率。")
            trace = gr.Dataframe(headers=["规则", "触发 / 检查结果", "说明"], datatype=["str", "str", "str"], type="array", interactive=False, wrap=True, label="规则检查记录", elem_id="rule-trace")
            full_report = gr.JSON(label="本次检测报告", elem_id="inspection-json")
        with gr.Row():
            json_download = gr.DownloadButton("下载 JSON 报告", interactive=False, elem_id="download-json")
            csv_download = gr.DownloadButton("下载 CSV 摘要", interactive=False, elem_id="download-csv")
        gr.Markdown("**演示方法：** 先检测正常样例，再检测局部异常样例，比较热图、异常面积与触发规则；低质量图片应进入复核或无法判定。结果只针对本次选择的模型和配置。")

        def run(image, model_id):
            return inspect_for_ui(image, model_id, project_root)

        def refresh_models(current):
            new_cards = _model_cards(project_root)
            chosen = current if current in new_cards else _default_model(new_cards)
            return gr.Dropdown(choices=list(new_cards), value=chosen), _model_info(chosen, project_root)

        def load_sample(label, current_model):
            path = samples.get(label)
            if path is None:
                return None, gr.skip(), gr.skip()
            with Image.open(path) as picture:
                loaded = picture.convert("RGB").copy()
            available = _model_cards(project_root)
            preference = ("synthetic_pixel",) if path.parent.name == "demo" else ("bottle_patch", "bottle_pixel")
            chosen = next((name for name in preference if name in available), current_model)
            return loaded, gr.Dropdown(choices=list(available), value=chosen), _model_info(chosen, project_root)

        detect.click(run, inputs=[input_image, model], outputs=[output_image, decision, metrics, reasons, trace, full_report, json_download, csv_download], api_name="inspect", concurrency_limit=1)
        model.change(lambda name: _model_info(name, project_root), inputs=model, outputs=model_info, queue=False)
        refresh.click(refresh_models, inputs=model, outputs=[model, model_info], queue=False, api_name=False)
        sample.change(load_sample, inputs=[sample, model], outputs=[input_image, model, model_info], queue=False, api_name=False)
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
    create_app().launch(server_name=args.host, server_port=args.port, share=False, inbrowser=False, show_error=False, show_api=False, max_file_size="20mb", enable_monitoring=False, ssr_mode=False)


if __name__ == "__main__":
    main()
