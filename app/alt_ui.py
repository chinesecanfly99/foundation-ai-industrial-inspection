"""A local-only Gradio interface; model inference is loaded on the first click."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

# This local English edition owns its caches and never reuses the source
# project cache or a globally configured Torch cache.
ROOT = Path(__file__).resolve().parents[1]
os.environ["GRADIO_TEMP_DIR"] = str(ROOT / ".cache" / "gradio")
os.environ["TORCH_HOME"] = str(ROOT / ".cache" / "torch")
# Gradio renames a completed upload into its cache. Keep scratch uploads on
# the same filesystem so first uploads publish atomically before their URLs
# are returned; cross-drive moves otherwise fall back to a background copy.
UPLOAD_TEMP_DIR = ROOT / ".cache" / "upload_tmp"
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)
for variable in ("TMP", "TEMP", "TMPDIR"):
    os.environ[variable] = str(UPLOAD_TEMP_DIR)
tempfile.tempdir = str(UPLOAD_TEMP_DIR)
# Keep local presentation independent of online analytics and downloaded fonts.
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
from PIL import Image, ImageOps
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from industrial_inspection.catalog import PRODUCTS, REAL_PRODUCTS, policy_path_for_product, product_id_from_card
from app.upload_cache import install_windows_upload_cache_fix
from app.localization import locale, localize_report, localized_products, locale_startup_js
from app.workflow_ui import build_workflow_tabs
from industrial_inspection.workflow import InspectionStore

LANGUAGE = 'en'
PRODUCTS = localized_products(LANGUAGE)
TEXT = locale(LANGUAGE)["ui"]


MAX_PIXELS = 16_000_000
MAX_DIMENSION = 6000
DECISIONS = {
    "pass": ("Pass", "pass"),
    "review": ("Manual review", "review"),
    "reject": ("Reject recommendation", "reject"),
    "undetermined": ("Cannot determine", "undetermined"),
}

CSS = """
.gradio-container { width: 100% !important; min-width: 0 !important;
  max-width: 1280px !important; margin: auto; box-sizing: border-box; }
.gradio-container main.app, .gradio-container main.app > .wrap { min-width: 0; }
#rule-trace, #inspection-json { max-width: 100%; min-width: 0 !important;
  overflow: auto !important; }
.project-hero { padding: 26px 30px; border-radius: 20px; background: #31204f;
  color: #f5fbff; margin-bottom: 18px; }
.project-eyebrow { color: #d0bfff; font-size: 12px; font-weight: 700;
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
    return [(("Synthetic washer check model" if product_id == "synthetic" else
              f"Feature memory model · {label} (ResNet18)" if name.endswith("_patch") else
              f"Pixel reference baseline · {label}"), name)
            for name in _product_models(cards, product_id)]


def _model_info(model_id: str | None, project_root: Path) -> str:
    cards = _model_cards(project_root)
    if not model_id or model_id not in cards:
        return "No model was found. Follow the README to prepare data and fit a model, then refresh the model list."
    card = cards[model_id]
    product_id = product_id_from_card(card)
    if product_id is None:
        return "The model card does not identify a supported product category. This model is unavailable in the interface."
    if product_id == "synthetic":
        method = "Pixel comparison on synthetic washers checks uploading, heatmaps, rules and downloads only."
    elif card.get("method") == "patch_memory":
        method = "Primary model: pretrained ResNet18 extracts local features and compares them with a normal-feature memory to localize visual anomalies."
    else:
        method = "Pixel baseline: compares color statistics at aligned image positions. It is more sensitive to position and lighting changes."
    counts = ""
    if isinstance(card.get("training_count"), int) and isinstance(card.get("calibration_count"), int):
        counts = f" Normal training: {card['training_count']} images; normal calibration: {card['calibration_count']} images."
    return f"**Product: {PRODUCTS[product_id]['label']}** · `{model_id}`  \n{method}{counts}  \nUse images of this product under similar capture conditions. The model outputs anomaly locations and scores, not defect names."


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
        choices.append((f"{labels.get(defect, defect)} · Sample {number}", stem))
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
    note = "For software workflow checks only." if product_id == "synthetic" else "Sample names come from dataset annotations, not predicted defect classes."
    return f"{product['description']} Available samples: **{count} images**. {note}"


def _policy_info(product_id: str, project_root: Path) -> str:
    try:
        policy = json.loads(policy_path_for_product(project_root, product_id).read_text(encoding="utf-8"))
        description = str(policy.get("description", "Demonstration policy"))
        description = locale(LANGUAGE)["policies"].get(product_id, description)
    except (OSError, ValueError):
        description = "No matching policy was found. Prepare the selected product category first."
    return f"**Policy: {PRODUCTS[product_id]['label']}** {description}  \nThe yellow outline marks the configured critical region. Thresholds and regions still require independent operational validation."


def _clear_results():
    return (None, '<div class="decision-card"><h2>Ready to inspect</h2><p>The image or model changed. Click Start inspection.</p></div>',
            "", "", [], None, gr.update(value=None, interactive=False), gr.update(value=None, interactive=False))


def _failed_results(message: str):
    cleared = list(_clear_results())
    cleared[1] = ('<div class="decision-card decision-undetermined"><h2>Inspection incomplete</h2>'
                  f'<p>{html.escape(message)}</p></div>')
    return tuple(cleared)


def _load_sample_for_ui(chosen_sample: str | None, chosen_product: str, current_model: str | None, project_root: Path):
    paths = _sample_paths(project_root, chosen_product)
    picture = _load_picture(paths.get(chosen_sample))
    if chosen_sample and picture is None:
        gr.Warning("This sample could not be read. The previous image and report were cleared. Refresh samples or upload an image.")
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
    label, css_name = DECISIONS.get(str(report.get("decision")), ("Cannot determine", "undetermined"))
    if report.get("label_en"):
        label = str(report["label_en"])
    return (
        f'<div class="decision-card decision-{css_name}"><h2>{html.escape(label)}</h2>'
        '<p>This demonstration recommendation uses anomaly location, area and quality rules. Review it alongside the original image and rule trace.</p></div>'
    )


def _metrics_html(report: dict[str, Any]) -> str:
    values = (
        ("Anomaly score", _number(report, "score")),
        ("Image threshold", _number(report, "image_threshold")),
        ("Anomalous area", _number(report, "anomaly_area_ratio", percent=True)),
        ("Anomaly inside critical region", _number(report, "critical_overlap_ratio", percent=True)),
    )
    return '<div class="metrics-grid">' + "".join(
        f'<div class="metric"><span class="label">{label}</span><span class="value">{value}</span></div>'
        for label, value in values
    ) + "</div>"


def _trace_rows(trace: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for index, step in enumerate(trace, 1):
        name = step.get("rule_id", step.get("rule", step.get("id", step.get("name", f"Rule {index}"))))
        outcome = step.get("triggered", step.get("matched", step.get("passed", step.get("result", step.get("condition", "Checked")))))
        if isinstance(outcome, bool):
            outcome = ("Matched and selected" if step.get("selected") else "Matched, not selected") if outcome else "Not matched"
        detail = step.get("reason", step.get("message", step.get("description", step.get("condition", ""))))
        if not detail:
            detail = json.dumps({k: v for k, v in step.items() if k not in {"rule_id", "rule", "id", "name", "triggered", "matched", "passed", "result"}}, ensure_ascii=False)
        rows.append([str(name), str(outcome), str(detail)])
    return rows


def inspect_for_ui(image: Image.Image | None, model_id: str | None, project_root: Path | None = None):
    """Run one inspection and prepare view/download values without changing a model."""
    project_root = (project_root or ROOT).resolve()
    if image is None:
        raise gr.Error("Upload an image or select a demonstration sample first.")
    if not isinstance(image, Image.Image):
        raise gr.Error("The image could not be read. Upload a PNG or JPEG image.")
    if image.width * image.height > MAX_PIXELS or max(image.size) > MAX_DIMENSION:
        raise gr.Error("The image is too large. Limit its longest side to 6000 pixels and its total size to 16 million pixels.")
    cards = _model_cards(project_root)
    if not model_id or model_id not in cards:
        raise gr.Error("No model is available. Fit a model first, then refresh the model list.")
    product_id = product_id_from_card(cards[model_id])
    if product_id is None or model_id not in _product_models(cards, product_id):
        raise gr.Error("The model card does not match a supported product. Select a model for the corresponding product.")
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
        scalar_report["reasons"] = "; ".join(str(item) for item in safe_report.get("reasons", []))
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(scalar_report))
            writer.writeheader()
            writer.writerow(scalar_report)
    except (ValueError, FileNotFoundError, RuntimeError, ImportError, OSError) as exc:
        raise gr.Error(f"Inspection could not be completed: {exc}") from exc
    reasons = safe_report.get("reasons", [])
    reasons_html = '<div class="rule-reasons"><strong>Decision rationale</strong><ul>' + "".join(
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
            raise gr.Error("The product and model are still switching. Wait for loading to finish, then inspect again.")
        if chosen_sample and isinstance(image, Image.Image):
            expected = _load_picture(_sample_paths(project_root, chosen_product).get(chosen_sample))
            if expected is None:
                raise gr.Error("The selected sample has not loaded. Refresh samples, then inspect again.")
            actual = ImageOps.exif_transpose(image).convert("RGB")
            expected.thumbnail((1024, 1024))
            actual.thumbnail((1024, 1024))
            if actual.size != expected.size or actual.tobytes() != expected.tobytes():
                raise gr.Error("The image and selected sample are not yet synchronized. Wait for loading to finish, then inspect again.")
        if isinstance(image, Image.Image):
            image.info["inspection_filename"] = f"{chosen_sample}.png" if chosen_sample else "uploaded_image.png"
        return _run_for_ui(image, model_id, project_root)
    except gr.Error as exc:
        message = str(exc.message)
        gr.Warning(message)
        return _failed_results(message)



def _english_ui_startup_js() -> str:
    # Gradio 5.49.1 has no forced-locale option. Its shared frontend locale
    # store controls built-in upload labels, settings and footer text as well
    # as document.lang. Run after Gradio initializes its browser-based locale.
    assets = Path(gr.__file__).resolve().parent / "templates" / "frontend" / "assets"
    locale_modules = [
        module for module in assets.glob("init-*.js")
        if "fallbackLocale" in module.read_text(encoding="utf-8")
    ]
    if len(locale_modules) != 1:
        raise RuntimeError("Cannot locate the pinned Gradio frontend locale module.")
    locale_url = json.dumps("./assets/" + locale_modules[0].name)
    return """async () => {
        const locale = await import(new URL(LOCALE_URL, document.baseURI).href);
        if (typeof locale.e !== "function") {
            throw new Error("Unsupported Gradio locale adapter; use Gradio 5.49.1.");
        }
        locale.e("en");
    }""".replace("LOCALE_URL", locale_url)


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
    theme = gr.themes.Default(primary_hue="indigo", secondary_hue="violet", neutral_hue="slate", font=["Arial", "sans-serif"], font_mono=["Consolas", "Courier New", "monospace"])
    with gr.Blocks(title="Industrial Inspection - EN LOCAL", theme=theme, css=CSS, js=locale_startup_js(LANGUAGE), analytics_enabled=False, delete_cache=(3600, 86400)) as demo:
        language_toggle = gr.Button(TEXT["switch"], elem_id="language-toggle", size="sm")
        language_toggle.click(fn=None, js="() => { window.location.assign('/'); return []; }", queue=False, api_name=False)
        gr.Markdown(TEXT["switch_note"], elem_id="language-note")
        if (project_root / "EN_LOCAL_ONLY.txt").exists():
            gr.Markdown('This is the separate local-only English project copy.')
        gr.HTML(
            '<div class="project-hero"><div class="project-eyebrow">ENGLISH VIEW · LOCAL INSPECTION</div>'
            '<h1>Industrial Anomaly Localization and Explainable Quality Decisions</h1>'
            '<p>Bottle, metal nut and tile surface: locate anomalies, inspect the rules and review pass, review or reject recommendations.</p></div>'
        )
        gr.HTML(
            '<div class="project-note">Course prototype - each product uses its own models and policy. '
            'Anomaly scores are not probabilities. Heatmaps show visual differences, not physical causes. Use the held-out evaluation to assess detection performance.</div>'
        )
        with gr.Tabs(elem_id="inspection-tabs"):
            with gr.Tab(TEXT["single"], elem_id="single-tab"):
                with gr.Row(equal_height=False):
                    with gr.Column(scale=5, min_width=320):
                        gr.Markdown("### 1 · Select a product, image and model")
                        product = gr.Dropdown(label="Product category", choices=[(PRODUCTS[name]["label"], name) for name in REAL_PRODUCTS], value=product_id, elem_id="product-choice", interactive=True)
                        product_info = gr.Markdown(_product_info(product_id, project_root), elem_id="product-description")
                        input_image = gr.Image(value=initial_image, label="Input image", type="pil", sources=["upload"], height=350, format="png", elem_id="inspection-input", placeholder="Upload an image of the selected product or choose a sample below")
                        sample = gr.Dropdown(label="Demonstration sample", choices=initial_choices, value=initial_sample, info="Names use dataset annotations; different sample numbers refer to different images. Uploading a custom image clears the sample selection.", elem_id="sample-choice", interactive=True)
                        with gr.Row():
                            model = gr.Dropdown(label="Inspection model", choices=_model_choices(cards, product_id), value=selected, scale=4, elem_id="model-choice", interactive=True)
                            refresh = gr.Button("Refresh models and samples", scale=1, min_width=110, elem_id="refresh-models")
                        model_info = gr.Markdown(_model_info(selected, project_root), elem_id="model-description")
                        with gr.Accordion("Advanced · Developer checks", open=False):
                            show_synthetic = gr.Checkbox(label="Show synthetic washer checks", value=False, info="Adds synthetic checks to the product list. These examples are separate from the three real industrial categories.", elem_id="show-synthetic")
                        detect = gr.Button("Start inspection", variant="primary", size="lg", elem_id="inspect-button")
                    with gr.Column(scale=6, min_width=320):
                        gr.Markdown("### 2 · Review anomalies and the recommendation")
                        output_image = gr.Image(label="Anomaly heatmap overlay - warmer colors indicate stronger responses", type="pil", interactive=False, height=350, format="png", elem_id="inspection-overlay")
                        decision = gr.HTML('<div class="decision-card"><h2>Ready to inspect</h2><p>Select an image, then click Start inspection.</p></div>', elem_id="quality-decision")
                        metrics = gr.HTML("", elem_id="inspection-metrics")
                        reasons = gr.HTML("", elem_id="inspection-reasons")
                with gr.Accordion("3 · Rule trace and full report", open=False):
                    policy_info = gr.Markdown(_policy_info(product_id, project_root), elem_id="policy-description")
                    trace = gr.Dataframe(headers=["Rule", "Match / check result", "Explanation"], datatype=["str", "str", "str"], type="array", interactive=False, wrap=True, label="Rule trace", elem_id="rule-trace")
                    full_report = gr.JSON(label="Inspection report", elem_id="inspection-json")
                with gr.Row():
                    json_download = gr.DownloadButton("Download JSON report", interactive=False, elem_id="download-json")
                    csv_download = gr.DownloadButton("Download CSV summary", interactive=False, elem_id="download-csv")
                gr.Markdown("**Reports:** CSV contains one summary row for the current image, suitable for spreadsheet comparisons. JSON contains metrics, product and model details, and the full rule trace. Download the heatmap separately from its image panel.")
                gr.Markdown("**Demonstration:** inspect normal samples and then defect samples for each product. Compare heatmaps, rules, the feature model and the pixel baseline. Poor-quality images should require review or be undetermined. Inspect again after changing the product, image or model.")
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
            cleared[1] = '<div class="decision-card"><h2>Inspecting</h2><p>The heatmap, rules and report will appear when inspection finishes.</p></div>'
            return (*[gr.update(interactive=False) for _ in controls], *cleared)

        def begin_selection():
            cleared = list(_clear_results())
            cleared[1] = '<div class="decision-card"><h2>Loading</h2><p>Updating the product, sample and model. Please wait.</p></div>'
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
