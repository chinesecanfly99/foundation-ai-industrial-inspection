"""Regression checks for the pure functions of the UI layer; only the Gradio component definitions are imported and no browser is started."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    # app/ is not part of the installed package; add the project root manually in the test.
    sys.path.insert(0, str(PROJECT_ROOT))

pytest.importorskip("gradio", reason="the UI tests need gradio")


def _ui():
    from app import app as module

    return module


@pytest.fixture
def project(tmp_path: Path) -> Path:
    """Minimal project copy: one bottle model card plus a calibration file and the policy."""
    (tmp_path / "configs" / "calibration").mkdir(parents=True)
    (tmp_path / "models" / "bottle_demo").mkdir(parents=True)
    (tmp_path / "configs" / "quality_rules.json").write_text(json.dumps({
        "version": "test",
        "input_gate": {"enabled": True, "max_aspect_deviation": 0.15, "resolution_tolerance": 0.25},
        "calibration": {"bottle": "calibration/bottle.json"},
    }), encoding="utf-8")
    (tmp_path / "configs" / "calibration" / "bottle.json").write_text(
        json.dumps({"schema_version": 1, "category": "bottle", "input_gate": {"long_edge_median": 900.0}}),
        encoding="utf-8")
    (tmp_path / "models" / "bottle_demo" / "model_card.json").write_text(
        json.dumps({"dataset_source": "mvtec_ad/bottle"}), encoding="utf-8")
    return tmp_path


def test_resolution_hint_reports_the_calibrated_long_edge_range(project: Path):
    ui = _ui()
    assert ui._resolution_bounds("bottle_demo", project) == (675, 1125)
    hint = ui._resolution_hint("bottle_demo", project)
    assert "675–1125" in hint
    assert "无法判定" in hint


def test_resolution_hint_explains_missing_calibration_without_crashing(project: Path):
    ui = _ui()
    (project / "configs" / "calibration" / "bottle.json").unlink()
    assert ui._resolution_bounds("bottle_demo", project) is None
    hint = ui._resolution_hint("bottle_demo", project)
    assert "无法判定" in hint
    assert "675" not in hint


def test_oversized_upload_is_refused_with_the_calibrated_range(project: Path):
    ui = _ui()
    overlay, decision, metrics, reasons, trace, report, json_button, csv_button = ui.inspect_for_ui(
        Image.new("RGB", (7000, 100), (10, 10, 10)), "bottle_demo", project)
    assert overlay is None
    assert "检测未完成" in decision
    assert metrics == "" and reasons == "" and trace == []
    assert "675–1125" in report["error"]
    assert "无法判定" in report["error"]
    assert json_button["interactive"] is False and csv_button["interactive"] is False


class UnexpectedPipelineError(Exception):
    """An exception outside the known error list, used to exercise the fallback branch."""


def _failing_pipeline(monkeypatch, error: Exception) -> None:
    from industrial_inspection import pipeline

    def boom(*args, **kwargs):
        raise error

    monkeypatch.setattr(pipeline, "inspect_image", boom)


def test_known_input_error_keeps_the_specific_reason(project: Path, monkeypatch):
    ui = _ui()
    _failing_pipeline(monkeypatch, ValueError("校准文件与模型训练清单不一致"))
    overlay, decision, _, _, trace, report, json_button, csv_button = ui.inspect_for_ui(
        Image.new("RGB", (128, 128), (20, 20, 20)), "bottle_demo", project)
    assert overlay is None and trace == []
    assert "检测未完成" in decision
    assert "校准文件与模型训练清单不一致" in report["error"]
    assert json_button["interactive"] is False and csv_button["interactive"] is False


def test_unexpected_error_also_clears_stale_results_and_is_logged(project: Path, monkeypatch, caplog):
    ui = _ui()
    _failing_pipeline(monkeypatch, UnexpectedPipelineError("内部细节不应展示"))
    caplog.set_level("ERROR", logger="app.app")
    with caplog.at_level("ERROR", logger="app.app"):
        overlay, decision, metrics, reasons, trace, report, json_button, csv_button = ui.inspect_for_ui(
            Image.new("RGB", (128, 128), (20, 20, 20)), "bottle_demo", project)
    assert overlay is None
    assert metrics == "" and reasons == "" and trace == []
    assert "检测未完成" in decision and "未预期的错误" in decision
    # Internal exception text only goes to the log, never to the page or the download records.
    assert "内部细节不应展示" not in decision
    assert "内部细节不应展示" not in report["error"]
    assert json_button["interactive"] is False and csv_button["interactive"] is False
    assert "内部细节不应展示" in caplog.text
