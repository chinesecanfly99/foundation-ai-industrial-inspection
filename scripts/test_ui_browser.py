"""Reusable upload/report checks and the three-product browser test entry point."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback

from PIL import Image, ImageOps
from playwright.sync_api import sync_playwright


def choose_model(page, model_id: str) -> None:
    """Compatibility helper for callers that still use internal model IDs."""
    from industrial_inspection.catalog import PRODUCTS
    from test_three_products_browser import choose_product, choose_model as select_model
    category, method = ("synthetic", "pixel") if model_id == "synthetic_pixel" else model_id.rsplit("_", 1)
    if page.locator("#product-choice input").input_value() != PRODUCTS[category]["label"]:
        choose_product(page, category)
    select_model(page, category, method)


def validate_report(report: dict, model_id: str) -> None:
    required = {
        "decision", "label_zh", "score", "image_threshold",
        "anomaly_area_ratio", "critical_overlap_ratio", "reasons", "trace",
        "model_id", "image_sha256", "policy_sha256",
    }
    assert required <= report.keys(), f"Report missing {required - report.keys()}"
    assert report["decision"] in {"pass", "review", "reject", "undetermined"}
    assert report["model_id"] == model_id
    assert isinstance(report["reasons"], list) and report["reasons"]
    assert isinstance(report["trace"], list) and report["trace"]
    assert sum(bool(row.get("selected")) for row in report["trace"]) == 1
    assert "mask" not in report and "heatmap" not in report
    for field in ("score", "image_threshold", "anomaly_area_ratio", "critical_overlap_ratio"):
        assert isinstance(report[field], (float, int)) and math.isfinite(report[field])
    assert report["image_threshold"] > 0
    assert 0 <= report["anomaly_area_ratio"] <= 1
    assert 0 <= report["critical_overlap_ratio"] <= 1


def inspect_case(page, image_path: Path, case_name: str, model_id: str, results: Path, timeout: int) -> dict:
    with Image.open(image_path) as source:
        working = ImageOps.exif_transpose(source).convert("RGB")
        working.thumbnail((1024, 1024))
        expected_hash = hashlib.sha256(working.tobytes()).hexdigest()
    upload = page.locator("#inspection-input input[type=file]")
    if upload.count() != 1:
        raise AssertionError("Expected one file upload input after DOM reconnaissance")
    upload.set_input_files(str(image_path))
    page.locator("#inspection-input img").wait_for(state="visible", timeout=timeout)
    page.wait_for_function("() => { const image = document.querySelector('#inspection-input img'); return image?.complete && image.naturalWidth > 0; }", timeout=timeout)
    page.get_by_role("button", name="开始检测", exact=True).click()
    page.wait_for_function(
        "hash => { const next = document.querySelector('#inspection-json')?.textContent || ''; return next.includes(hash) && !document.querySelector('#download-json').disabled; }",
        arg=expected_hash,
        timeout=timeout,
    )
    page.locator("#inspection-overlay img").wait_for(state="visible", timeout=timeout)
    page.wait_for_function("() => { const image = document.querySelector('#inspection-input img'); return image?.complete && image.naturalWidth > 0; }", timeout=timeout)
    page.wait_for_function("() => { const image = document.querySelector('#inspection-overlay img'); return image?.complete && image.naturalWidth > 0; }", timeout=timeout)
    with page.expect_download(timeout=timeout) as download_info:
        page.locator("#download-json").click()
    download = download_info.value
    assert download.failure() is None
    report_path = results / f"ui_{case_name}_report.json"
    download.save_as(str(report_path))
    report = json.loads(report_path.read_text(encoding="utf-8"))
    validate_report(report, model_id)
    assert report["image_sha256"] == expected_hash, "Downloaded report does not belong to the uploaded image"
    page.locator("#quality-decision").get_by_text(report["label_zh"], exact=True).wait_for(state="visible", timeout=timeout)
    assert page.locator("#inspection-metrics").get_by_text("异常分数", exact=True).is_visible()
    page.screenshot(path=str(results / f"ui_{case_name}_desktop.png"), full_page=True)
    return report


def main() -> int:
    from test_three_products_browser import main as run_acceptance
    return run_acceptance()


if __name__ == "__main__":
    sys.exit(main())
