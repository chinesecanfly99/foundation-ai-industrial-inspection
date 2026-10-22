"""Reproducible browser smoke test against an already running local demo.

Run from the project virtual environment after installing Playwright. A locally
installed Edge or Chromium can be supplied instead of downloading a browser.
This checks synthetic examples only; it is not an industrial model benchmark.
"""

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
    control = page.locator("#model-choice input")
    if control.count() != 1:
        raise AssertionError("Expected one model dropdown input after DOM reconnaissance")
    control.click()
    control.fill(model_id)
    page.get_by_role("option", name=model_id, exact=True).click()
    assert control.input_value() == model_id


def clear_upload(page) -> None:
    """Click the image component's clear button (label depends on the UI locale)."""
    buttons = page.locator("#inspection-input button")
    count = buttons.count()
    assert count >= 1, "the image component did not render a clear button"
    tokens = ("remove image", "clear", "remove", "清除", "清空", "移除", "删除")
    for index in range(count):
        button = buttons.nth(index)
        labels = " ".join(part for part in (
            button.get_attribute("aria-label") or "",
            button.get_attribute("title") or "",
            button.text_content() or "",
        ))
        if any(token in labels.lower() for token in tokens):
            button.click()
            return
    raise AssertionError("no clear button found on the image component")


def validate_report(report: dict, model_id: str) -> None:
    required = {
        "decision", "label_zh", "score", "image_threshold",
        "anomaly_area_ratio", "critical_area_ratio", "critical_overlap_ratio",
        "brightness", "sharpness", "reasons", "trace",
        "model_id", "image_sha256", "policy_sha256",
    }
    assert required <= report.keys(), f"Report missing {required - report.keys()}"
    assert report["decision"] in {"pass", "review", "reject", "undetermined"}
    assert report["model_id"] == model_id
    assert isinstance(report["reasons"], list) and report["reasons"]
    assert isinstance(report["trace"], list) and report["trace"]
    assert sum(bool(row.get("selected")) for row in report["trace"]) == 1
    assert "mask" not in report and "heatmap" not in report
    for field in ("score", "image_threshold", "anomaly_area_ratio", "critical_area_ratio",
                  "critical_overlap_ratio", "brightness", "sharpness",
                  "alignment_ncc", "illumination_gain"):
        assert isinstance(report[field], (float, int)) and math.isfinite(report[field])
    assert report["image_threshold"] > 0
    assert 0 <= report["anomaly_area_ratio"] <= 1
    assert 0 <= report["critical_overlap_ratio"] <= 1
    assert 0 <= report["critical_area_ratio"] <= 1
    assert -1 <= report["alignment_ncc"] <= 1
    assert report["illumination_gain"] >= 0
    assert all(row.get("evidence") for row in report["trace"])


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
    parser = argparse.ArgumentParser(description="Test the local Gradio UI with synthetic examples")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--browser-executable", default=os.environ.get("UI_BROWSER_EXECUTABLE"))
    parser.add_argument("--model", default="synthetic_pixel")
    parser.add_argument("--include-bottle", action="store_true", help="Also upload two real MVTec bottle examples using bottle_patch")
    parser.add_argument("--timeout-ms", type=int, default=30_000)
    args = parser.parse_args()
    project_root = args.root.resolve()
    results = project_root / "results"
    results.mkdir(parents=True, exist_ok=True)
    evidence = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_url": args.base_url,
        "project_root": str(project_root),
        "model_id": args.model,
        "scope": "UI functional smoke test, using synthetic examples and optional real examples; no dataset-level performance claim",
        "passed": False,
        "checks": [],
        "browser_console_errors": [],
        "queue_responses": [],
        "request_failures": [],
        "http_errors": [],
    }
    browser = None
    playwright = None
    page = None
    try:
        good_path = project_root / "samples" / "demo" / "synthetic_good_0.png"
        anomaly_path = project_root / "samples" / "demo" / "synthetic_scratch_0.png"
        for path in (good_path, anomaly_path):
            assert path.is_file(), f"Missing example: {path}"
        black_path = results / "ui_black_input.png"
        Image.new("RGB", (256, 256), (0, 0, 0)).save(black_path)
        playwright = sync_playwright().start()
        launch = {"headless": True}
        if args.browser_executable:
            launch["executable_path"] = args.browser_executable
        browser = playwright.chromium.launch(**launch)
        evidence["browser_version"] = browser.version
        context = browser.new_context(viewport={"width": 1280, "height": 900}, accept_downloads=True)
        page = context.new_page()
        page.set_default_timeout(args.timeout_ms)
        page_errors = []
        page.on("pageerror", lambda error: page_errors.append(str(error)))
        page.on("console", lambda message: evidence["browser_console_errors"].append(message.text) if message.type == "error" else None)
        page.on("response", lambda response: evidence["queue_responses"].append({"status": response.status, "url": response.url}) if "/queue/" in response.url else None)
        page.on("response", lambda response: evidence["http_errors"].append({"status": response.status, "url": response.url}) if response.status >= 400 else None)
        page.on("requestfailed", lambda request: evidence["request_failures"].append({"url": request.url, "failure": request.failure}))
        page.goto(args.base_url, wait_until="networkidle", timeout=args.timeout_ms)
        # Inspect actual rendered state before using selectors for actions.
        page.screenshot(path=str(results / "ui_initial_desktop.png"), full_page=True)
        reconnaissance = {
            "buttons": page.get_by_role("button").all_text_contents(),
            "inputs": page.locator("input").evaluate_all(
                "els => els.map(el => ({type: el.type, placeholder: el.placeholder, value: el.value}))"
            ),
            "component_ids": page.locator("[id]").evaluate_all(
                "els => els.map(el => el.id).filter(id => id.startsWith('inspection-') || id.includes('choice') || id.includes('download') || id.includes('quality'))"
            ),
        }
        (results / "ui_dom_recon.json").write_text(json.dumps(reconnaissance, ensure_ascii=False, indent=2), encoding="utf-8")
        evidence["checks"].append("Rendered DOM reconnaissance and initial desktop screenshot")
        assert "工业异常定位与可解释质量决策" in page.locator("body").inner_text()
        assert page.get_by_role("button", name="开始检测", exact=True).count() == 1

        # An empty upload must fail visibly without producing a report.
        page.get_by_role("button", name="开始检测", exact=True).click()
        page.get_by_text("请先上传图片，或从演示样例中选择一张。", exact=True).wait_for(state="visible", timeout=10_000)
        assert page.locator("#download-json").is_disabled()
        evidence["checks"].append("Empty image: friendly error and no download")
        # Loading a demo sample must populate the image and invalidate any old result.
        sample = page.locator("#sample-choice input")
        assert sample.count() == 1
        sample.click()
        sample.fill("合成联调 / synthetic_good_0")
        page.get_by_role("option", name="合成联调 / synthetic_good_0", exact=True).click()
        page.locator("#inspection-input img").wait_for(state="visible", timeout=args.timeout_ms)
        page.get_by_text("等待检测", exact=True).wait_for(state="visible", timeout=args.timeout_ms)
        assert page.locator("#download-json").is_disabled()
        evidence["checks"].append("Demo sample loads an image and invalidates previous results")

        choose_model(page, args.model)
        page.get_by_role("button", name="3 · 查看逐条规则与完整记录").click()
        page.locator("#inspection-json").wait_for(state="visible")
        good = inspect_case(page, good_path, "normal", args.model, results, args.timeout_ms)
        assert good["decision"] in {"pass", "review"}, f"Unexpected normal decision: {good['decision']}"
        evidence["checks"].append("Normal PNG upload, heatmap, decision and valid JSON download")

        anomaly = inspect_case(page, anomaly_path, "anomaly", args.model, results, args.timeout_ms)
        assert anomaly["decision"] in {"review", "reject"}, f"Unexpected anomaly decision: {anomaly['decision']}"
        assert anomaly["score"] > good["score"], "Anomalous synthetic example should score above normal example"
        assert anomaly["image_sha256"] != good["image_sha256"]
        evidence["checks"].append("Anomalous PNG upload, updated report and higher anomaly score")
        with page.expect_download(timeout=args.timeout_ms) as csv_download:
            page.locator("#download-csv").click()
        csv_path = results / "ui_anomaly_summary.csv"
        csv_download.value.save_as(str(csv_path))
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            csv_rows = list(csv.DictReader(handle))
        assert len(csv_rows) == 1 and csv_rows[0]["image_sha256"] == anomaly["image_sha256"]
        evidence["checks"].append("CSV download matches the latest inspected image")

        page.locator("#rule-trace").wait_for(state="visible")
        assert "R1" in page.locator("#rule-trace").inner_text()
        assert "命中并采用" in page.locator("#rule-trace").inner_text()
        page.screenshot(path=str(results / "ui_rules_desktop.png"), full_page=True)
        evidence["checks"].append("Rule accordion opens and shows selected rule")

        black = inspect_case(page, black_path, "black", args.model, results, args.timeout_ms)
        # A fully black image fails the input gate; when the anomaly area already forms strong reject evidence, the evidence-priority rule rejects it directly.
        assert black["decision"] in {"undetermined", "reject"}, black["decision"]
        evidence["checks"].append(f"Black image routes to {black['decision']} (input gate fails, evidence priority)")
        reports = {"normal": good, "anomaly": anomaly, "black": black}

        # After clicking the image clear button, no old heatmap, decision, rule or download may remain.
        clear_upload(page)
        page.get_by_text("等待检测", exact=True).wait_for(state="visible", timeout=args.timeout_ms)
        assert page.locator("#download-json").is_disabled()
        assert page.locator("#download-csv").is_disabled()
        assert not page.locator("#inspection-overlay img").count()
        page.wait_for_function("() => !document.querySelector('#rule-trace')?.textContent.includes('命中并采用')", timeout=args.timeout_ms)
        report_text = page.locator("#inspection-json").inner_text()
        assert black["image_sha256"] not in report_text, report_text[:200]
        evidence["checks"].append("Clearing the uploaded image removes heatmap, decision, rules and downloads")

        # Changing the model must invalidate the previous image, decision, and downloads.
        alternate = "bottle_pixel" if (project_root / "models" / "bottle_pixel" / "model_card.json").is_file() else None
        if alternate:
            choose_model(page, alternate)
            page.get_by_text("等待检测", exact=True).wait_for(state="visible", timeout=args.timeout_ms)
            assert page.locator("#download-json").is_disabled()
            assert page.locator("#download-csv").is_disabled()
            assert not page.locator("#inspection-overlay img").count()
            evidence["checks"].append("Changing model invalidates previous decision and downloads")
        if args.include_bottle:
            bottle_samples = {
                "bottle_good": project_root / "samples" / "bottle" / "good_000.png",
                "bottle_broken": project_root / "samples" / "bottle" / "broken_large_000.png",
            }
            assert (project_root / "models" / "bottle_patch" / "model_card.json").is_file()
            choose_model(page, "bottle_patch")
            for name, path in bottle_samples.items():
                assert path.is_file(), f"Missing real bottle example: {path}"
                reports[name] = inspect_case(page, path, name, "bottle_patch", results, args.timeout_ms)
            assert reports["bottle_good"]["image_sha256"] != reports["bottle_broken"]["image_sha256"]
            evidence["checks"].append("Real MVTec good/broken bottle uploads with bottle_patch, heatmaps and JSON downloads")

        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        page.wait_for_function("document.documentElement.scrollWidth <= innerWidth + 1", timeout=5000)
        page.screenshot(path=str(results / "ui_mobile.png"), full_page=True)
        dimensions = page.evaluate("({width: innerWidth, scrollWidth: document.documentElement.scrollWidth})")
        assert dimensions["scrollWidth"] <= dimensions["width"] + 1, f"Mobile page overflows: {dimensions}"
        evidence["checks"].append("390px mobile layout without whole-page horizontal overflow")
        evidence["case_summaries"] = {
            key: {field: report[field] for field in ("decision", "score", "image_threshold", "anomaly_area_ratio", "critical_overlap_ratio")}
            for key, report in reports.items()
        }
        evidence["page_errors"] = page_errors
        assert not page_errors, f"Browser JavaScript errors: {page_errors}"
        known_progress_errors = [item for item in evidence["http_errors"] if item["status"] == 404 and "/upload_progress?upload_id=undefined" in item["url"]]
        unexpected_http_errors = [item for item in evidence["http_errors"] if item not in known_progress_errors]
        assert not unexpected_http_errors, f"Unexpected HTTP resource errors: {unexpected_http_errors}"
        # This Gradio frontend sometimes polls an absent optional progress ID.
        # Preserve that exact warning in evidence; never waive failed images,
        # decoding failures, arbitrary HTTP errors or JavaScript exceptions.
        allowed_console = {"Failed to load resource: the server responded with a status of 404 (Not Found)", "Response"} if known_progress_errors else set()
        unexpected_console = [message for message in evidence["browser_console_errors"] if message not in allowed_console]
        assert not unexpected_console, f"Browser console errors: {unexpected_console}"
        evidence["known_nonblocking_warnings"] = known_progress_errors
        unexpected_request_failures = [item for item in evidence["request_failures"] if not ("/queue/data?" in item["url"] and item["failure"] == "net::ERR_ABORTED")]
        assert not unexpected_request_failures, f"Failed resources: {unexpected_request_failures}"
        evidence["passed"] = True
        browser.close()
        browser = None
    except Exception as exc:
        evidence["error"] = str(exc)
        evidence["traceback"] = traceback.format_exc()
        try:
            if page is not None and not page.is_closed():
                page.screenshot(path=str(results / "ui_failure.png"), full_page=True)
                (results / "ui_failure_dom.html").write_text(page.content(), encoding="utf-8")
                evidence["overflow_elements"] = page.evaluate("() => [...document.querySelectorAll('body,*')].map(el => ({tag:el.tagName,id:el.id,classes:String(el.className).slice(0,80),style:el.getAttribute('style'),right:el.getBoundingClientRect().right,width:el.getBoundingClientRect().width,minWidth:getComputedStyle(el).minWidth})).filter(el => el.right > innerWidth + 1).slice(0,30)")
        except Exception:
            pass
    finally:
        if browser is not None:
            browser.close()
        if playwright is not None:
            playwright.stop()
        evidence["finished_at_utc"] = datetime.now(timezone.utc).isoformat()
        (results / "ui_smoke.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": evidence["passed"], "checks": evidence["checks"], "error": evidence.get("error")}, ensure_ascii=False))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
