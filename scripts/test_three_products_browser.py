"""Browser acceptance of product isolation, current-image reports and recovery."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import re
import traceback

from PIL import Image
from playwright.sync_api import expect, sync_playwright, TimeoutError as BrowserTimeout
from industrial_inspection.catalog import PRODUCTS, REAL_PRODUCTS
from test_ui_browser import inspect_case


def choose_product(page, category: str) -> None:
    if category == "synthetic" and not page.locator("#show-synthetic input").is_checked():
        page.get_by_role("button", name=re.compile("高级 · 开发流程检查")).click()
        with page.expect_response(lambda response: response.request.method == "POST" and "/queue/join" in response.url) as joined:
            page.locator("#show-synthetic input").check()
        joined.value.finished()
        expect(page.locator("#product-choice input")).to_be_enabled()
    control = page.locator("#product-choice input")
    label = PRODUCTS[category]["label"]
    if control.input_value() != label:
        control.click()
        control.fill(label)
        page.get_by_role("option", name=label, exact=True).click()
    expect(control).to_have_value(label)
    expect(page.locator("#model-description")).to_contain_text(PRODUCTS[category]["default_model"])
    expect(page.locator("#inspect-button")).to_be_enabled()
    expect(page.locator("#download-json")).to_be_disabled()


def choose_model(page, category: str, method: str) -> str:
    label = ("合成垫圈流程测试模型" if category == "synthetic" else
             f"特征记忆主模型 · {PRODUCTS[category]['label']}（ResNet18）" if method == "patch" else
             f"像素对比基线 · {PRODUCTS[category]['label']}")
    control = page.locator("#model-choice input")
    if control.input_value() != label:
        control.click()
        control.fill(label)
        page.get_by_role("option", name=label, exact=True).click()
    expect(control).to_have_value(label)
    model_id = "synthetic_pixel" if category == "synthetic" else f"{category}_{method}"
    expect(page.locator("#model-description")).to_contain_text(model_id)
    return model_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--base-url", default="http://127.0.0.1:7861")
    parser.add_argument("--browser-executable", default=None)
    parser.add_argument("--timeout-ms", type=int, default=60_000)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--products", nargs="+", choices=REAL_PRODUCTS, default=list(REAL_PRODUCTS))
    parser.add_argument("--skip-synthetic", action="store_true")
    parser.add_argument("--include-bottle", action="store_true", help="Compatibility flag; real products are now tested by default")
    parser.add_argument("--model", default=None, help="Legacy option retained; the acceptance run now checks each product's primary and baseline models")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve() if args.output else root / "results/three_products/ui"
    output.mkdir(parents=True, exist_ok=True)
    evidence = {"started_at": datetime.now(timezone(timedelta(hours=8))).isoformat(), "url": args.base_url,
                "passed": False, "checks": [], "cases": {}, "page_errors": [], "http_errors": [],
                "request_failures": [], "console_errors": [], "scope": "Functional browser acceptance; detector performance is evaluated separately on all held-out images."}
    browser, page, playwright = None, None, None
    try:
        playwright = sync_playwright().start()
        launch = {"headless": True}
        if args.browser_executable:
            launch["executable_path"] = args.browser_executable
        browser = playwright.chromium.launch(**launch)
        context = browser.new_context(viewport={"width": 1280, "height": 960}, accept_downloads=True)
        page = context.new_page()
        page.set_default_timeout(args.timeout_ms)
        page.on("pageerror", lambda exc: evidence["page_errors"].append(str(exc)))
        page.on("console", lambda msg: evidence["console_errors"].append(msg.text) if msg.type == "error" else None)
        page.on("response", lambda response: evidence["http_errors"].append({"status": response.status, "url": response.url}) if response.status >= 400 else None)
        page.on("requestfailed", lambda request: evidence["request_failures"].append({"url": request.url, "failure": request.failure}))
        page.goto(args.base_url, wait_until="domcontentloaded")
        try:
            page.wait_for_load_state("networkidle", timeout=3000)
        except BrowserTimeout:
            # Workflow State intentionally maintains a long-poll heartbeat.
            page.locator("#inspect-button").wait_for(state="visible")
        page.screenshot(path=str(output / "initial.png"), full_page=True)
        (output / "dom_recon.json").write_text(json.dumps({"buttons": page.get_by_role("button").all_text_contents(),
            "inputs": page.locator("input").evaluate_all("els => els.map(el => ({type:el.type,value:el.value}))")}, ensure_ascii=False, indent=2), encoding="utf-8")
        assert page.locator("#product-choice input").count() == 1
        page.locator("#product-choice input").click()
        initial_products = [label.replace("✓", "").strip() for label in page.get_by_role("option").all_text_contents()]
        assert initial_products == [PRODUCTS[category]["label"] for category in REAL_PRODUCTS], initial_products
        page.locator("#product-choice input").press("Escape")
        evidence["checks"].append("Three real product choices; synthetic development examples hidden initially")
        page.get_by_role("button", name=re.compile("3 · 查看逐条规则与完整记录")).click()
        page.locator("#inspection-json").wait_for(state="visible")

        # Expected errors must finish the status and unlock controls.
        page.locator("#inspection-input").get_by_role("button", name="Remove Image", exact=True).click()
        page.get_by_role("button", name="开始检测", exact=True).click()
        expect(page.locator("#quality-decision")).to_contain_text("检测未完成")
        expect(page.locator("#inspect-button")).to_be_enabled()
        expect(page.locator("#download-json")).to_be_disabled()
        evidence["checks"].append("Empty input error finishes status, restores controls and leaves downloads disabled")

        # Click detection immediately after a product choice. It must wait for
        # the new selection or reject an incomplete snapshot, never expose an
        # old product's downloadable report against the new product label.
        product_control = page.locator("#product-choice input")
        product_control.click()
        page.get_by_role("option", name=PRODUCTS["metal_nut"]["label"], exact=True).click()
        page.get_by_role("button", name="开始检测", exact=True).click()
        expect(page.locator("#model-description")).to_contain_text("metal_nut_patch")
        expect(page.locator("#inspect-button")).to_be_enabled()
        if page.locator("#download-json").is_enabled():
            with page.expect_download() as quick_download:
                page.locator("#download-json").click()
            quick_path = output / "quick_switch_report.json"
            quick_download.value.save_as(str(quick_path))
            current = json.loads(quick_path.read_text(encoding="utf-8"))
            assert current["product_category"] == "metal_nut" and current["model_id"] == "metal_nut_patch"
        evidence["checks"].append("Immediate detection after product switch never exposes a previous product report")
        choose_product(page, "bottle")

        def run_case(category: str, method: str, image: Path, name: str) -> dict:
            model_id = choose_model(page, category, method)
            report = inspect_case(page, image, name, model_id, output, args.timeout_ms)
            assert report["product_category"] == category
            assert report["policy_name"] == Path(PRODUCTS[category]["policy_file"]).name
            assert report["product_label"] == PRODUCTS[category]["label"]
            expect(page.locator("#inspect-button")).to_be_enabled()
            evidence["cases"][name] = {key: report[key] for key in ("product_category", "model_id", "policy_name", "decision", "score", "image_threshold", "image_sha256")}
            return report

        for category in args.products:
            choose_product(page, category)
            page.locator("#sample-choice input").click()
            sample_options = page.get_by_role("option").all_text_contents()
            count = len(list((root / "samples" / category).glob("*.png")))
            assert len(sample_options) == count and count >= 20
            assert any("正常" in label for label in sample_options)
            page.locator("#sample-choice input").press("Escape")
            evidence["checks"].append(f"{category}: {count} Chinese-labeled real samples and correct product-specific model choices")
            normal = root / "samples" / category / "good_000.png"
            defect = sorted(path for path in (root / "samples" / category).glob("*.png") if not path.name.startswith("good_"))[0]
            for case, image in (("normal", normal), ("defect", defect)):
                report = run_case(category, "patch", image, f"{category}_{case}")
                defect_type, original_id = image.stem.rsplit("_", 1)
                predictions_path = root / "results/three_products" / category / "patch/predictions.csv"
                with predictions_path.open(encoding="utf-8", newline="") as handle:
                    expected = next(row for row in csv.DictReader(handle) if row["image_path"] == f"{category}/test/{defect_type}/{original_id}.png")
                assert math.isclose(report["score"], float(expected["score"]), rel_tol=1e-5, abs_tol=1e-6)
                if category == "tile" and case == "defect":
                    page.screenshot(path=str(output / "preview.png"), full_page=False)
            evidence["checks"].append(f"{category}: primary normal/defect uploads, decoded heatmaps, matching image-specific JSON and benchmark scores")
            baseline = run_case(category, "pixel", defect, f"{category}_baseline")
            with page.expect_download() as download:
                page.locator("#download-csv").click()
            csv_path = output / f"{category}_summary.csv"
            download.value.save_as(str(csv_path))
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            assert len(rows) == 1 and rows[0]["image_sha256"] == baseline["image_sha256"]
            assert rows[0]["product_category"] == category and rows[0]["model_id"] == f"{category}_pixel"
            assert "命中并采用" in page.locator("#rule-trace").inner_text()
            evidence["checks"].append(f"{category}: baseline comparison, selected rule and one-row CSV match current image/model")
            page.screenshot(path=str(output / f"{category}_desktop.png"), full_page=True)

        # Verify rejection/recovery after a successful report, not only an empty page.
        tiny = output / "tiny.png"
        Image.new("RGB", (16, 16), "gray").save(tiny)
        page.locator("#inspection-input input[type=file]").set_input_files(str(tiny))
        expect(page.locator("#download-json")).to_be_disabled()
        page.get_by_role("button", name="开始检测", exact=True).click()
        expect(page.locator("#quality-decision")).to_contain_text("检测未完成")
        expect(page.locator("#inspect-button")).to_be_enabled()
        expect(page.locator("#download-json")).to_be_disabled()
        evidence["checks"].append("Tiny image after successful inspection clears old result and recovers controls")
        black = output / "black.png"
        Image.new("RGB", (256, 256), "black").save(black)
        active = args.products[-1]
        report = run_case(active, "patch", black, "black")
        assert report["decision"] == "undetermined"
        evidence["checks"].append("Black image reaches input-quality undetermined decision")
        if not args.skip_synthetic:
            choose_product(page, "synthetic")
            good = run_case("synthetic", "pixel", root / "samples/demo/synthetic_good_0.png", "synthetic_normal")
            bad = run_case("synthetic", "pixel", root / "samples/demo/synthetic_scratch_0.png", "synthetic_scratch")
            assert good["decision"] == "pass" and bad["decision"] in {"review", "reject"}
            assert bad["score"] > good["score"]
            evidence["checks"].append("Advanced synthetic tests use their separate model and policy")
            page.locator("#show-synthetic input").uncheck()
            expect(page.locator("#product-choice input")).to_have_value("瓶口")
            expect(page.locator("#download-json")).to_be_disabled()
            evidence["checks"].append("Hiding development examples returns to a real product and clears report")
        page.set_viewport_size({"width": 390, "height": 844})
        page.evaluate("() => new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))")
        page.wait_for_function("document.documentElement.scrollWidth <= innerWidth + 1")
        page.screenshot(path=str(output / "mobile.png"), full_page=True)
        evidence["checks"].append("390px mobile view has no whole-page horizontal overflow")
        assert not evidence["page_errors"], evidence["page_errors"]
        known = [item for item in evidence["http_errors"] if item["status"] == 404 and "/upload_progress?upload_id=undefined" in item["url"]]
        assert all(item in known for item in evidence["http_errors"]), evidence["http_errors"]
        allowed = {"Failed to load resource: the server responded with a status of 404 (Not Found)", "Response"} if known else set()
        assert all(item in allowed for item in evidence["console_errors"]), evidence["console_errors"]
        assert all("/queue/data?" in item["url"] and item["failure"] == "net::ERR_ABORTED" for item in evidence["request_failures"]), evidence["request_failures"]
        evidence["known_nonblocking_warnings"] = known
        evidence["passed"] = True
        browser.close()
        browser = None
    except Exception as exc:
        evidence["error"] = str(exc)
        evidence["traceback"] = traceback.format_exc()
        if page is not None:
            try:
                page.screenshot(path=str(output / "failure.png"), full_page=True)
                (output / "failure.html").write_text(page.content(), encoding="utf-8")
            except Exception:
                pass
    finally:
        if browser is not None:
            browser.close()
        if playwright is not None:
            playwright.stop()
        evidence["finished_at"] = datetime.now(timezone(timedelta(hours=8))).isoformat()
        (output / "ui_smoke.json").write_text(json.dumps(evidence, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"passed": evidence["passed"], "checks": len(evidence["checks"]), "error": evidence.get("error")}, ensure_ascii=False))
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
