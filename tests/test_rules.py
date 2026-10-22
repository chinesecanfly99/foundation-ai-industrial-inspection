import hashlib
import json
from pathlib import Path

import pytest
from PIL import Image

from industrial_inspection.rules import decide, validate_policy

ROOT = Path(__file__).resolve().parents[1]

POLICY = {"version": "test", "critical_roi": [0.2, 0.2, 0.8, 0.8], "min_brightness": 15,"max_brightness": 245,"min_sharpness": 3,"review_margin": .1,"min_anomaly_area": .005,"pass_area_ratio": .015,"pass_critical_area_ratio": .005,"reject_area": .06,"critical_overlap": .2}


@pytest.mark.parametrize("score,area,overlap,critical_area,brightness,expected", [
    (1,0,0,0,0,"undetermined"), (.5,0,0,0,100,"pass"),
    (.5,.008,0,0,100,"pass"), (.5,.002,0,.008,100,"review"),
    (1,.001,0,0,100,"review"), (2,.02,.8,.02,100,"reject"),
    (2,.1,0,0,100,"reject"), (2,.001,0,0,100,"review"),
    (1.05,.08,0,0,100,"reject"), (1.05,.01,.8,.02,100,"reject"),
    (.95,.08,0,0,100,"review")])
def test_decision_priority(score, area, overlap, critical_area, brightness, expected):
    facts = {"score": score,"image_threshold": 1,"anomaly_area_ratio": area,"critical_overlap_ratio": overlap,"critical_area_ratio": critical_area,"brightness": brightness,"sharpness": 100}
    result = decide(facts, POLICY)
    assert result["decision"] == expected
    assert sum(t["selected"] for t in result["trace"]) == 1


def test_bad_facts_are_not_accepted():
    with pytest.raises(ValueError):
        decide({"score":float("nan")}, POLICY)


def test_pixel_model_regressions_do_not_repeat_rule_inversions():
    """Facts are the packaged bottle_pixel demo samples after model-grid measurement."""
    threshold = 4.480851387977601
    cases = [
        # normal image: low score, modest total noise, almost no ROI anomaly
        (3.2014, 0.0113, 0.012, 0.0013, "pass"),
        # small real defect missed by the image score, but concentrated in the ROI
        (3.0738, 0.0033, 0.968, 0.0084, "review"),
        # real defect sitting just inside the review band still needs rejection
        (4.9281, 0.0226, 0.998, 0.0619, "reject"),
    ]
    for score, area, overlap, critical_area, expected in cases:
        facts = {"score": score, "image_threshold": threshold, "anomaly_area_ratio": area,
                     "critical_overlap_ratio": overlap, "critical_area_ratio": critical_area,
                     "brightness": 136.9, "sharpness": 40.0}
        assert decide(facts, POLICY)["decision"] == expected


def test_string_numbers_are_normalised_before_deciding():
    policy = {key: (str(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else value)
              for key, value in POLICY.items()}
    policy["critical_roi"] = [str(value) for value in POLICY["critical_roi"]]
    facts = {"score": "0.5", "image_threshold": "1", "anomaly_area_ratio": "0.002",
             "critical_overlap_ratio": "0", "critical_area_ratio": "0",
             "brightness": "100", "sharpness": "100"}
    result = decide(facts, policy)
    assert result["decision"] == "pass"
    assert policy["min_anomaly_area"] == pytest.approx(0.005)
    assert policy["pass_area_ratio"] == pytest.approx(0.015)
    assert policy["critical_roi"] == [0.2, 0.2, 0.8, 0.8]
    assert validate_policy(dict(POLICY, min_brightness="15"))["min_brightness"] == pytest.approx(15.0)


def test_shipped_policy_and_calibration_resolve_for_packaged_samples():
    """The committed policy, calibration, models and samples must agree."""
    from industrial_inspection.pipeline import inspect_image

    if not (ROOT / "models" / "bottle_patch" / "model.npz").is_file():
        pytest.skip("packaged models are not tracked by git")

    rules = ROOT / "configs" / "quality_rules.json"
    policy = json.loads(rules.read_text(encoding="utf-8"))
    calibration = json.loads((ROOT / "configs" / "calibration" / "bottle.json").read_text(encoding="utf-8"))
    assert calibration["rules_sha256"] == hashlib.sha256(rules.read_bytes()).hexdigest()
    for model_id in ("bottle_patch", "bottle_pixel"):
        recorded = calibration["models"][model_id]
        for sample in ("good_000.png", "contamination_000.png"):
            with Image.open(ROOT / "samples" / "bottle" / sample) as image:
                _, report = inspect_image(image.convert("RGB"), ROOT / "models" / model_id, rules)
            effective = report["effective_policy"]
            assert effective["calibrated"] is True
            assert effective["min_anomaly_area"] == pytest.approx(policy["min_anomaly_area"])
            for key in ("pass_area_ratio", "pass_critical_area_ratio"):
                # Calibration may only tighten the shipped ceiling.
                assert effective[key] == pytest.approx(min(policy[key], recorded[key]))
                assert effective[key] <= policy[key]
            assert report["decision"] in {"pass", "review", "reject", "undetermined"}
            assert len(report["trace"]) == 6


def test_trace_includes_numeric_evidence_for_each_rule():
    facts = {"score": 3.2, "image_threshold": 4.0, "anomaly_area_ratio": .01,
                 "critical_overlap_ratio": .1, "critical_area_ratio": .001,
                 "brightness": 100, "sharpness": 20}
    result = decide(facts, POLICY)
    assert len(result["trace"]) == 6
    assert all(step["evidence"] for step in result["trace"])
    r2 = next(step for step in result["trace"] if step["rule"] == "R2")
    assert "0.800" in r2["evidence"]
    assert "1.000" in r2["evidence"]


def test_malformed_policy_is_rejected_before_deciding():
    facts = {"score": 1, "image_threshold": 1, "anomaly_area_ratio": 0,
                 "critical_overlap_ratio": 0, "critical_area_ratio": 0,
                 "brightness": 100, "sharpness": 100}
    for key, value in (("review_margin", 1.5), ("min_anomaly_area", -0.1),
                       ("reject_area", 0), ("critical_overlap", 1.1),
                       ("min_brightness", 255), ("min_sharpness", -1),
                       ("pass_area_ratio", 0.5), ("pass_critical_area_ratio", 0.5)):
        broken = dict(POLICY)
        broken[key] = value
        with pytest.raises(ValueError, match=key):
            decide(facts, broken)
    missing = {key: value for key, value in POLICY.items() if key != "reject_area"}
    with pytest.raises(ValueError, match="reject_area"):
        decide(facts, missing)
    invalid_roi = dict(POLICY, critical_roi=[0.8, 0.2, 0.2, 0.8])
    with pytest.raises(ValueError, match="关键区域"):
        decide(facts, invalid_roi)


GATED_POLICY = dict(POLICY, input_gate={"enabled": True, "max_aspect_deviation": 0.15, "resolution_tolerance": 0.25},
                    alignment_min=0.96, illumination_min=0.95, illumination_max=1.05,
                    expected_long_edge=900.0, resolution_tolerance=0.25)


def _gate_facts(**overrides):
    facts = {"score": 3.2, "image_threshold": 4.0, "anomaly_area_ratio": .01,
                 "critical_overlap_ratio": .1, "critical_area_ratio": .001,
                 "brightness": 100, "sharpness": 20, "alignment_ncc": .99,
                 "illumination_gain": 1.0, "aspect_deviation": 0.0, "resolution_ratio": 1.0}
    facts.update(overrides)
    return facts


@pytest.mark.parametrize("overrides", [
    {"alignment_ncc": .64}, {"illumination_gain": .80},
    {"illumination_gain": 1.08}, {"aspect_deviation": .5},
    {"resolution_ratio": 0.6}, {"resolution_ratio": 1.4}])
def test_input_gate_routes_unreliable_input_to_undetermined(overrides):
    result = decide(_gate_facts(**overrides), GATED_POLICY)
    assert result["decision"] == "undetermined"
    assert next(step for step in result["trace"] if step["selected"])["rule"] == "R0"


def test_input_gate_passes_consistent_input_through_normal_rules():
    result = decide(_gate_facts(), GATED_POLICY)
    assert result["decision"] == "pass"
    assert next(step for step in result["trace"] if step["selected"])["rule"] == "R2"


@pytest.mark.parametrize("overrides", [
    # contamination/006 and /012 shape: 16%-17% area, but the stain lifts the
    # illumination gain above the gate ceiling.
    {"score": 9.4554, "image_threshold": 4.4809, "anomaly_area_ratio": .1588,
     "critical_overlap_ratio": .9454, "critical_area_ratio": .4149, "illumination_gain": 1.0751},
    {"score": 5.7462, "image_threshold": 4.4809, "anomaly_area_ratio": .1656,
     "critical_overlap_ratio": .8939, "critical_area_ratio": .4092, "illumination_gain": 1.0571},
    # Alignment failure and a low-brightness image must not hide a rejectable defect either.
    {"score": 4.5, "alignment_ncc": .64, "anomaly_area_ratio": .02, "critical_overlap_ratio": .8},
    {"score": 4.5, "brightness": 5, "anomaly_area_ratio": .08},
])
def test_decisive_reject_outranks_an_input_gate_failure(overrides):
    result = decide(_gate_facts(**overrides), GATED_POLICY)
    assert result["decision"] == "reject"
    selected = next(step for step in result["trace"] if step["selected"])
    assert selected["rule"] == "R4"
    assert "拒收" in selected["evidence"] and "门控" in selected["evidence"]
    failed = next(step for step in result["trace"] if step["rule"] in ("R0", "R1") and step["condition"])
    assert failed["selected"] is False
    assert "拒收证据已成立" in failed["evidence"]


def test_input_gate_still_routes_a_review_case_to_undetermined():
    # The gate failure wins when the defect evidence is not yet rejectable.
    result = decide(_gate_facts(score=3.2, anomaly_area_ratio=.002, illumination_gain=1.075), GATED_POLICY)
    assert result["decision"] == "undetermined"
    assert next(step for step in result["trace"] if step["selected"])["rule"] == "R0"


def test_gate_thresholds_may_live_inside_input_gate():
    policy = dict(POLICY)
    policy["input_gate"] = {
        "enabled": True, "max_aspect_deviation": 0.15, "alignment_min": 0.96,
        "illumination_min": 0.95, "illumination_max": 1.05,
        "expected_long_edge": 900.0, "resolution_tolerance": 0.25,
    }
    assert decide(_gate_facts(), policy)["decision"] == "pass"
    assert decide(_gate_facts(alignment_ncc=0.5), policy)["decision"] == "undetermined"


def test_policy_version_is_required():
    missing = {key: value for key, value in POLICY.items() if key != "version"}
    with pytest.raises(ValueError, match="version"):
        decide(_gate_facts(), missing)
    with pytest.raises(ValueError, match="version"):
        decide(_gate_facts(), dict(POLICY, version="  "))
