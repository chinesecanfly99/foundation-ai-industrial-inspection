"""Deterministic, inspectable policy, separate from visual prediction."""
import math

LABELS = {"pass": "Pass", "review": "Manual review", "reject": "Reject recommendation", "undetermined": "Cannot determine"}


def decide(facts: dict, policy: dict) -> dict:
    """Return the first matched rule and the entire evaluated priority trace.

    These rules encode a teaching policy, not validated manufacturing standards.
    critical_overlap_ratio is the fraction of predicted anomalous pixels in ROI.
    """
    required = ("score", "image_threshold", "anomaly_area_ratio", "critical_overlap_ratio", "brightness", "sharpness")
    if any(k not in facts or not math.isfinite(float(facts[k])) for k in required):
        raise ValueError("Rule facts must include every required field and contain finite numbers.")
    if facts["image_threshold"] <= 0:
        raise ValueError("Anomaly threshold must be positive.")
    ratio = facts["score"] / facts["image_threshold"]
    area = facts["anomaly_area_ratio"]
    quality_bad = not policy["min_brightness"] <= facts["brightness"] <= policy["max_brightness"] or facts["sharpness"] < policy["min_sharpness"]
    candidates = [
        ("R1", quality_bad, "undetermined", "Image brightness or sharpness does not meet the input requirements. Retake the image."),
        ("R2", ratio < 1 - policy["review_margin"] and area < policy["min_anomaly_area"], "pass", "The score is below the review band and the predicted anomalous area is small."),
        ("R3", ratio <= 1 + policy["review_margin"], "review", "The score is close to the threshold or a suspicious local region is present. Manual review is recommended."),
        ("R4", area >= policy["reject_area"] or (area >= policy["min_anomaly_area"] and facts["critical_overlap_ratio"] >= policy["critical_overlap"]), "reject", "A clear anomaly intersects the demonstration critical region or exceeds the area limit."),
        ("R5", True, "review", "The anomaly evidence is insufficient for automatic acceptance or rejection. Manual review is recommended."),
    ]
    trace, chosen = [], None
    for rule_id, matched, decision, reason in candidates:
        selected = chosen is None and bool(matched)
        trace.append({"rule": rule_id, "condition": bool(matched), "selected": selected, "suggestion": LABELS[decision], "reason": reason})
        if selected:
            chosen = (decision, reason)
    decision, reason = chosen
    return {"decision": decision, "label_en": LABELS[decision], "reasons": [reason], "trace": trace, "policy_version": policy["version"]}
