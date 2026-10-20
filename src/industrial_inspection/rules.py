"""Deterministic, inspectable policy, separate from visual prediction."""
import math

LABELS = {"pass": "通过", "review": "人工复核", "reject": "拒收建议", "undetermined": "无法判定"}


def decide(facts: dict, policy: dict) -> dict:
    """Return the first matched rule and the entire evaluated priority trace.

    These rules encode a teaching policy, not validated manufacturing standards.
    critical_overlap_ratio is the fraction of predicted anomalous pixels in ROI.
    """
    required = ("score", "image_threshold", "anomaly_area_ratio", "critical_overlap_ratio", "brightness", "sharpness")
    if any(k not in facts or not math.isfinite(float(facts[k])) for k in required):
        raise ValueError("规则事实必须完整且为有限数值")
    if facts["image_threshold"] <= 0:
        raise ValueError("异常阈值必须为正")
    ratio = facts["score"] / facts["image_threshold"]
    area = facts["anomaly_area_ratio"]
    quality_bad = not policy["min_brightness"] <= facts["brightness"] <= policy["max_brightness"] or facts["sharpness"] < policy["min_sharpness"]
    candidates = [
        ("R1", quality_bad, "undetermined", "图像亮度或清晰度未满足输入要求，请重新拍摄。"),
        ("R2", ratio < 1 - policy["review_margin"] and area < policy["min_anomaly_area"], "pass", "分数低于复核区间，预测异常面积较小。"),
        ("R3", ratio <= 1 + policy["review_margin"], "review", "分数接近阈值或存在局部可疑区域，建议人工复核。"),
        ("R4", area >= policy["reject_area"] or (area >= policy["min_anomaly_area"] and facts["critical_overlap_ratio"] >= policy["critical_overlap"]), "reject", "明显异常涉及示范关键区域或超过面积上限。"),
        ("R5", True, "review", "异常证据不足以自动通过或拒收，建议人工复核。"),
    ]
    trace, chosen = [], None
    for rule_id, matched, decision, reason in candidates:
        selected = chosen is None and bool(matched)
        trace.append({"rule": rule_id, "condition": bool(matched), "selected": selected, "suggestion": LABELS[decision], "reason": reason})
        if selected:
            chosen = (decision, reason)
    decision, reason = chosen
    return {"decision": decision, "label_zh": LABELS[decision], "reasons": [reason], "trace": trace, "policy_version": policy["version"]}
