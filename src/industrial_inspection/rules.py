"""Deterministic, inspectable policy, separate from visual prediction."""
import math

LABELS = {"pass": "通过", "review": "人工复核", "reject": "拒收建议", "undetermined": "无法判定"}



REQUIRED_POLICY_FIELDS = (
    "review_margin", "min_anomaly_area", "reject_area", "critical_overlap",
    "min_brightness", "max_brightness", "min_sharpness",
)

NUMERIC_POLICY_FIELDS = (*REQUIRED_POLICY_FIELDS, "pass_area_ratio", "pass_critical_area_ratio")
GATE_NUMERIC_FIELDS = (
    "alignment_min", "illumination_min", "illumination_max",
    "expected_long_edge", "max_aspect_deviation", "resolution_tolerance",
)


def _policy_number(policy: dict, key: str) -> float:
    if key not in policy:
        raise ValueError(f"策略缺少必填字段：{key}")
    try:
        value = float(policy[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"策略字段 {key} 必须是数值") from exc
    if not math.isfinite(value):
        raise ValueError(f"策略字段 {key} 必须是有限数值")
    return value


def _gate_number(policy: dict, gate: dict, key: str) -> float:
    """Read a gate threshold from the resolved policy or its nested defaults."""
    if key in policy:
        return _policy_number(policy, key)
    if key not in gate:
        raise ValueError(f"策略缺少必填字段：{key}")
    try:
        value = float(gate[key])
    except (TypeError, ValueError) as exc:
        raise ValueError(f"策略字段 {key} 必须是数值") from exc
    if not math.isfinite(value):
        raise ValueError(f"策略字段 {key} 必须是有限数值")
    return value


def _validate_input_gate(policy: dict) -> None:
    gate = policy.get("input_gate")
    if gate is None:
        return
    if not isinstance(gate, dict):
        raise ValueError("策略字段 input_gate 必须是对象")  # noqa: TRY004 - callers handle ValueError
    if not gate.get("enabled"):
        return
    for key in ("alignment_min", "illumination_min", "illumination_max", "expected_long_edge", "resolution_tolerance"):
        _gate_number(policy, gate, key)
    aspect = _gate_number(policy, gate, "max_aspect_deviation")
    if not -1 <= _gate_number(policy, gate, "alignment_min") <= 1:
        raise ValueError("策略字段 alignment_min 必须在-1到1之间")
    if not 0 < _gate_number(policy, gate, "illumination_min") <= _gate_number(policy, gate, "illumination_max"):
        raise ValueError("策略光照门槛必须满足 0 < illumination_min ≤ illumination_max")
    if not 0 <= aspect < 1:
        raise ValueError("策略字段 max_aspect_deviation 必须在0到1之间")
    if _gate_number(policy, gate, "expected_long_edge") <= 0:
        raise ValueError("策略字段 expected_long_edge 必须为正")
    if not 0 <= _gate_number(policy, gate, "resolution_tolerance") < 1:
        raise ValueError("策略字段 resolution_tolerance 必须在0到1之间")


def _normalise_policy(policy: dict) -> dict:
    """Write float values back so later comparisons cannot hit str/int surprises."""
    for key in NUMERIC_POLICY_FIELDS:
        if key in policy:
            policy[key] = float(policy[key])
    for key in GATE_NUMERIC_FIELDS:
        if key in policy:
            policy[key] = float(policy[key])
    if "critical_roi" in policy:
        policy["critical_roi"] = [float(value) for value in policy["critical_roi"]]
    gate = policy.get("input_gate")
    if isinstance(gate, dict):
        for key in GATE_NUMERIC_FIELDS:
            if key in gate:
                gate[key] = float(gate[key])
    return policy


def validate_policy(policy: dict) -> dict:
    """Reject a malformed policy and return it with numeric fields as float.

    Values that arrive as strings (JSON edited by hand, UI form input) would
    otherwise raise TypeError in the comparisons inside decide(). The mapping is
    also updated in place so a caller that ignores the return value still gets
    the normalised numbers.
    """
    if not isinstance(policy, dict):
        raise ValueError("策略必须是 JSON 对象")  # noqa: TRY004 - callers handle ValueError
    for key in REQUIRED_POLICY_FIELDS:
        _policy_number(policy, key)
    version = policy.get("version")
    if not isinstance(version, str) or not version.strip():
        raise ValueError("策略字段 version 必须是非空字符串")
    roi = policy.get("critical_roi")
    if not isinstance(roi, (list, tuple)) or len(roi) != 4:
        raise ValueError("策略字段 critical_roi 必须是4个归一化数值")
    x0, y0, x1, y1 = (float(value) for value in roi)
    if not (0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1):
        raise ValueError("关键区域必须为0到1的合法归一化矩形")
    margin = _policy_number(policy, "review_margin")
    if not 0 < margin < 1:
        raise ValueError("策略字段 review_margin 必须在0到1之间")
    min_area, reject_area = _policy_number(policy, "min_anomaly_area"), _policy_number(policy, "reject_area")
    if not 0 <= min_area <= 1 or not 0 < reject_area <= 1:
        raise ValueError("策略面积门槛 min_anomaly_area 必须在0到1之间，且 reject_area 必须大于0")
    if reject_area < min_area:
        raise ValueError("策略字段 reject_area 不能小于 min_anomaly_area")
    for key in ("pass_area_ratio", "pass_critical_area_ratio"):
        value = _policy_number(policy, key) if key in policy else min_area
        if not 0 <= value <= 1:
            raise ValueError(f"策略字段 {key} 必须在0到1之间")
        if value > reject_area:
            raise ValueError(f"策略字段 {key} 不能大于 reject_area，否则大缺陷可能被自动通过")
    overlap = _policy_number(policy, "critical_overlap")
    if not 0 <= overlap <= 1:
        raise ValueError("策略字段 critical_overlap 必须在0到1之间")
    minimum, maximum = _policy_number(policy, "min_brightness"), _policy_number(policy, "max_brightness")
    if not 0 <= minimum < maximum <= 255:
        raise ValueError("策略亮度门槛 min_brightness/max_brightness 必须满足 0 ≤ min < max ≤ 255")
    if _policy_number(policy, "min_sharpness") < 0:
        raise ValueError("策略字段 min_sharpness 不能为负")
    _validate_input_gate(policy)
    return _normalise_policy(policy)


def decide(facts: dict, policy: dict) -> dict:
    """Return the winning rule and the entire evaluated priority trace.

    Rules are listed in priority order R0-R5, with one deliberate exception:
    a decisive reject (R4) is not downgraded by an input-gate or image-quality
    failure (R0/R1), because those rules only remove the ability to certify a
    pass or ask for review.

    These rules encode a teaching policy, not validated manufacturing standards.
    critical_overlap_ratio is the fraction of predicted anomalous pixels in ROI.
    """
    policy = validate_policy(policy)
    required = ["score", "image_threshold", "anomaly_area_ratio", "critical_overlap_ratio", "critical_area_ratio", "brightness", "sharpness"]
    if (policy.get("input_gate") or {}).get("enabled"):
        required += ["alignment_ncc", "illumination_gain", "aspect_deviation", "resolution_ratio"]
    facts = dict(facts)
    for key in required:
        if key not in facts:
            raise ValueError("规则事实必须完整且为有限数值")
        try:
            facts[key] = float(facts[key])
        except (TypeError, ValueError) as exc:
            raise ValueError("规则事实必须完整且为有限数值") from exc
        if not math.isfinite(facts[key]):
            raise ValueError("规则事实必须完整且为有限数值")
    if facts["image_threshold"] <= 0:
        raise ValueError("异常阈值必须为正")
    ratio = facts["score"] / facts["image_threshold"]
    area = facts["anomaly_area_ratio"]
    critical_area = facts["critical_area_ratio"]
    # A permissive total-area limit alone lets a small but concentrated ROI
    # anomaly pass. The second gate keeps that evidence out of the pass rule.
    pass_area = policy.get("pass_area_ratio", policy["min_anomaly_area"])
    pass_critical_area = policy.get("pass_critical_area_ratio", policy["min_anomaly_area"])
    quality_bad = not policy["min_brightness"] <= facts["brightness"] <= policy["max_brightness"] or facts["sharpness"] < policy["min_sharpness"]
    gate_enabled = bool((policy.get("input_gate") or {}).get("enabled"))
    gate_policy = policy.get("input_gate") or {}
    max_aspect_deviation = _gate_number(policy, gate_policy, "max_aspect_deviation") if gate_enabled else 0.15
    resolution_ok = True
    if gate_enabled:
        alignment_min = _gate_number(policy, gate_policy, "alignment_min")
        illumination_min = _gate_number(policy, gate_policy, "illumination_min")
        illumination_max = _gate_number(policy, gate_policy, "illumination_max")
        tolerance = _gate_number(policy, gate_policy, "resolution_tolerance")
        resolution_ok = 1 - tolerance <= facts["resolution_ratio"] <= 1 + tolerance
        gate_bad = (
            facts["alignment_ncc"] < alignment_min
            or not illumination_min <= facts["illumination_gain"] <= illumination_max
            or facts["aspect_deviation"] > max_aspect_deviation
            or not resolution_ok
        )
    else:
        gate_bad = False
    reject_evidence = ratio >= 1 and (
        area >= policy["reject_area"]
        or (area >= policy["min_anomaly_area"] and facts["critical_overlap_ratio"] >= policy["critical_overlap"])
    )
    # An unreliable input cannot certify a pass, but it must not downgrade a
    # defect that is already large enough to reject: a stain can lift the
    # illumination gain while still covering 16% of the image.
    gate_superseded = gate_bad and reject_evidence
    quality_superseded = quality_bad and reject_evidence
    input_superseded = gate_superseded or quality_superseded
    gate_note = "；输入门控/图像质量未通过，但面积或关键区拒收证据已成立，仍判拒收" if input_superseded else ""
    candidates = [
        ("R0", gate_bad, "undetermined", "输入与校准拍摄条件不一致，无法可靠判定。请按固定工位、固定光照与对齐重新拍摄。",
         (f"对齐一致性 {facts['alignment_ncc']:.3f}（≥{alignment_min:.3f}），光照增益 {facts['illumination_gain']:.3f}（{illumination_min:.3f}–{illumination_max:.3f}），宽高偏差 {facts['aspect_deviation']:.3f}（≤{max_aspect_deviation:.3f}），分辨率比 {facts['resolution_ratio']:.3f}（{1-tolerance:.2f}–{1+tolerance:.2f}）" if gate_enabled else "输入门控未启用")
         + ("；但面积/关键区拒收证据已成立，不按门控降级" if gate_superseded else "")),
        ("R1", quality_bad, "undetermined", "图像亮度或清晰度未满足输入要求，请重新拍摄。",
         f"亮度 {facts['brightness']:.1f}（{policy['min_brightness']}–{policy['max_brightness']}），清晰度 {facts['sharpness']:.2f}（≥{policy['min_sharpness']}）"
         + ("；但面积/关键区拒收证据已成立，不按图像质量降级" if quality_superseded else "")),
        ("R2", ratio < 1 - policy["review_margin"] and area < pass_area and critical_area < pass_critical_area, "pass", "分数低于复核区间，异常总量和关键区异常面积均较小。",
         f"分数比 {ratio:.3f}（<{1 - policy['review_margin']:.3f}），异常面积 {area:.3%}（<{pass_area:.3%}），关键区异常面积 {critical_area:.3%}（<{pass_critical_area:.3%}）"),
        # Do not let a score that merely sits inside the review band hide a
        # large or ROI-concentrated anomaly; that evidence takes precedence.
        ("R3", ratio <= 1 + policy["review_margin"] and not reject_evidence, "review", "分数接近阈值或存在局部可疑区域，建议人工复核。",
         f"分数比 {ratio:.3f}（≤{1 + policy['review_margin']:.3f}），面积拒收证据 {str(area >= policy['reject_area']).lower()}，关键区拒收证据 {str(area >= policy['min_anomaly_area'] and facts['critical_overlap_ratio'] >= policy['critical_overlap']).lower()}"),
        ("R4", reject_evidence, "reject", "异常分数达到阈值，且异常面积或关键区域证据足以拒收。"
         + ("输入质量不满足要求，但拒收证据已成立，仍建议拒收。" if input_superseded else ""),
         f"分数比 {ratio:.3f}（≥1），异常面积 {area:.3%}（拒收≥{policy['reject_area']:.3%}），关键区重叠 {facts['critical_overlap_ratio']:.3f}（≥{policy['critical_overlap']} 且面积≥{policy['min_anomaly_area']:.3%}）"
         + gate_note),
        ("R5", True, "review", "异常证据不足以自动通过或拒收，建议人工复核。",
         f"分数比 {ratio:.3f}，异常面积 {area:.3%}，关键区异常面积 {critical_area:.3%}；未形成自动通过或拒收证据"),
    ]
    # A decisive area/ROI reject outranks the input gate: the gate only stops
    # us from certifying pass/review, it does not erase visible defect evidence.
    winner = "R4" if reject_evidence else next(rule for rule, matched, *_ in candidates if matched)
    trace, chosen = [], None
    for rule_id, matched, decision, reason, evidence in candidates:
        selected = rule_id == winner
        trace.append({"rule": rule_id, "condition": bool(matched), "selected": selected, "suggestion": LABELS[decision], "reason": reason, "evidence": evidence})
        if selected:
            chosen = (decision, reason)
    decision, reason = chosen
    return {"decision": decision, "label_zh": LABELS[decision], "reasons": [reason], "trace": trace, "policy_version": policy["version"]}
