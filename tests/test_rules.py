import pytest
from industrial_inspection.rules import decide

POLICY = dict(version="test", min_brightness=15,max_brightness=245,min_sharpness=3,review_margin=.1,min_anomaly_area=.005,reject_area=.06,critical_overlap=.2)


@pytest.mark.parametrize("score,area,overlap,brightness,expected", [
    (1,0,0,0,"undetermined"), (.5,0,0,100,"pass"), (1,.001,0,100,"review"),
    (2,.02,.8,100,"reject"), (2,.1,0,100,"reject"), (2,.001,0,100,"review")])
def test_decision_priority(score, area, overlap, brightness, expected):
    facts = dict(score=score,image_threshold=1,anomaly_area_ratio=area,critical_overlap_ratio=overlap,brightness=brightness,sharpness=100)
    result = decide(facts, POLICY)
    assert result["decision"] == expected
    assert sum(t["selected"] for t in result["trace"]) == 1


def test_bad_facts_are_not_accepted():
    with pytest.raises(ValueError):
        decide({"score":float("nan")}, POLICY)
