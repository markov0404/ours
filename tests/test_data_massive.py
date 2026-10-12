import pytest
from dec.data import load

@pytest.mark.timeout(900)
def test_massive_two_questions_per_state():
    items = load("massive", "test", locale="en-US")
    by_state = {}
    for it in items: by_state.setdefault(it.state_id, []).append(it)
    assert all(len(v) == 2 for v in by_state.values())
    sc = [it for it in items if it.question_key == "massive:scenario"][0]
    inns = [it for it in items if it.question_key == "massive:intent"]
    assert len(sc.options) == 18 and len(inns[0].options) == 60
    scenario_keys = sc.option_keys()
    for it in inns:
        assert it.meta["scenario_of_intent"][it.gold] in scenario_keys
