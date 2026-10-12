import pytest
from dec.data import load

@pytest.mark.timeout(900)
def test_plumb_test_split_shape():
    items = load("plumb", "test")
    assert len(items) == 131
    it = items[0]
    assert it.qtype in ("noul", "choice", "score")
    assert it.gold in it.option_keys()
    assert it.family and it.domain
    assert it.question_key.startswith("plumb:")

@pytest.mark.timeout(900)
def test_plumb_groups_questions_by_state():
    items = load("plumb", "train")
    by_state = {}
    for it in items:
        by_state.setdefault(it.state_id, []).append(it)
    assert max(len(v) for v in by_state.values()) == 3
    assert sum(1 for v in by_state.values() if len(v) > 1) > 1000
