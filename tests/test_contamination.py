# tests/test_contamination.py
from dec.schema import DecisionItem, Option
from dec.data.contamination import AuditReport, audit, assert_clean
import pytest

def _it(i, text):
    return DecisionItem(id=i, dataset="t", state_id=i, state=text, qtype="choice", instructions="?",
                        options=[Option("a"), Option("b")], gold="a", question_key="q")

def test_detects_shared_8gram_and_exact():
    base = "the quick brown fox jumps over the lazy dog near the river bank today"
    rep = audit([_it("m1", base)], [_it("e1", base), _it("e2", "completely different text about cats")])
    assert ("m1", "e1") in rep.exact
    assert ("m1", "e1") in rep.overlaps
    assert all(e != "e2" for _, e in rep.overlaps)
    with pytest.raises(AssertionError):
        assert_clean(rep)

def test_short_memory_item_contained_in_longer_eval_is_overlap():
    rep = audit([_it("m1", "fox jumps over")],
                [_it("e1", "the quick brown fox jumps over the lazy dog near the river bank today")])
    assert ("m1", "e1") in rep.overlaps
    assert rep.exact == []

def test_assert_clean_raises_explicitly():
    rep = AuditReport(exact=[("m1", "e1")])
    with pytest.raises(AssertionError, match="idénticos"):
        assert_clean(rep)
