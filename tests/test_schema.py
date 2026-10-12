import json
import pytest
from dec.schema import DecisionItem, Option, Decision, CostMatrix

def _item(options, gold="a", qtype="choice"):
    return DecisionItem(id="x", dataset="t", state_id="s", state="hola", qtype=qtype,
                        instructions="?", options=options, gold=gold, question_key="q")

def test_option_keys_and_gold_index():
    it = _item([Option("a"), Option("b")], gold="b")
    assert it.option_keys() == ["a", "b"]
    assert it.gold_index() == 1

def test_duplicate_keys_rejected():
    with pytest.raises(ValueError):
        _item([Option("a"), Option("a")])

def test_empty_key_rejected():
    with pytest.raises(ValueError):
        _item([Option(""), Option("b")])

def test_gold_must_be_an_option():
    with pytest.raises(ValueError):
        _item([Option("a"), Option("b")], gold="zzz")

def test_noul_requires_yes_no():
    with pytest.raises(ValueError):
        _item([Option("si"), Option("no")], gold="no", qtype="noul")

def test_cost_matrix_defaults():
    cm = CostMatrix(loss={("a", "b"): 5.0})
    assert cm.cost("a", "a") == 0.0
    assert cm.cost("a", "b") == 5.0
    assert cm.cost("b", "a") == 1.0

def test_decision_roundtrip_json():
    d = Decision(question_id="x", type="choice", probs={"a": 0.7, "b": 0.3}, action="a",
                 expected_cost=0.3, mode="benchmark")
    s = json.dumps(d.to_dict())
    assert json.loads(s)["action"] == "a"
