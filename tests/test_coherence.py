import time

import pytest

from dec.coherence import implies, hierarchy, is_consistent, joint_decode, mine_constraints, incoherence_rate

PARENT = {"alarm_set": "alarm", "alarm_query": "alarm", "weather_query": "weather"}
H = hierarchy("intent", "scenario", PARENT)


def test_hierarchy_fixes_inconsistent_argmax():
    probs = {"scenario": {"alarm": 0.55, "weather": 0.45}, "intent": {"weather_query": 0.5, "alarm_set": 0.4, "alarm_query": 0.1}}
    assert not is_consistent({"scenario": "alarm", "intent": "weather_query"}, [H])
    adj, info = joint_decode(probs, [H])
    assert info["adjusted"] and info["incoherent_before"]
    assert is_consistent(info["map"], [H])
    assert info["map"] == {"scenario": "weather", "intent": "weather_query"}
    assert all(abs(sum(p.values()) - 1) < 1e-9 for p in adj.values())
    assert adj["intent"]["weather_query"] == pytest.approx(0.45)


def test_hierarchy_unknown_child_is_inconsistent():
    assert not is_consistent({"scenario": "alarm", "intent": "unknown_intent"}, [H])


def test_implies_and_mining():
    C = implies("severe", "yes", "toxic", {"yes"})
    assert not is_consistent({"severe": "yes", "toxic": "no"}, [C]) and is_consistent({"severe": "no", "toxic": "no"}, [C])
    data = [{"severe": "yes", "toxic": "yes"}] * 60 + [{"severe": "no", "toxic": "no"}] * 60 + [{"severe": "no", "toxic": "yes"}] * 30
    mined = mine_constraints(data, min_support=50)
    assert any(c.if_q == "severe" and c.if_val == "yes" and c.then_q == "toxic" for c in mined)
    assert incoherence_rate(data + [{"severe": "yes", "toxic": "no"}], [C]) == 1 / 151


def test_incoherence_rate_empty_is_zero():
    assert incoherence_rate([], [implies("a", "x", "b", {"y"})]) == 0.0


def test_no_consistent_combo_leaves_probs_unchanged():
    probs = {"a": {"x": 1.0}, "b": {"y": 1.0}}
    adj, info = joint_decode(probs, [implies("a", "x", "b", {"z"})])
    assert adj == probs and not info["adjusted"]
    assert info["map_consistent"] is False


def test_seven_binary_questions_prune_fast_and_return_consistent_map():
    qs = [f"q{i}" for i in range(7)]
    probs = {q: {"0": 0.4, "1": 0.6} for q in qs}
    C = implies("q0", "1", "q1", {"1"})
    start = time.perf_counter()
    adj, info = joint_decode(probs, [C])
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
    assert info["map_consistent"] is True
    assert is_consistent(info["map"], [C])
    assert info["top_m_used"] == 5


def test_top_m_reduced_when_enumeration_too_large():
    qs = [f"q{i}" for i in range(8)]
    probs = {q: {"a": 0.5, "b": 0.2, "c": 0.1, "d": 0.1, "e": 0.1} for q in qs}
    adj, info = joint_decode(probs, [])
    assert info["top_m_used"] == 3
    assert info["map_consistent"] is True
