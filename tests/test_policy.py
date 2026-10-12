import json
import numpy as np
from dec.schema import DecisionItem, Option, CostMatrix
from dec.memory import MemoryHit
from dec.router import RouterVerdict
from dec.policy import bayes_action, decide, RiskState, LoopDetector

IT = DecisionItem(id="i", dataset="t", state_id="s", state="x", qtype="choice", instructions="?",
                  options=[Option("approve"), Option("reject")], gold="approve", question_key="q")
GREEN = RouterVerdict("verde", "verde", "baja", {}, False)
RED = RouterVerdict("rojo", "verde", "baja", {"missing_terms": ["44.39"]}, True)


def test_bayes_action_flips_with_asymmetric_costs():
    p = {"approve": 0.7, "reject": 0.3}
    assert bayes_action(p, None, False)[0] == "approve"
    cm = CostMatrix(loss={("approve", "reject"): 10.0})
    assert bayes_action(p, cm, False)[0] == "reject"
    cm2 = CostMatrix(loss={("approve", "reject"): 10.0}, c_escalate=0.5)
    assert bayes_action(p, cm2, True)[0] == "ESCALATE"


def test_benchmark_mode_never_escalates_and_empty_memory_is_prior():
    p0 = np.array([0.6, 0.4]); hit = MemoryHit({}, 0.0, [])
    d = decide(IT, p0, hit, p0, RED, RiskState(float("inf"), 0.05, float("nan")), CostMatrix(c_escalate=1.0), mode="benchmark", reason="sin_precedentes")
    assert d.action == "approve" and d.reason_code == "sin_precedentes" and d.router["huecos"] == "rojo" and not d.risk["automated"]
    assert d.probs == {"approve": 0.6, "reject": 0.4}


def test_escalation_mode_names_the_gap():
    p0 = np.array([0.5, 0.5]); hit = MemoryHit({}, 0.0, [])
    d = decide(IT, p0, hit, p0, RED, RiskState(0.9, 0.05, 0.03), CostMatrix(c_escalate=1.0), mode="escalation")
    assert d.action == "ESCALATE" and "44.39" in d.gap


def test_loop_detector():
    ld = LoopDetector()
    assert not ld.seen("s", "q", "approve") and ld.seen("s", "q", "approve")


def test_cold_start_oraculo_does_not_escalate():
    p0 = np.array([0.6, 0.4]); hit = MemoryHit({}, 0.0, [])
    v = RouterVerdict("verde", "rojo", "baja", {}, True)
    d = decide(IT, p0, hit, p0, v, RiskState(0.5, 0.05, 0.03), CostMatrix(c_escalate=1.0), mode="escalation", reason="sin_precedentes")
    assert d.action == "approve" and d.router["evidencia"]["oraculo_na"] is True


def test_oraculo_red_escalates_when_not_cold_start():
    p0 = np.array([0.6, 0.4]); hit = MemoryHit({}, 0.0, [])
    v = RouterVerdict("verde", "rojo", "baja", {}, True)
    d = decide(IT, p0, hit, p0, v, RiskState(0.5, 0.05, 0.03), CostMatrix(c_escalate=0.1), mode="escalation", reason="prior")
    assert d.action == "ESCALATE" and "oraculo" in d.gap


def test_green_goes_bayes_and_to_dict_is_json_with_numpy():
    p0 = np.array([0.6, 0.4], dtype=np.float32)
    hit = MemoryHit({"approve": np.float32(0.8)}, np.float64(2.0), [("p1", "approve", np.float32(0.5))])
    d = decide(IT, p0, hit, p0, GREEN, RiskState(np.float64(0.5), 0.05, np.float32(0.02)), CostMatrix(), mode="escalation",
               J_mask=np.array([True, False, True]), coherence_info={"k": np.int64(3)}, loop=LoopDetector())
    assert d.action == "approve" and d.risk["automated"] is True and d.subspace["size"] == 2
    json.dumps(d.to_dict())
