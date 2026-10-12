import warnings
import numpy as np
import pytest
from dec.schema import DecisionItem, Option, CostMatrix
from dec.router import gap_signals, route, RouterThresholds, fit_thresholds, GapSignals

def _item(state, instr="When will the 44.39 credit be applied to the account?"):
    return DecisionItem(id="i", dataset="t", state_id="s", state=state, qtype="choice", instructions=instr,
                        options=[Option("a"), Option("b"), Option("c")], gold="a", question_key="q")

def test_overlap_detects_missing_entities():
    s_ok = gap_signals(_item("The 44.39 credit will be applied to the account in April."), np.array([0.34, 0.33, 0.33]), 0.0)
    s_bad = gap_signals(_item("The weather is nice today."), np.array([0.34, 0.33, 0.33]), 0.0)
    assert s_ok.overlap > 0.6 and s_bad.overlap < 0.3 and "44.39" in s_bad.missing_terms
    assert abs(s_ok.entropy_norm - 1.0) < 0.01

def test_entropy_k1_and_unnormalized():
    item = _item("The 44.39 credit")
    assert gap_signals(item, np.array([1.0]), 0.0).entropy_norm == 0.0
    assert abs(gap_signals(item, np.array([0.9, 0.9, 0.9]), 0.0).entropy_norm - 1.0) < 1e-9

def test_route_lexicographic():
    thr = RouterThresholds(entropy=0.8, mass=1.0, overlap=0.3, c_high=5.0)
    red = GapSignals(0.95, 0.1, 0.1, 3, ["44.39"]); green = GapSignals(0.2, 5.0, 0.9, 3, [])
    v = route(red, None, thr, has_constraints=False)
    assert v.huecos == "rojo" and v.oraculo == "rojo" and v.any_red
    v2 = route(green, None, thr, has_constraints=False)
    assert v2.huecos == "verde" and v2.oraculo == "verde" and v2.consecuencia == "baja" and not v2.any_red
    v3 = route(green, CostMatrix(loss={("a", "b"): 50.0}), thr, has_constraints=False)
    assert v3.consecuencia == "alta" and v3.any_red
    v4 = route(GapSignals(0.95, 0.1, 0.9, 3, []), None, thr, has_constraints=True)
    assert v4.huecos == "verde" and v4.oraculo == "verde"

def test_mass_axis_disabled_when_zero():
    thr = RouterThresholds(entropy=0.8, mass=0.0, overlap=0.3)
    v = route(GapSignals(0.95, 0.0, 0.1, 3, ["44.39"]), None, thr, has_constraints=False)
    assert v.huecos == "rojo" and v.oraculo == "rojo"
    v2 = route(GapSignals(0.95, 0.0, 0.1, 3, ["44.39"]), None, thr, has_constraints=True)
    assert v2.oraculo == "verde"

def test_fit_thresholds_separates_synthetic():
    rng = np.random.default_rng(0)
    ans = [GapSignals(rng.uniform(0, 0.7), rng.uniform(1, 5), rng.uniform(0.5, 1), 4, []) for _ in range(200)]
    un = [GapSignals(rng.uniform(0.7, 1), rng.uniform(0, 1), rng.uniform(0, 0.4), 4, ["x"]) for _ in range(200)]
    thr = fit_thresholds(ans, un)
    hits = np.mean([route(s, None, thr, False).huecos == "rojo" for s in un] + [route(s, None, thr, False).huecos == "verde" for s in ans])
    assert hits > 0.9

def test_fit_all_zero_masses_uses_entropy_overlap():
    rng = np.random.default_rng(1)
    ans = [GapSignals(rng.uniform(0, 0.7), 0.0, rng.uniform(0.5, 1), 4, []) for _ in range(200)]
    un = [GapSignals(rng.uniform(0.7, 1), 0.0, rng.uniform(0, 0.4), 4, ["x"]) for _ in range(200)]
    thr = fit_thresholds(ans, un)
    assert thr.mass == 0.0
    hits = np.mean([route(s, None, thr, False).huecos == "rojo" for s in un] + [route(s, None, thr, False).huecos == "verde" for s in ans])
    assert hits > 0.9

def test_fit_rejects_empty():
    with pytest.raises(ValueError):
        fit_thresholds([], [GapSignals(0.9, 0.0, 0.1, 3, [])])

def test_fit_warns_when_no_separation():
    same = [GapSignals(0.5, 1.0, 0.5, 3, []) for _ in range(20)]
    with pytest.warns(UserWarning):
        fit_thresholds(same, same)
