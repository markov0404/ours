"""Ronda 2: recalibración post-fusión (hip-11), configs nuevas de la escalera y corrección por prior de clase (hip-13)."""
import numpy as np
from dec.fusion import FusionRecord, FusionParams, fit_post_temperature, fuse, p_mem_vector, temper
from dec.memory import MemoryHit, PrecedentMemory
from dec.eval.ladder import fit_fusion_grouped, run_config, CONFIGS
from tests.test_ladder_synthetic import _synthetic


def _overconfident(n=400, K=3, seed=0):
    rng = np.random.default_rng(seed); P, g = [], []
    for _ in range(n):
        y = int(rng.integers(0, K)); p = np.full(K, 0.02); p[y if rng.random() < 0.7 else int(rng.integers(0, K))] = 0.96; p /= p.sum()
        P.append(p); g.append(y)
    return np.stack(P), np.array(g)


def test_post_temperature_reduces_nll_of_overconfident_outputs():
    P, g = _overconfident()
    T = fit_post_temperature(P, g)
    nll = lambda Q: -np.mean(np.log(Q[np.arange(len(g)), g]))
    Pc = np.stack([temper(p, T) for p in P])
    assert T > 1.0 and nll(Pc) < nll(P) - 0.05
    assert fit_post_temperature(P[:10], g[:10]) == 1.0   # pocos datos: sin recalibrar


def test_fit_fusion_grouped_post_sets_T_post():
    rng = np.random.default_rng(1); recs = []
    for i in range(80):
        y = int(rng.integers(0, 3)); p0 = np.full(3, 0.1); p0[y] = 0.8; pm = np.zeros(3); pm[y if rng.random() < 0.6 else 0] = 1.0
        recs.append(FusionRecord(p0, pm, 10.0, y))
    bank = fit_fusion_grouped(recs, post=True)
    assert all(p.T_post > 0 for p in bank.by_k.values()) and bank.default.T_post == bank.by_k[3].T_post


def test_ladder_round2_configs_run():
    prep = _synthetic()
    r1 = run_config(prep, "ours1"); rc = run_config(prep, "ours1cal")
    assert "ours1cal" in CONFIGS and 0 <= rc["acc"] <= 1 and abs(rc["acc"] - r1["acc"]) < 0.05
    assert all("T_post" in p for p in rc["fusion"]["by_k"].values())
    rb = run_config(prep, "ours1", prior_correction=True)
    assert 0 <= rb["acc"] <= 1


def test_prior_correction_rebalances_rare_class():
    hit = MemoryHit({"no": 0.9, "yes": 0.1}, 12.0, [])
    plain = p_mem_vector(hit, ["yes", "no"]); corr = p_mem_vector(hit, ["yes", "no"], prior={"yes": 0.05, "no": 0.95})
    assert plain[0] < plain[1] and corr[0] > corr[1] and abs(corr.sum() - plain.sum()) < 1e-6


def test_class_freq():
    m = PrecedentMemory(4); m.add_many(np.zeros((4, 4), np.float32), ["a", "a", "a", "b"], ["q"] * 4, list("wxyz")); m.build()
    f = m.class_freq("q"); assert abs(f["a"] - 0.75) < 1e-9 and abs(f["b"] - 0.25) < 1e-9 and m.class_freq("nope") == {}
