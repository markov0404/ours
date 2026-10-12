# tests/test_fusion.py
import numpy as np
import pytest
from dec.fusion import FusionParams, FusionRecord, _nll, fit_fusion, fuse, reason_code, temper

def _records(seed=0, n=400, K=4):
    rng = np.random.default_rng(seed); recs = []
    for _ in range(n):
        g = rng.integers(0, K); p0 = rng.dirichlet(np.ones(K)); p0[g] += 0.3; p0 /= p0.sum()   # prior ruidoso
        mass = float(rng.uniform(0, 10)); pm = np.full(K, 0.02); pm[g] = 0.94                 # memoria casi perfecta
        if mass < 2: pm = rng.dirichlet(np.ones(K))                                            # con poca masa, ruido
        recs.append(FusionRecord(p0, pm, mass, int(g)))
    return recs

def test_fit_learns_to_trust_memory_when_mass_is_high():
    params = fit_fusion(_records())
    hi = fuse(np.array([0.4, 0.3, 0.2, 0.1]), np.array([0.02, 0.94, 0.02, 0.02]), 9.0, params)
    lo = fuse(np.array([0.4, 0.3, 0.2, 0.1]), np.array([0.02, 0.94, 0.02, 0.02]), 0.1, params)
    assert hi.argmax() == 1 and lo.argmax() == 0

def test_zero_mass_returns_tempered_prior_and_sin_precedentes():
    params = fit_fusion(_records())
    p0 = np.array([0.4, 0.3, 0.2, 0.1])
    out = fuse(p0, np.zeros(4), 0.0, params)
    assert np.allclose(out, temper(p0, params.T))
    assert reason_code(p0, np.zeros(4), 0.0, params) == "sin_precedentes"

def test_reason_codes():
    params = fit_fusion(_records())
    p0 = np.array([0.6, 0.2, 0.1, 0.1]); pm = np.array([0.05, 0.9, 0.03, 0.02])
    assert reason_code(p0, pm, params.strong_mass * 2, params) == "atribuible"
    assert reason_code(p0, pm, params.weak_mass / 2, params) == "divergencia"
    assert reason_code(p0, np.array([0.9, 0.05, 0.03, 0.02]), params.strong_mass * 2, params) == "prior"

def _fit_reference(records):
    """Referencia lenta (bucle por registro) para comprobar la versión vectorizada de fit_fusion."""
    g = np.array([r.gold_idx for r in records]); masses = np.array([r.mass for r in records])
    Ts = np.linspace(0.5, 3.0, 26)
    T = float(min(Ts, key=lambda t: _nll(np.stack([temper(r.p0, t) for r in records]), g)))
    best, bp = float("inf"), (1.0, 0.0)
    for a in np.linspace(0.0, 4.0, 17):
        for b in np.linspace(-6.0, 4.0, 21):
            P = np.stack([fuse(r.p0, r.p_mem, r.mass, FusionParams(T, a, b)) for r in records]); n = _nll(P, g)
            if n < best: best, bp = n, (float(a), float(b))
    return T, bp[0], bp[1]

def test_vectorized_fit_matches_reference_loop():
    recs = _records(seed=5, n=120, K=3)
    params = fit_fusion(recs)
    T, a, b = _fit_reference(recs)
    assert (params.T, params.a, params.b) == (T, a, b)

def test_mixed_option_counts_raise_value_error():
    recs = _records(seed=6, n=10, K=4)
    recs[0] = FusionRecord(np.array([0.5, 0.5]), np.array([0.5, 0.5]), 1.0, 0)
    with pytest.raises(ValueError, match="mismo número de opciones"):
        fit_fusion(recs)

def test_on_grid_edge_flag_is_set_on_boundary_fit():
    assert fit_fusion(_records()).on_grid_edge is True
