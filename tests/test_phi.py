import numpy as np, torch
from dec.phi import GlobalDiagonalMetric, PhiMask, fit_nca
from dec.memory import PrecedentMemory

def _synthetic(seed=0, n=600, d=64):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n); q = rng.integers(0, 2, n)
    H = rng.normal(size=(n, d)).astype(np.float32)
    H[q == 0, 0:8] += y[q == 0, None] * 1.2      # pregunta A: dims 0-7
    H[q == 1, 8:16] += y[q == 1, None] * 1.2     # pregunta B: dims 8-15
    Q = np.zeros((n, 4), np.float32); Q[q == 0, 0] = 1; Q[q == 1, 1] = 1
    return H, [f"c{v}" for v in y], [f"q{v}" for v in q], Q

def _knn_acc(mem, H, y, qk, masks):
    ok = 0
    for i in range(0, 200):
        hit = mem.search(H[i], qk[i], mask=masks[i], exclude_id=f"i{i}")
        ok += max(hit.p_mem, key=hit.p_mem.get) == y[i]
    return ok / 200

def test_global_metric_upweights_informative_dims_and_improves_knn():
    H, y, qk, Q = _synthetic(); torch.manual_seed(0)
    m = GlobalDiagonalMetric(64); fit_nca(m, H, y, qk, None, epochs=30, lr=3e-2)
    w = m.mask(); assert w[:16].mean() > 2 * w[16:].mean()
    mem = PrecedentMemory(64, k=16); mem.add_many(H, y, qk, [f"i{i}" for i in range(len(y))]); mem.build()
    base = _knn_acc(mem, H, y, qk, [None] * len(y)); learned = _knn_acc(mem, H, y, qk, [w] * len(y))
    assert learned > base + 0.05

def test_phi_mask_selects_question_specific_dims():
    H, y, qk, Q = _synthetic(); torch.manual_seed(0)
    phi = PhiMask(64, d_q=4, d_p=16, J=8); fit_nca(phi, H, y, qk, Q, epochs=25, lr=3e-2)
    mA = phi.mask(Q[qk.index("q0")]); mB = phi.mask(Q[qk.index("q1")])
    assert mA.sum() == 8 and mB.sum() == 8
    assert mA[:8].sum() >= 6 and mB[8:16].sum() >= 6

def test_no_nan_when_anchor_has_no_positive_and_stats_returned():
    H, y, qk, Q = _synthetic(n=200)
    y[0] = "unico"; qk[0] = "q0"  # clase única en su grupo
    for i in range(1, 200):
        if y[i] == "unico": y[i] = "c0"
    m = GlobalDiagonalMetric(64); out = fit_nca(m, H, y, qk, None, epochs=3)
    assert np.isfinite(out["loss"]).all() and np.isfinite(m.mask()).all()
    assert out["mu"].shape == (64,) and out["sd"].shape == (64,)

def test_phi_rejects_J_larger_than_d():
    import pytest
    with pytest.raises(ValueError): PhiMask(8, 4, J=9)

def test_singleton_question_group_keeps_params_finite():
    H, y, qk, Q = _synthetic(n=200)
    qk[0] = "solo"
    for m in (GlobalDiagonalMetric(64), PhiMask(64, 4, d_p=8, J=8)):
        out = fit_nca(m, H, y, qk, None if isinstance(m, GlobalDiagonalMetric) else Q, epochs=3)
        assert np.isfinite(out["loss"]).all()
        assert all(torch.isfinite(p).all() for p in m.parameters())
