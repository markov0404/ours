import numpy as np
from dec.risk import sgr_threshold, conformal_quantile, conformal_set

def _calibrated(seed: int, n: int = 2000) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed); conf = rng.uniform(0.5, 1.0, n)
    correct = (rng.uniform(size=n) < conf).astype(int)          # perfectamente calibrado
    return conf, correct

def test_sgr_bound_holds_on_fresh_data():
    # Con r*=0.05 hay una ventana factible (137-195 ejemplos) que la búsqueda de 11 pasos no alcanza;
    # devuelve "sin umbral" (seguro). Aquí r*=0.1 para que el caso ejercite un corte real.
    conf, corr = _calibrated(0); res = sgr_threshold(conf, corr, r_star=0.1, delta=0.01)
    assert not res.insufficient and res.bound <= 0.1 and 0.1 < res.coverage < 0.9
    conf2, corr2 = _calibrated(1); sel = conf2 >= res.theta
    assert 1 - corr2[sel].mean() <= 0.1           # riesgo real en lo automatizado bajo r*

def test_sgr_insufficient_data():
    res = sgr_threshold(np.array([0.9, 0.8]), np.array([1, 1]), r_star=0.05)
    assert res.insufficient and res.theta == float("inf") and res.coverage == 0.0

def test_sgr_tied_confidences_are_tie_safe():
    conf = np.concatenate([np.full(500, 0.9), np.full(500, 0.6)])
    idx = np.arange(1000)
    correct = np.where(idx < 500, idx % 10 != 0, idx % 2 == 0).astype(int)  # 10% errores en 0.9, 50% en 0.6
    res = sgr_threshold(conf, correct, r_star=0.2)
    sel = conf >= res.theta
    assert res.theta == 0.9 and not res.insufficient
    assert 1 - correct[sel].mean() <= res.bound              # riesgo realizado en calibración <= cota
    assert np.isclose(res.coverage, sel.mean())              # cobertura = fracción seleccionada real

def test_conformal_coverage():
    rng = np.random.default_rng(0); K = 5
    P = rng.dirichlet(np.ones(K), 3000); g = np.array([rng.choice(K, p=p) for p in P])
    cal = slice(0, 1500)
    q = conformal_quantile(1 - P[cal][np.arange(1500), g[cal]], alpha=0.1)
    cover = np.mean([g[i] in conformal_set(P[i], q) for i in range(1500, 3000)])
    assert cover >= 0.88
