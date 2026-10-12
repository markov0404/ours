"""Clasificación selectiva con garantía (Geifman & El-Yaniv 2017) y conformal dividido."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np
from scipy.stats import beta

@dataclass
class SGRResult:
    theta: float; coverage: float; bound: float; insufficient: bool

def _upper_bound(errors: int, n: int, delta: float) -> float:
    """Cota superior Clopper-Pearson unilateral del riesgo con confianza 1-delta."""
    return float(beta.ppf(1 - delta, errors + 1, max(n - errors, 1e-9)))

def sgr_threshold(conf: np.ndarray, correct: np.ndarray, r_star: float, delta: float = 0.05, min_n: int = 30, grid: int = 20) -> SGRResult:
    """Umbral de confianza con riesgo acotado <= r_star con probabilidad 1-delta (Geifman & El-Yaniv 2017).

    Se evalúa la cota Clopper-Pearson en una REJILLA fija de `grid` niveles de cobertura (cuantiles de confianza, fijados antes
    de mirar los datos) con unión de Bonferroni delta/grid, y se elige la mayor cobertura cuya cota cumple. La búsqueda binaria
    original supone la cota monótona en la cobertura, lo que falla con n ~ 1000 (coberturas pequeñas → cotas enormes) y la
    dejaba sin solución aunque existiera. Los cortes caen solo donde la confianza baja estrictamente (empates juntos).
    Si no hay cobertura factible: theta = inf, coverage 0 y `bound` = la menor cota observada en la rejilla.
    """
    conf = np.asarray(conf, float); correct = np.asarray(correct, int)
    if len(conf) < min_n:
        return SGRResult(float("inf"), 0.0, float("nan"), True)
    order = np.argsort(-conf, kind="stable"); c = conf[order]
    err = np.cumsum(1 - correct[order]); n = np.arange(1, len(c) + 1)
    valid = np.array([k for k in range(len(c)) if k == len(c) - 1 or c[k] > c[k + 1]])   # cortes tie-safe
    # rejilla: para cada nivel de cobertura q, el mayor corte válido con n <= q·N (al menos min_n)
    targets = sorted({max(min_n, int(round(q * len(c)))) for q in np.linspace(1.0 / grid, 1.0, grid)})
    cands = sorted({int(valid[valid <= t - 1].max()) for t in targets if (valid <= t - 1).any()})
    d = delta / max(1, len(cands))
    bounds = [_upper_bound(int(err[k]), int(n[k]), d) for k in cands]
    ok = [j for j, b in enumerate(bounds) if b <= r_star]
    if not ok:
        return SGRResult(float("inf"), 0.0, float(min(bounds)) if bounds else float("nan"), False)
    j = max(ok); k = cands[j]; theta = float(c[k])
    selected = conf >= theta
    return SGRResult(theta, float(selected.mean()), float(bounds[j]), False)

def conformal_quantile(nonconf: np.ndarray, alpha: float) -> float:
    """Cuantil conformal dividido (1-alpha) con corrección finita (n+1)."""
    s = np.sort(np.asarray(nonconf, float)); n = len(s)
    k = int(np.ceil((n + 1) * (1 - alpha))); k = min(max(k, 1), n)
    return float(s[k - 1])

def conformal_set(probs: np.ndarray, qhat: float) -> list[int]:
    """Conjunto de clases con puntuación 1-p <= qhat."""
    p = np.asarray(probs, float); return [int(i) for i in np.where(1 - p <= qhat)[0]]
