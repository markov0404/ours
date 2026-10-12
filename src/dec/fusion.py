"""Fusión del prior p0 con el voto de precedentes p_mem por masa de evidencia; códigos de razón (GRAPH 'atribuible')."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from dec.memory import MemoryHit

@dataclass
class FusionRecord:
    """Un caso de entrenamiento: prior, voto de memoria, masa de evidencia y índice de la opción correcta."""
    p0: np.ndarray
    p_mem: np.ndarray
    mass: float
    gold_idx: int

@dataclass
class FusionParams:
    """Parámetros aprendidos: temperatura T del prior, λ(masa)=sigmoide(a*log(masa)+b), umbrales de masa.

    on_grid_edge indica que T, a o b cayeron en el borde de la rejilla de ajuste (señal de rejilla insuficiente).
    """
    T: float = 1.0
    a: float = 1.0
    b: float = 0.0
    weak_mass: float = 0.0
    strong_mass: float = 0.0
    on_grid_edge: bool = False
    T_post: float = 1.0   # ronda 2 (hip-11): temperatura sobre la salida fusionada, ajustada en LOO; 1.0 = sin recalibrar

def temper(p0: np.ndarray, T: float) -> np.ndarray:
    """Aplica temperatura T a una distribución (softmax de log p / T)."""
    z = np.log(np.clip(p0, 1e-9, 1)) / T
    z -= z.max()
    e = np.exp(z)
    return e / e.sum()

def _lambda(mass: float, a: float, b: float) -> float:
    """Peso λ de la memoria para una masa dada."""
    return float(1 / (1 + np.exp(-(a * np.log(mass + 1e-6) + b))))

def fuse(p0: np.ndarray, p_mem: np.ndarray, mass: float, params: FusionParams) -> np.ndarray:
    """Mezcla prior tempered y voto de memoria con peso λ(masa); sin masa o sin voto devuelve el prior tempered."""
    pt = temper(p0, params.T)
    if mass <= 0 or p_mem.sum() <= 0:
        return pt
    lam = _lambda(mass, params.a, params.b)
    pm = p_mem / p_mem.sum()
    out = (1 - lam) * pt + lam * pm
    return out / out.sum()

def _nll(P: np.ndarray, g: np.ndarray) -> float:
    """Log-pérdida media de la probabilidad asignada a la opción correcta."""
    return float(-np.mean(np.log(np.clip(P[np.arange(len(g)), g], 1e-9, 1))))

def _stack(records: list[FusionRecord]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apila los registros en matrices (p0, p_mem normalizado, válido, masa, gold) para evaluar en vectorial."""
    P0 = np.stack([np.asarray(r.p0, np.float64) for r in records])
    PM = np.stack([np.asarray(r.p_mem, np.float64) for r in records])
    s = PM.sum(1)
    valid = (np.array([r.mass for r in records]) > 0) & (s > 0)
    PM = np.where(valid[:, None], PM / np.where(s > 0, s, 1.0)[:, None], 0.0)
    masses = np.array([r.mass for r in records], np.float64)
    g = np.array([r.gold_idx for r in records])
    return P0, PM, valid, masses, g

def fit_fusion(records: list[FusionRecord]) -> FusionParams:
    """Ajusta T (rejilla 0.5..3.0), (a, b) (rejilla 17x21) por NLL y los umbrales débil/fuerte por cuantiles de masa.

    Todos los records deben tener el mismo número de opciones K: los parámetros se ajustan por dataset
    (o por question_key), no mezclando datasets con K distintos. Lanza ValueError si no se cumple.
    """
    ks = {r.p0.shape[0] for r in records} | {r.p_mem.shape[0] for r in records}
    if len(ks) != 1:
        raise ValueError("fit_fusion requiere records con el mismo número de opciones (ajustar por dataset/question_key)")
    P0, PM, valid, masses, g = _stack(records)
    Ts = np.linspace(0.5, 3.0, 26)

    def _temper_all(T: float) -> np.ndarray:
        z = np.log(np.clip(P0, 1e-9, 1)) / T
        z -= z.max(1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(1, keepdims=True)

    T = float(min(Ts, key=lambda t: _nll(_temper_all(t), g)))
    Pt = _temper_all(T)
    logm = np.zeros(len(masses))
    logm[valid] = np.log(masses[valid] + 1e-6)
    a_grid = np.linspace(0.0, 4.0, 17)
    b_grid = np.linspace(-6.0, 4.0, 21)
    best, bp = float("inf"), (1.0, 0.0)
    for a in a_grid:
        for b in b_grid:
            lam = np.where(valid, 1 / (1 + np.exp(-(a * logm + b))), 0.0)[:, None]
            P = (1 - lam) * Pt + lam * PM
            P = P / P.sum(1, keepdims=True)
            n = _nll(P, g)
            if n < best:
                best, bp = n, (float(a), float(b))
    pos = masses[masses > 0]
    weak = float(np.quantile(pos, 0.2)) if len(pos) else 0.0
    strong = float(np.quantile(pos, 0.6)) if len(pos) else 0.0
    a_fit, b_fit = bp
    on_edge = (
        bool(np.isclose(T, Ts[0]) or np.isclose(T, Ts[-1]))
        or bool(np.isclose(a_fit, a_grid[0]) or np.isclose(a_fit, a_grid[-1]))
        or bool(np.isclose(b_fit, b_grid[0]) or np.isclose(b_fit, b_grid[-1]))
    )
    return FusionParams(T, a_fit, b_fit, weak, strong, on_edge)

def fit_post_temperature(P: np.ndarray, g: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Temperatura T_post que minimiza la NLL de las salidas FUSIONADAS (LOO) `P` frente a `g` (hip-11, recalibración post-fusión)."""
    P = np.asarray(P, np.float64); g = np.asarray(g)
    if len(g) < 20:
        return 1.0
    grid = np.linspace(0.5, 3.0, 26) if grid is None else grid
    def _nll_T(T: float) -> float:
        z = np.log(np.clip(P, 1e-9, 1)) / T; z -= z.max(1, keepdims=True); e = np.exp(z); return _nll(e / e.sum(1, keepdims=True), g)
    return float(min(grid, key=_nll_T))


def reason_code(p0: np.ndarray, p_mem: np.ndarray, mass: float, params: FusionParams) -> str:
    """Código de razón de la decisión: sin_precedentes, divergencia, atribuible o prior."""
    if mass <= 0 or p_mem.sum() <= 0:
        return "sin_precedentes"
    if mass < params.weak_mass:
        return "divergencia"
    if p_mem.argmax() != p0.argmax() and mass >= params.strong_mass and p_mem.max() >= 0.7:
        return "atribuible"
    return "prior"

def p_mem_vector(hit: MemoryHit, option_keys: list[str], prior: dict[str, float] | None = None) -> np.ndarray:
    """Convierte el p_mem de un MemoryHit en vector ordenado según option_keys.

    `prior` (hip-13): frecuencias de clase de la partición de memoria; si se pasa, el voto se divide por ellas y se renormaliza
    (corrige el sesgo hacia la clase mayoritaria con positivos raros; la masa no cambia).
    """
    v = np.array([hit.p_mem.get(k, 0.0) for k in option_keys], np.float32)
    if prior and v.sum() > 0:
        w = np.array([1.0 / max(prior.get(k, 0.0), 1e-3) for k in option_keys], np.float32)
        v = v * w; v = v / v.sum() * float(np.array([hit.p_mem.get(k, 0.0) for k in option_keys]).sum())
    return v
