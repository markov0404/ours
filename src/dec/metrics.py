"""Métricas de la escalera de baselines y de los gates."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, cohen_kappa_score

from dec.coherence import incoherence_rate  # noqa: F401
from dec.schema import CostMatrix


def accuracy(probs: np.ndarray | Sequence[Sequence[float]], gold_idx: np.ndarray | Sequence[int]) -> float:
    """Fracción de filas cuya clase argmax coincide con la clase gold."""
    return float(np.mean(np.argmax(np.asarray(probs), 1) == np.asarray(gold_idx)))


def ece(probs: np.ndarray | Sequence[Sequence[float]], gold_idx: np.ndarray | Sequence[int], bins: int = 10) -> float:
    """Expected calibration error con bins uniformes sobre la confianza."""
    p = np.asarray(probs); g = np.asarray(gold_idx)
    conf = p.max(1); correct = (p.argmax(1) == g).astype(float)
    edges = np.linspace(0, 1, bins + 1); total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(total)


def brier(probs: np.ndarray | Sequence[Sequence[float]], gold_idx: np.ndarray | Sequence[int]) -> float:
    """Brier score multiclase: media de la suma de errores cuadráticos por fila."""
    p = np.asarray(probs); y = np.zeros_like(p); y[np.arange(len(p)), np.asarray(gold_idx)] = 1
    return float(np.mean(((p - y) ** 2).sum(1)))


def qwk(pred_idx: np.ndarray | Sequence[int], gold_idx: np.ndarray | Sequence[int], K: int) -> float:
    """Quadratic weighted kappa sobre K clases ordinales."""
    return float(cohen_kappa_score(gold_idx, pred_idx, weights="quadratic", labels=list(range(K))))


def auroc(scores: np.ndarray | Sequence[float], positives: np.ndarray | Sequence[int]) -> float:
    """Área bajo la curva ROC para scores frente a etiquetas binarias."""
    return float(roc_auc_score(np.asarray(positives), np.asarray(scores)))


def flip_rate(pred_a: np.ndarray | Sequence[object], pred_b: np.ndarray | Sequence[object]) -> float:
    """Fracción de predicciones que cambian entre dos sistemas."""
    return float(np.mean(np.asarray(pred_a) != np.asarray(pred_b)))


def realized_cost(actions: Sequence[str], truths: Sequence[str], cost_matrix: CostMatrix) -> float:
    """Coste total realizado de las acciones frente a las verdades, con ESCALATE a c_escalate."""
    return float(sum(cost_matrix.cost(a, t) if a != "ESCALATE" else (cost_matrix.c_escalate or 0.0) for a, t in zip(actions, truths)))
