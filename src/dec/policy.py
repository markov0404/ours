"""Política: acción Bayes-óptima con ESCALATE, riesgo selectivo, y ensamblado del objeto Decision (spec §3)."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from dec.schema import DecisionItem, Decision, CostMatrix
from dec.memory import MemoryHit
from dec.router import RouterVerdict


def _py(x: Any) -> Any:
    """Convierte recursivamente tipos numpy a tipos nativos de Python (serializable a JSON)."""
    if isinstance(x, np.ndarray): return [_py(v) for v in x.tolist()]
    if isinstance(x, np.generic): return x.item()
    if isinstance(x, dict): return {k: _py(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [_py(v) for v in x]
    if isinstance(x, set): return sorted(_py(v) for v in x)
    return x


@dataclass
class RiskState:
    """Estado de riesgo selectivo: umbral de confianza theta, nivel alpha y cota de riesgo."""
    theta: float
    alpha: float
    bound: float


class LoopDetector:
    """Detecta repetición de la misma acción para el mismo (estado, pregunta)."""
    def __init__(self) -> None:
        self._seen: set[tuple[str, str, str]] = set()

    def seen(self, state_id: str, question_key: str, action: str) -> bool:
        """Devuelve True si el triple ya se había visto; lo registra."""
        key = (state_id, question_key, action); was = key in self._seen; self._seen.add(key); return was


def bayes_action(probs: dict[str, float], costs: CostMatrix | None, allow_escalate: bool) -> tuple[str, float]:
    """Acción de mínimo coste esperado; ESCALATE compite con coste fijo c_escalate si está permitido."""
    cm = costs or CostMatrix()
    exp = {a: sum(cm.cost(a, y) * p for y, p in probs.items()) for a in probs}
    if allow_escalate and cm.c_escalate is not None: exp["ESCALATE"] = float(cm.c_escalate)
    a = min(exp, key=exp.get); return a, float(exp[a])


def decide(item: DecisionItem, p0: np.ndarray, hit: MemoryHit, fused: np.ndarray, verdict: RouterVerdict, risk: RiskState,
           costs: CostMatrix | None, mode: str, J_mask: np.ndarray | None = None, coherence_info: dict | None = None,
           reason: str = "prior", loop: LoopDetector | None = None, lam: float = 0.0) -> Decision:
    """Ensambla la Decision; en modo benchmark nunca escala; en escalation escala por huecos/consecuencia/oráculo."""
    keys = item.option_keys(); probs = {k: float(v) for k, v in zip(keys, fused)}
    conf = float(max(probs.values())); automated = bool(conf >= float(risk.theta))
    cold = bool(hit.mass == 0 and not hit.precedents and reason == "sin_precedentes")
    evidencia = dict(verdict.evidencia)
    if cold: evidencia["oraculo_na"] = True
    gap = None
    if mode == "benchmark":
        action, ecost = bayes_action(probs, costs, allow_escalate=False)
    else:
        allow = costs is not None and costs.c_escalate is not None
        orac_red = verdict.oraculo == "rojo" and not cold
        if allow and (verdict.huecos == "rojo" or verdict.consecuencia == "alta" or orac_red):
            action, ecost = "ESCALATE", float(costs.c_escalate)
            axes = [a for a, on in (("huecos", verdict.huecos == "rojo"), ("oraculo", orac_red), ("consecuencia", verdict.consecuencia == "alta")) if on]
            miss = evidencia.get("missing_terms", [])
            gap = f"ejes en rojo: {', '.join(axes)}" + (f"; falta en el estado: {', '.join(miss)}" if miss else "")
        else:
            action, ecost = bayes_action(probs, costs, allow_escalate=allow)
    loop_signal = bool(loop.seen(item.state_id, item.question_key, action)) if loop is not None else False
    irreversible = costs.irreversible if costs else set()
    return Decision(
        question_id=item.id, type=item.qtype, probs=_py(probs), action=action, expected_cost=float(ecost), mode=mode,
        risk=_py({"automated": automated, "alpha": risk.alpha, "bound": risk.bound, "theta": risk.theta}),
        router=_py({"huecos": verdict.huecos, "oraculo": verdict.oraculo, "consecuencia": verdict.consecuencia, "evidencia": evidencia}),
        gap=gap,
        evidence=_py({"prior": {k: float(v) for k, v in zip(keys, p0)}, "precedent": hit.p_mem or None, "mass": hit.mass, "lambda": lam}),
        reason_code=reason,
        precedents=_py([{"id": i, "label": l, "weight": w} for i, l, w in hit.precedents[:10]]),
        subspace=_py({"J": [int(i) for i in np.nonzero(J_mask)[0][:64]] if J_mask is not None else [], "size": int(np.count_nonzero(J_mask)) if J_mask is not None else 0}),
        coherence=_py(coherence_info or {}),
        reversible=(action not in irreversible) if costs else None,
        inverse=None, loop_signal=loop_signal)
