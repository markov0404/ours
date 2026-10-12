"""Tipos del contrato de dec (spec §3)."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import json
import numpy as np

QTYPES = ("noul", "choice", "score")


def state_text(state: str | dict | list) -> str:
    """Texto del estado: tal cual si es str; JSON legible (indent=1, sin escapes ASCII) si es dict/list.

    JevBench trae 35 ítems públicos con `state` estructurado. El estilo jevk5 lo embebe como JSON nativo
    (fiel al runtime original); todo lo demás (numeric, router, memoria MiniLM, contaminación) usa este texto.
    """
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False, indent=1)

@dataclass(frozen=True)
class Option:
    key: str
    desc: str = ""

@dataclass
class DecisionItem:
    id: str
    dataset: str
    state_id: str
    state: str | dict | list
    qtype: str
    instructions: str
    options: list[Option]
    gold: str | None
    question_key: str
    domain: str = ""
    family: str = ""
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.qtype not in QTYPES:
            raise ValueError(f"qtype inválido: {self.qtype}")
        keys = [o.key for o in self.options]
        if len(keys) < 2:
            raise ValueError("se necesitan al menos 2 opciones")
        if any(not k or not k.strip() for k in keys):
            raise ValueError("clave de opción vacía")
        if len(set(keys)) != len(keys):
            raise ValueError(f"claves duplicadas: {keys}")
        if self.qtype == "noul" and set(keys) != {"yes", "no"}:
            raise ValueError("noul requiere exactamente las claves yes/no")
        if self.gold is not None and self.gold not in keys:
            raise ValueError(f"gold {self.gold!r} no está entre las opciones {keys}")

    def option_keys(self) -> list[str]:
        return [o.key for o in self.options]

    def gold_index(self) -> int | None:
        return None if self.gold is None else self.option_keys().index(self.gold)

@dataclass
class Encoded:
    item_id: str
    p0: np.ndarray
    logits: np.ndarray
    h: np.ndarray
    h_last: np.ndarray
    readout: str
    truncated: bool

@dataclass
class CostMatrix:
    loss: dict[tuple[str, str], float] = field(default_factory=dict)
    c_escalate: float | None = None
    irreversible: set[str] = field(default_factory=set)

    def cost(self, action: str, truth: str) -> float:
        if (action, truth) in self.loss:
            return float(self.loss[(action, truth)])
        return 0.0 if action == truth else 1.0

@dataclass
class Decision:
    question_id: str
    type: str
    probs: dict[str, float]
    action: str
    expected_cost: float
    mode: str
    risk: dict[str, Any] = field(default_factory=dict)
    router: dict[str, Any] = field(default_factory=dict)
    gap: str | None = None
    evidence: dict[str, Any] = field(default_factory=dict)
    reason_code: str = "prior"
    precedents: list[dict[str, Any]] = field(default_factory=list)
    subspace: dict[str, Any] = field(default_factory=dict)
    coherence: dict[str, Any] = field(default_factory=dict)
    reversible: bool | None = None
    inverse: str | None = None
    loop_signal: bool = False

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["probs"] = {k: float(v) for k, v in self.probs.items()}
        return d
