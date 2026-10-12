"""Auditoría de 8-gramas entre la memoria y el conjunto de evaluación (convención JevBench)."""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from dec.schema import DecisionItem, state_text

_TOK = re.compile(r"[a-z0-9]+")

def _norm(s: str) -> list[str]:
    return _TOK.findall(s.lower())

def _ngrams(toks: list[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}

def _contains(hay: list[str], needle: list[str]) -> bool:
    k = len(needle)
    return k > 0 and any(hay[i:i + k] == needle for i in range(len(hay) - k + 1))

@dataclass
class AuditReport:
    overlaps: list[tuple[str, str]] = field(default_factory=list)
    exact: list[tuple[str, str]] = field(default_factory=list)

def audit(memory_items: list[DecisionItem], eval_items: list[DecisionItem], n: int = 8) -> AuditReport:
    rep = AuditReport()
    ev = [(e.id, _norm(state_text(e.state))) for e in eval_items]
    index: dict[tuple[str, ...], list[str]] = {}
    for eid, toks in ev:
        if len(toks) >= n:
            for g in _ngrams(toks, n):
                index.setdefault(g, []).append(eid)
    short_ev = [(eid, toks) for eid, toks in ev if 0 < len(toks) < n]
    exact_text = {" ".join(toks): eid for eid, toks in ev if toks}
    for m in memory_items:
        toks = _norm(state_text(m.state))
        if not toks:
            continue
        text = " ".join(toks)
        if text in exact_text:
            rep.exact.append((m.id, exact_text[text]))
        hit: set[str] = set()
        if len(toks) >= n:
            for g in _ngrams(toks, n):
                hit.update(index.get(g, ()))
            # ítems de evaluación más cortos que n: se comprueba si están contenidos en la memoria
            hit.update(eid for eid, etoks in short_ev if _contains(toks, etoks))
        else:
            # memoria más corta que n: se busca como subsecuencia contigua en cada ítem de evaluación
            hit.update(eid for eid, etoks in ev if _contains(etoks, toks))
        rep.overlaps.extend((m.id, eid) for eid in sorted(hit))
    return rep

def assert_clean(rep: AuditReport) -> None:
    if rep.exact:
        raise AssertionError(f"{len(rep.exact)} ítems de memoria idénticos a ítems de evaluación: {rep.exact[:5]}")
