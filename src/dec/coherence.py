"""Decodificación conjunta de preguntas acopladas bajo restricciones (QCD P3 llevado a decisiones)."""
from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

MAX_JOINT_COMBOS = 200_000


@dataclass(frozen=True)
class Constraint:
    """Restricción entre dos preguntas.

    kind="implies": si if_q toma if_val, then_q debe tomar un valor en then_vals.
    kind="hierarchy": el valor de if_q (hijo) debe tener como padre, según parent_of, el valor de then_q.
    """

    kind: str
    if_q: str
    if_val: str | None
    then_q: str
    then_vals: frozenset[str] | None
    parent_of: dict[str, str] | None


def implies(if_q: str, if_val: str, then_q: str, then_vals: Sequence[str] | set[str] | frozenset[str]) -> Constraint:
    """Crea la restricción: si `if_q == if_val`, entonces `then_q` ∈ `then_vals`."""
    return Constraint("implies", if_q, if_val, then_q, frozenset(then_vals), None)


def hierarchy(child_q: str, parent_q: str, parent_of: dict[str, str]) -> Constraint:
    """Crea la restricción: el valor de `child_q` debe tener como padre el valor de `parent_q`.

    Un valor hijo ausente de `parent_of` se considera incoherente (estricto).
    """
    return Constraint("hierarchy", child_q, None, parent_q, None, dict(parent_of))


def is_consistent(assign: dict[str, str], constraints: Sequence[Constraint]) -> bool:
    """True si la asignación cumple todas las restricciones cuyas dos preguntas están presentes."""
    for c in constraints:
        if c.if_q not in assign or c.then_q not in assign:
            continue
        if c.kind == "implies" and assign[c.if_q] == c.if_val and assign[c.then_q] not in c.then_vals:
            return False
        if c.kind == "hierarchy" and c.parent_of.get(assign[c.if_q]) != assign[c.then_q]:
            return False
    return True


def _enumerate_consistent(
    qs: list[str],
    cands: list[list[str]],
    probs_by_q: dict[str, dict[str, float]],
    constraints: Sequence[Constraint],
) -> list[tuple[float, dict[str, str]]]:
    """Enumera por backtracking las asignaciones completas consistentes, podando en cada nivel."""
    scored: list[tuple[float, dict[str, str]]] = []
    partial: dict[str, str] = {}

    def recurse(depth: int, score: float) -> None:
        if depth == len(qs):
            scored.append((score, dict(partial)))
            return
        q = qs[depth]
        for v in cands[depth]:
            partial[q] = v
            if is_consistent(partial, constraints):
                recurse(depth + 1, score * probs_by_q[q][v])
            del partial[q]

    recurse(0, 1.0)
    return scored


def joint_decode(
    probs_by_q: dict[str, dict[str, float]],
    constraints: Sequence[Constraint],
    top_m: int = 5,
) -> tuple[dict[str, dict[str, float]], dict[str, Any]]:
    """Decodificación conjunta sobre los top_m valores por pregunta.

    Devuelve (marginales renormalizadas sobre las combinaciones coherentes, info). En `info`:
    - `adjusted`: hubo al menos una combinación coherente.
    - `incoherent_before`: el argmax por pregunta ya violaba alguna restricción.
    - `map`: la asignación conjunta de máxima probabilidad entre las coherentes. Solo es la
      asignación a actuar cuando `map_consistent` es True; si no, es el argmax (incoherente).
    - `map_consistent`: True cuando `adjusted` es True.
    - `top_m_used`: el top_m efectivo tras reducirlo (5 -> 3 -> 2) para acotar la enumeración.

    Las marginales llevan las probabilidades; la decisión concreta sale de `info["map"]`.
    Si no hay combinación coherente, devuelve `probs_by_q` sin cambios.
    """
    qs = list(probs_by_q)
    argmax = {q: max(p, key=p.get) for q, p in probs_by_q.items()}
    incoherent = not is_consistent(argmax, constraints)

    m_used = top_m
    for m in [top_m] + [k for k in (3, 2) if k < top_m]:
        m_used = m
        sizes = [min(m, len(probs_by_q[q])) for q in qs]
        if math.prod(sizes) <= MAX_JOINT_COMBOS:
            break
    cands = [sorted(probs_by_q[q], key=probs_by_q[q].get, reverse=True)[:m_used] for q in qs]

    scored = _enumerate_consistent(qs, cands, probs_by_q, constraints)
    if not scored:
        return probs_by_q, {
            "adjusted": False,
            "incoherent_before": incoherent,
            "map": argmax,
            "map_consistent": False,
            "top_m_used": m_used,
        }

    total = sum(s for s, _ in scored)
    adj: dict[str, dict[str, float]] = {}
    for q in qs:
        marg = {v: 0.0 for v in probs_by_q[q]}
        for s, a in scored:
            marg[a[q]] += s / total
        adj[q] = marg
    best = max(scored, key=lambda t: t[0])[1]
    return adj, {
        "adjusted": True,
        "incoherent_before": incoherent,
        "map": best,
        "map_consistent": True,
        "top_m_used": m_used,
    }


def mine_constraints(
    assignments: list[dict[str, str]],
    min_support: int = 50,
    min_conf: float = 0.995,
) -> list[Constraint]:
    """Mina implicaciones `qa=va -> qb=vb` con soporte ≥ min_support y confianza ≥ min_conf.

    Filtro: la pregunta consecuente qb debe tomar más de un valor distinto en el conjunto de datos;
    si es constante, la implicación es trivial y no se devuelve.
    """
    qs = sorted({q for a in assignments for q in a})
    out: list[Constraint] = []
    for qa in qs:
        for qb in qs:
            if qa == qb:
                continue
            values_b = {r[qb] for r in assignments if qb in r}
            if len(values_b) <= 1:
                continue
            vals_a = {a[qa] for a in assignments if qa in a}
            for va in vals_a:
                rows = [a for a in assignments if a.get(qa) == va and qb in a]
                if len(rows) < min_support or not rows:
                    continue
                counts: dict[str, int] = {}
                for r in rows:
                    counts[r[qb]] = counts.get(r[qb], 0) + 1
                vb, n = max(counts.items(), key=lambda t: t[1])
                if n / len(rows) >= min_conf:
                    out.append(implies(qa, va, qb, {vb}))
    return out


def incoherence_rate(assignments: list[dict[str, str]], constraints: Sequence[Constraint]) -> float:
    """Fracción de asignaciones que violan alguna restricción (0.0 si no hay asignaciones)."""
    if not assignments:
        return 0.0
    return sum(not is_consistent(a, constraints) for a in assignments) / len(assignments)
