"""crh225/plumb-decisions: estados con hasta 3 preguntas tipadas (noul/choice/score)."""
from __future__ import annotations
import hashlib
from datasets import load_dataset
from dec.schema import DecisionItem, Option
from . import register

def _options(q: dict) -> tuple[list[Option], str]:
    """Opciones de una pregunta. Para `score`, criteria puede ser lista o dict con
    claves enteras (se ordenan numéricamente); si hay claves no enteras se
    conserva el orden del dict en vez de fallar."""
    crit = q["criteria"]
    if q["type"] == "noul":
        crit = crit or {}
        return [Option("yes", str(crit.get("true", ""))), Option("no", str(crit.get("false", "")))], "noul"
    if q["type"] == "score":
        if isinstance(crit, list):
            levels = crit
        else:
            try:
                levels = [crit[k] for k in sorted(crit, key=lambda x: int(x))]
            except (ValueError, TypeError):
                levels = list(crit.values())
        return [Option(str(i), str(desc)) for i, desc in enumerate(levels)], "score"
    return [Option(str(k), str(v)) for k, v in crit.items()], "choice"

def _gold(q: dict, expected: str, options: list[Option]) -> str:
    e = str(expected).strip().lower()
    if q["type"] == "noul":
        if e in ("yes", "true", "y", "1"): return "yes"
        if e in ("no", "false", "n", "0"): return "no"
        raise ValueError(f"expected noul no reconocido: {expected!r}")
    keys = [o.key for o in options]
    if str(expected) in keys: return str(expected)
    lower = {k.lower(): k for k in keys}
    if e in lower: return lower[e]
    raise ValueError(f"expected {expected!r} no está en {keys}")

@register("plumb")
def load_plumb(split: str) -> list[DecisionItem]:
    ds = load_dataset("crh225/plumb-decisions", split=split)
    items: list[DecisionItem] = []
    for r in ds:
        q = r["question"]
        opts, qtype = _options(q)
        gold = _gold(q, r["expected"], opts)
        state_id = "plumb:" + hashlib.sha1(r["state"].encode()).hexdigest()[:12]
        items.append(DecisionItem(
            id=f"plumb:{split}:{r['id']}", dataset="plumb", state_id=state_id, state=r["state"],
            qtype=qtype, instructions=q["instructions"], options=opts, gold=gold,
            question_key=f"plumb:{r['family']}:{qtype}", domain=r["domain"], family=r["family"],
            meta={"length": r["length"], "teacher_probs": r.get("teacher_probs"), "explanation": r.get("explanation")},
        ))
    return items
