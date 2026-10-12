"""Ítems públicos de JevBench (vendor/jevbench/datasets/public/*.jsonl, MIT)."""
from __future__ import annotations
import json
from pathlib import Path
from dec.schema import DecisionItem, Option
from . import register

PUBLIC = Path("vendor/jevbench/datasets/public")
FILES = {"original": "original.jsonl", "easy": "easy.jsonl", "hard": "hard.jsonl"}

def _to_item(r: dict, tier: str) -> DecisionItem:
    q, labels = r["question"], [str(x) for x in r["labels"]]
    crit = q.get("criteria") or {}
    if q["type"] == "noul":
        opts = [Option("yes", str(crit.get("true", "yes"))), Option("no", str(crit.get("false", "no")))]
        gold = "yes" if str(r["expected"]).lower() in ("yes", "true") else "no"
    elif q["type"] == "score":
        levels = crit if isinstance(crit, list) else [crit[k] for k in labels]
        opts = [Option(str(i), str(d)) for i, d in enumerate(levels)]
        gold = str(r["expected"])
    else:
        if isinstance(crit, list):
            descs = [str(d) for d in crit] if len(crit) == len(labels) else labels
            opts = [Option(k, d) for k, d in zip(labels, descs)]
        else:
            # Orden de `criteria` (lo que envía el harness y ve el modelo), no de `labels`: difieren en 193/231 ítems
            # públicos y JevK5 es sensible al orden. Etiquetas ausentes en criteria van al final.
            opts = [Option(str(k), str(v)) for k, v in crit.items()] + [Option(k, k) for k in labels if k not in crit]
        gold = str(r["expected"])
    return DecisionItem(id=f"jevbench:{r['id']}", dataset="jevbench", state_id=f"jevbench:{r['id']}",
        state=r["state"], qtype=q["type"], instructions=q["instructions"], options=opts, gold=gold,
        question_key=f"jevbench:{tier}:{q['type']}", domain=r.get("family", ""), family=r.get("family", ""),
        meta={"tier": tier, "group": r.get("group"), "split": r.get("split")})

@register("jevbench")
def load_jevbench(split: str = "public") -> list[DecisionItem]:
    """Carga los tres tiers públicos. `split` se ignora (solo existe "public")."""
    if not PUBLIC.exists():
        raise FileNotFoundError("corré scripts/fetch_jevbench.sh")
    items = []
    for tier, fn in FILES.items():
        for line in (PUBLIC / fn).open():
            if line.strip():
                items.append(_to_item(json.loads(line), tier))
    return items
