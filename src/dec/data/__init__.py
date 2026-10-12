"""Registro de loaders. Cada loader devuelve list[DecisionItem] y cachea en cache/data/."""
from __future__ import annotations
import json, hashlib
from pathlib import Path
from typing import Callable
from dec.schema import DecisionItem, Option

CACHE_DIR = Path("cache/data")
REGISTRY: dict[str, Callable[..., list[DecisionItem]]] = {}

def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco

def _cache_path(name: str, split: str, kw: dict) -> Path:
    tag = hashlib.sha1(json.dumps(kw, sort_keys=True).encode()).hexdigest()[:8] if kw else "base"
    return CACHE_DIR / f"{name}_{split}_{tag}.jsonl"

def to_json(it: DecisionItem) -> dict:
    d = it.__dict__.copy()
    d["options"] = [{"key": o.key, "desc": o.desc} for o in it.options]
    return d

def from_json(d: dict) -> DecisionItem:
    d = dict(d)
    d["options"] = [Option(**o) for o in d["options"]]
    return DecisionItem(**d)

def load(name: str, split: str, refresh: bool = False, **kw) -> list[DecisionItem]:
    from . import plumb, massive, clinc, civil, banking77, jevbench  # noqa: F401  registra
    if name not in REGISTRY:
        raise KeyError(f"dataset desconocido {name!r}; disponibles: {sorted(REGISTRY)}")
    p = _cache_path(name, split, kw)
    if p.exists() and not refresh:
        return [from_json(json.loads(l)) for l in p.open()]
    items = REGISTRY[name](split, **kw)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w") as f:
        for it in items:
            f.write(json.dumps(to_json(it), ensure_ascii=False) + "\n")
    return items
