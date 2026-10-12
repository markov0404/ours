"""G0: mide candidatos de backbone en los 231 ítems públicos de JevBench (precisión, ECE, Brier, latencia, flips).
Uso: python scripts/gates/g0_backbone.py --dtype bf16 --candidates alibiserikbay/JevK5:jevk5:1.22
Cada invocación fusiona sus resultados con results/gates/G0.json (un candidato por invocación en el pod).
"""
from __future__ import annotations
import argparse, time, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from dec.data import load
from dec.backbone.model import Backbone
from dec.backbone.cache import write_cache, cache_path
from dec.metrics import accuracy, ece, brier, flip_rate
from dec.schema import DecisionItem
from scripts.gates import common
from scripts.gates.common import write_gate, record_attempt, HIP

TOKEN_BUDGET = 1024   # medido: en backbones Qwen3.5 el lote es más lento; ~1 ítem por lote

def _reversed(it: DecisionItem) -> DecisionItem:
    return DecisionItem(**{**it.__dict__, "id": it.id + ":rev", "options": list(reversed(it.options))})

def _fla_installed() -> bool:
    try:
        import fla  # noqa: F401
        return True
    except Exception:
        return False

def evaluate(model_id: str, style: str, T: float, dtype: str) -> dict:
    items = load("jevbench", "public")   # estados dict: jevk5 los embebe nativos; numeric usa schema.state_text()
    bb = Backbone(model_id, dtype=dtype, style=style, temperature=T)
    ntok = sum(len(bb._prepare(it)[1]) for it in items)
    t0 = time.time(); enc = bb.encode(items, token_budget=TOKEN_BUDGET); wall = time.time() - t0
    lat = wall / len(items) * 1000
    write_cache(cache_path("jevbench", "public", bb.backbone_id), enc, bb.backbone_id)
    K = max(len(e.p0) for e in enc)   # nº de opciones variable: se rellena con prob. 0 (no altera argmax ni métricas)
    P = np.stack([np.pad(e.p0, (0, K - len(e.p0))) for e in enc]); G = np.array([it.gold_index() for it in items])
    multi = [i for i, it in enumerate(items) if len(it.options) >= 3 and it.qtype == "choice"]
    rev = bb.encode([_reversed(items[i]) for i in multi], token_budget=TOKEN_BUDGET)
    pred_a = P[multi].argmax(1)
    pred_b = np.array([len(items[i].options) - 1 - r.p0.argmax() for i, r in zip(multi, rev)])
    by_tier = {}
    for tier in ("original", "easy", "hard"):
        idx = [i for i, it in enumerate(items) if it.meta["tier"] == tier]
        by_tier[tier] = {"n": len(idx), "acc": accuracy(P[idx], G[idx]), "ece": ece(P[idx], G[idx])}
    return {"model": model_id, "style": style, "T": T, "dtype": dtype, "backbone_id": bb.backbone_id,
            "acc": accuracy(P, G), "ece": ece(P, G), "brier": brier(P, G), "ms_per_item_local": lat,
            "tokens_per_s": ntok / wall, "n_tokens": ntok, "wall_s": wall, "n_multi_reorder": len(multi),
            "fla_installed": _fla_installed(), "flip_rate_reorder": flip_rate(pred_a, pred_b), "by_tier": by_tier}

def try_clef() -> dict:
    """Intento rápido de carga estándar; si no, se difiere."""
    mid = "cloudflare/clef-flash"
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
        import torch
        cfg = AutoConfig.from_pretrained(mid)
        with torch.device("meta"):   # sin reservar pesos en RAM del host
            AutoModelForCausalLM.from_config(cfg)  # falla si la arquitectura no está en transformers
        return {"model": mid, "status": f"config cargable ({cfg.model_type}); no evaluado en este harness"}
    except Exception as e:
        return {"model": mid, "status": "no evaluable en local; requiere su runtime; diferido", "error": repr(e)[:300]}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="*", default=[], help="model:style:T")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "int8"])
    ap.add_argument("--clef", action="store_true", help="intentar cloudflare/clef-flash")
    ap.add_argument("--final", action="store_true", help="registrar el intento en el libro aunque haya un solo candidato")
    a = ap.parse_args()
    path = common.GATES_DIR / "G0.json"
    results = json.loads(path.read_text())["results"] if path.exists() else []
    def put(r):
        key = r.get("backbone_id") or r["model"]
        results[:] = [x for x in results if (x.get("backbone_id") or x["model"]) != key] + [r]
    if a.clef:
        r = try_clef(); print(json.dumps(r, indent=2)); put(r)
    for c in a.candidates:
        model, style, T = c.split(":"); r = evaluate(model, style, float(T), a.dtype); print(json.dumps(r, indent=2)); put(r)
    ev = [r for r in results if "acc" in r]
    if not ev:
        write_gate("G0", {"gate": "G0", "results": results}); return 0
    best = max(ev, key=lambda r: (round(r["acc"], 3), -r["ece"]))
    small = [r for r in ev if "9B" not in r["model"] and "clef" not in r["model"].lower()]
    if small and best not in small and best["acc"] - max(s["acc"] for s in small) <= 0.03:
        best = max(small, key=lambda r: (round(r["acc"], 3), -r["ece"]))   # 9B solo si el 4B queda > 3 pp por debajo
    payload = {"gate": "G0", "results": results, "chosen": best["backbone_id"], "rule": "max acc; ECE desempata; 9B solo si 4B > 3pp por debajo"}
    write_gate("G0", payload)
    if len(ev) >= 2 or a.final:   # el libro solo recibe el veredicto final (varios candidatos o --final), no cada invocación parcial
        record_attempt(HIP["G0"], f"G0 medido: elegido {best['backbone_id']}", json.dumps({r['backbone_id']: (r['acc'], r['ece']) for r in ev}), sustiene=True)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
