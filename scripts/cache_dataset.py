#!/usr/bin/env python
"""Una pasada del backbone por dataset/split, en trozos < --max-minutes, reanudable. Ver Task 21."""
from __future__ import annotations
import argparse, time
from pathlib import Path
from loguru import logger
from dec.data import load
from dec.backbone.model import Backbone
import dec.backbone.cache as C
from dec.backbone.cache import write_cache, read_cache
from dec.eval.ladder import _limit_states
from dec.eval.unanswerable import build_g8_items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True); ap.add_argument("--split", required=True); ap.add_argument("--backbone", required=True)
    ap.add_argument("--style", default="numeric"); ap.add_argument("--T", type=float, default=1.0); ap.add_argument("--dtype", default="bf16")
    ap.add_argument("--chunk", type=int, default=400); ap.add_argument("--max-minutes", type=float, default=25.0)
    ap.add_argument("--token-budget", type=int, default=16384)
    ap.add_argument("--limit-states", type=int, default=None); ap.add_argument("--limit-items", type=int, default=None); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--g8", action="store_true"); ap.add_argument("--cache-dir", default=None)
    a = ap.parse_args()
    if a.cache_dir:
        C.CACHE = Path(a.cache_dir)          # cache_path lee C.CACHE en cada llamada
    items = load(a.dataset, a.split) if a.dataset != "civil" else load(a.dataset, a.split, n=3000, seed=0)
    if a.limit_states: items = _limit_states(items, a.limit_states, a.seed)
    if a.g8 and a.dataset == "plumb": items = items + build_g8_items(items)
    if a.limit_items: items = items[: a.limit_items]
    bb = Backbone(a.backbone, dtype=a.dtype, style=a.style, temperature=a.T)
    path = C.cache_path(a.dataset, a.split, bb.backbone_id)
    done = read_cache(path) if path.exists() else {}
    todo = [it for it in items if it.id not in done]
    print(f"{len(todo)} ítems pendientes de {len(items)} · cache {path}", flush=True)
    for s in range(0, len(todo), a.chunk):
        t0 = time.time(); enc = bb.encode(todo[s:s + a.chunk], token_budget=a.token_budget)
        for e in enc: done[e.item_id] = e
        write_cache(path, list(done.values()), bb.backbone_id)
        mins = (time.time() - t0) / 60
        logger.info(f"trozo {s // a.chunk + 1}: {len(enc)} ítems en {mins:.1f} min ({len(enc) / max(mins, 1e-9):.1f} ítems/min)")
        if mins > a.max_minutes:
            logger.error(f"trozo > {a.max_minutes} min: trozo demasiado largo: reducir --chunk o usar más GPU"); return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
