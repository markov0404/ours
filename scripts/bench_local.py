"""Latencia por decisión en esta máquina: construye un pool con el historial dado y mide encode + memoria + fusión por petición.

    python scripts/bench_local.py --backbone Qwen/Qwen3-0.6B --style numeric --device cpu --n 10
    python scripts/bench_local.py --backbone alibiserikbay/JevK5 --style jevk5 --T 1.22 --device auto

Mide en proceso (sin HTTP; el servidor añade ~1-2 ms). Imprime mediana, p90 y el hardware.
"""
from __future__ import annotations
import argparse, json, os, platform, tempfile, time
from pathlib import Path
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backbone", required=True); ap.add_argument("--style", default="numeric"); ap.add_argument("--T", type=float, default=1.0)
    ap.add_argument("--device", default="auto"); ap.add_argument("--dtype", default="auto"); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--history", default=str(Path(__file__).resolve().parents[1] / "examples" / "history_example.jsonl"))
    ap.add_argument("--threads", type=int, default=None, help="torch.set_num_threads (por defecto, el de torch)")
    a = ap.parse_args()
    import torch
    if a.threads:
        torch.set_num_threads(a.threads)
    from dec.backbone.model import Backbone
    from dec.pool import build_pool, load_history
    from dec.serve.memory_pool import MemoryPool
    t0 = time.time(); bb = Backbone(a.backbone, dtype=a.dtype, style=a.style, temperature=a.T, device=a.device); t_load = time.time() - t0
    with tempfile.TemporaryDirectory() as d:
        t0 = time.time(); build_pool(a.history, d, a.backbone, a.style, a.T, bb.device, "minilm", bb=bb, log=lambda *_: None); t_build = time.time() - t0
        t0 = time.time(); pool = MemoryPool.from_dir(d, bb=bb, device="cuda" if bb.device.startswith("cuda") else "cpu"); t_fit = time.time() - t0
    items = load_history(a.history)
    bb.encode([items[0]], log_every=10**9, token_budget=bb.default_token_budget())   # calentamiento
    lat = []
    for i in range(a.n):
        it = items[i % len(items)]; t0 = time.time()
        e = bb.encode([it], log_every=10**9, token_budget=bb.default_token_budget())[0]
        qk = pool.map_key(it); hit = pool.hit(pool.query_vector(it, e), qk); pool.fuse(e.p0, hit, it.option_keys())
        lat.append(time.time() - t0)
    hw = {"cpu": platform.processor() or platform.machine(), "cpus": os.cpu_count(), "torch_threads": torch.get_num_threads(),
          "gpu": torch.cuda.get_device_name(0) if bb.device.startswith("cuda") else None}
    import resource
    mem = {"peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 1),
           "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if bb.device.startswith("cuda") else None}
    print(json.dumps({"backbone": bb.backbone_id, "device": bb.device, "load_s": round(t_load, 1), "build_pool_s": round(t_build, 1),
                      "fit_s": round(t_fit, 2), "n": a.n, "median_s": round(float(np.median(lat)), 3),
                      "p90_s": round(float(np.percentile(lat, 90)), 3), "mem": mem, "hw": hw}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
