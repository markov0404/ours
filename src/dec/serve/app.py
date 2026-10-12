"""dec-serve: endpoint compatible con el adaptador typesafe de JevBench. Modo benchmark (B0) o memory (pool del cliente)."""
from __future__ import annotations
import argparse
import json
import numpy as np
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dec.backbone.model import Backbone
from dec.memory import MemoryHit
from dec.policy import RiskState, decide
from dec.router import RouterThresholds, gap_signals, route
from .wire import format_response, parse_request


def create_app(bb: Backbone, mode: str = "benchmark", memory_dir: str | None = None, pool=None) -> FastAPI:
    """Crea la app FastAPI; encode va de a un item (batch-1). `mode="memory"` exige un `MemoryPool` (`pool`)."""
    if mode == "memory" and pool is None:
        raise ValueError("--mode memory requiere un MemoryPool")
    app = FastAPI()
    stats = {"requests": 0, "items": 0, "memory_hits": 0, "unmapped": 0}

    @app.get("/health")
    def health() -> dict:
        h = {"ok": True, "backbone": bb.backbone_id, "mode": mode}
        if mode == "memory":
            h["device"] = getattr(bb, "device", None); h.update(pool.info()); h.update(stats)
        return h

    @app.post("/v1/systemone")
    async def systemone(req: Request):
        try:
            model, items = parse_request(await req.json())
        except (ValueError, KeyError, TypeError) as e:
            return JSONResponse({"error": str(e)}, status_code=422)
        out = []; n_hit = n_unm = 0
        for it in items:
            e = bb.encode([it], log_every=10**9)[0]
            if mode != "memory":
                hit = MemoryHit({}, 0.0, [])
                v = route(gap_signals(it, e.p0, 0.0), None, RouterThresholds(mass=1.0), False)
                out.append(decide(it, e.p0, hit, e.p0, v, RiskState(float("inf"), 0.05, float("nan")), None,
                                  mode=mode, reason="sin_precedentes"))
                continue
            fused, hit, risk, reason, qk = pool.predict(it, e)
            if qk is None:   # sin mapeo: camino solo-backbone (B0), contado
                n_unm += 1; J = None
            else:
                J = pool.mask(); n_hit += int(hit.mass > 0 and bool(hit.precedents))
            v = route(gap_signals(it, fused, hit.mass), None, RouterThresholds(mass=1.0), False)
            out.append(decide(it, e.p0, hit, fused, v, risk, None, mode=mode, J_mask=None if J is None else (np.asarray(J) > 1.0), reason=reason))   # J = dims con peso sobre el promedio
        usage = {}
        if mode == "memory":
            stats["requests"] += 1; stats["items"] += len(items); stats["memory_hits"] += n_hit; stats["unmapped"] += n_unm
            usage = {"items": len(items), "memory_hits": n_hit, "unmapped": n_unm}
        return format_response(model, out, items, usage)

    return app


def main(argv: list[str] | None = None) -> int:
    """CLI dec-serve."""
    ap = argparse.ArgumentParser(prog="dec-serve")
    ap.add_argument("--backbone", default=None, help="id de Hugging Face; con --pool-dir se toma de pool.json si se omite")
    ap.add_argument("--style", default=None, help="jevk5|numeric (por defecto: el del pool, o numeric)")
    ap.add_argument("--T", type=float, default=None, help="temperatura del backbone (por defecto: la del pool, o 1.0)")
    ap.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"], help="auto = bf16 en cuda, fp32 en cpu/mps")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"], help="auto = cuda > mps > cpu")
    ap.add_argument("--threads", type=int, default=None, help="hilos de CPU de torch (por defecto: la cuota de CPU del contenedor si la hay, si no el default de torch)")
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 para exponerlo fuera de la máquina (Docker)")
    ap.add_argument("--port", type=int, default=8090); ap.add_argument("--mode", default=None, choices=["benchmark", "memory"],
                                                                       help="benchmark (por defecto) o memory (implícito con --pool-dir)")
    ap.add_argument("--memory-dir", default=None)
    ap.add_argument("--pool-dir", default=None, help="directorio creado con `dec-pool build` (historial del cliente)")
    ap.add_argument("--memory-pool", default=None, help="dataset:split del pool etiquetado de investigación (ej. plumb:train)")
    ap.add_argument("--memory-n", type=int, default=1000); ap.add_argument("--memory-seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--qkey-map", default=None, help='JSON {"<serve:qtype | prefijo de instructions>": "<question_key del pool>"}')
    ap.add_argument("--qkey-strategy", default="instructions", choices=["instructions", "type", "none"])
    a = ap.parse_args(argv)
    mode = a.mode or ("memory" if a.pool_dir else "benchmark")
    meta = None
    if a.pool_dir:
        if mode != "memory":
            ap.error("--pool-dir implica --mode memory")
        if a.memory_pool:
            ap.error("--pool-dir y --memory-pool son excluyentes")
        from dec.pool import read_pool_meta
        try:
            meta = read_pool_meta(a.pool_dir)
        except FileNotFoundError as e:
            ap.error(str(e))
    backbone = a.backbone or (meta or {}).get("backbone")
    if not backbone:
        ap.error("falta --backbone (o --pool-dir con pool.json)")
    style = a.style or (meta or {}).get("style") or "numeric"
    T = a.T if a.T is not None else float((meta or {}).get("T", 1.0))
    from dec._compat import set_cpu_threads
    set_cpu_threads(a.threads)
    bb = Backbone(backbone, dtype=a.dtype, style=style, temperature=T, device=a.device,
                  revision=(meta or {}).get("revision"))
    pool = None
    qmap = json.loads(a.qkey_map) if a.qkey_map else None
    mem_dev = "cuda" if bb.device.startswith("cuda") else "cpu"   # búsqueda exacta en GPU solo con CUDA
    if a.pool_dir:
        from .memory_pool import MemoryPool
        pool = MemoryPool.from_dir(a.pool_dir, bb=bb, device=mem_dev, alpha=a.alpha, qkey_strategy=a.qkey_strategy, qkey_map=qmap)
    elif mode == "memory":
        if not a.memory_pool or ":" not in a.memory_pool:
            ap.error("--mode memory requiere --pool-dir DIR o --memory-pool dataset:split")
        from .memory_pool import MemoryPool
        ds, sp = a.memory_pool.split(":", 1)
        pool = MemoryPool(bb.backbone_id, bb=bb, pool_from=(ds, sp), n=a.memory_n, seed=a.memory_seed, alpha=a.alpha, device=mem_dev,
                          qkey_strategy=a.qkey_strategy, qkey_map=qmap)
    if pool is not None and pool.question_keys:   # calentamiento (kernels, caches de tokenizer) para que la 1.ª petición no pague el arranque
        from dec.schema import DecisionItem, Option
        bb.encode([DecisionItem(id="warm-up", dataset="serve", state_id="w", state="warm-up", qtype="noul", instructions="warm-up?",
                                options=[Option("yes"), Option("no")], gold=None, question_key="serve:noul")], log_every=10**9)
    uvicorn.run(create_app(bb, mode, a.memory_dir, pool), host=a.host, port=a.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
