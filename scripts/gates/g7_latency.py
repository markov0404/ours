"""G7: latencia. Sobrecoste de toda la capa (phi.mask + memory.search + fusión + decodificación conjunta + router + política)
por decisión de 3 preguntas, medido con scripts/bench_layer.py. Regla (spec §7): p50 < 3 ms en H100. Se mide con la búsqueda
de precedentes en GPU (`--device cuda`, réplica exacta de la de CPU) y se reporta también la de CPU, con 21k y 3k precedentes."""
from __future__ import annotations
import argparse, json, platform, subprocess, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.gates._util import finish

THRESHOLD_MS = 3.0


def bench(n_mem: int, device: str, iters: int, out: Path) -> dict:
    cmd = [sys.executable, "scripts/bench_layer.py", "--n-mem", str(n_mem), "--d", "2560", "--k", "32", "--iters", str(iters), "--device", device, "--out", str(out)]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return json.loads(out.read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--results-dir", default="results")
    ap.add_argument("--iters", type=int, default=2000); ap.add_argument("--gpu-name", default=None)
    a = ap.parse_args(argv)
    try:
        import torch; gpu = a.gpu_name or (torch.cuda.get_device_name(0) if torch.cuda.is_available() else "sin GPU")
    except Exception:  # noqa: BLE001
        gpu = a.gpu_name or "desconocida"
    runs = {}
    for n_mem in (21000, 3000):
        for dev in ("cuda", "cpu"):
            r = bench(n_mem, dev, a.iters if dev == "cuda" else max(200, a.iters // 5), Path(a.results_dir) / "gates" / f"G7_{dev}_{n_mem}.json")
            runs[f"{dev}@{n_mem}"] = {"p50_ms": r["p50_ms"], "p95_ms": r["p95_ms"], "components_p50": r["components"]}
    main_run = runs["cuda@21000"]; ok = main_run["p50_ms"] < THRESHOLD_MS
    v = {"zone": "A" if ok else "C", "detail": {"p50_ms_cuda_21k": main_run["p50_ms"], "threshold_ms": THRESHOLD_MS, "gpu": gpu, "is_h100": "H100" in gpu}}
    nums = "; ".join(f"{k}: p50 {x['p50_ms']:.2f} ms p95 {x['p95_ms']:.2f} ms" for k, x in runs.items()) + f" · GPU {gpu}"
    return finish("G7", v, {"runs": runs, "machine": {"gpu": gpu, "platform": platform.platform()}}, {"rule": f"p50 < {THRESHOLD_MS} ms (cuda, 21k precedentes, 3 preguntas)"}, numbers=nums)


if __name__ == "__main__":
    raise SystemExit(main())
