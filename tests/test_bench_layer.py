"""El microbenchmark de latencia corre rápido y escribe el JSON esperado."""
import json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_bench_layer_corre_y_escribe_json(tmp_path):
    out = tmp_path / "g7.json"
    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "bench_layer.py"), "--d", "256", "--n-mem", "5000", "--iters", "50", "--out", str(out)],
        capture_output=True, text=True, cwd=ROOT, env={**__import__("os").environ, "PYTHONPATH": str(ROOT / "src")},
    )
    assert r.returncode == 0, r.stderr
    assert time.perf_counter() - t0 < 60
    res = json.loads(out.read_text())
    assert res["p50_ms"] > 0 and res["p95_ms"] >= res["p50_ms"]
    for k in ("phi_ms_p50", "memory_ms_p50", "fusion_ms_p50", "coherence_ms_p50", "router_policy_ms_p50"):
        assert k in res["components"]
    assert res["n_mem"] == 5000 and res["iters"] == 50
