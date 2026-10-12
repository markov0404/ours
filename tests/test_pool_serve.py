"""Producción local: dec-pool build sobre examples/history_example.jsonl y dec-serve --pool-dir (CPU; variante GPU si hay CUDA)."""
import json, os, socket, subprocess, sys, time, urllib.request
from pathlib import Path
import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "history_example.jsonl"
MID = "Qwen/Qwen3-0.6B"
pytestmark = pytest.mark.slow
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU"))]


def _env(device: str) -> dict:
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""   # CPU de verdad: ni el backbone ni MiniLM pueden tocar la GPU
    return env


def _pool_cmd(*args) -> list[str]:
    return [sys.executable, "-m", "dec.pool", *args]


@pytest.fixture(scope="module", params=DEVICES)
def pool(request, tmp_path_factory):
    dev = request.param; out = tmp_path_factory.mktemp(f"pool_{dev}") / "pool"
    r = subprocess.run(_pool_cmd("build", "--history", str(EXAMPLE), "--backbone", MID, "--style", "numeric", "--T", "1.0",
                                 "--out", str(out), "--device", dev), capture_output=True, text=True, cwd=ROOT, env=_env(dev), timeout=1500)
    assert r.returncode == 0, r.stdout + r.stderr
    return dev, out, r.stdout


def test_build_outputs_and_resume(pool):
    dev, out, stdout = pool
    assert {p.name for p in out.iterdir()} >= {"items.jsonl", "enc.parquet", "minilm.npy", "pool.json"}
    meta = json.loads((out / "pool.json").read_text())
    assert meta["n_items"] == 18 and meta["features"] == "minilm" and meta["config"] == "ours3cal"
    assert meta["dtype"] == ("fp32" if dev == "cpu" else "bf16") and meta["backbone"] == MID and meta["backbone_id"].endswith(meta["dtype"])
    assert set(meta["question_keys"]) == {"support:priority", "support:escalate"} and meta["dec_version"]
    assert np.load(out / "minilm.npy").shape == (18, 384)
    assert "por codificar 18" in stdout
    r = subprocess.run(_pool_cmd("build", "--history", str(EXAMPLE), "--backbone", MID, "--style", "numeric", "--T", "1.0",
                                 "--out", str(out), "--device", dev), capture_output=True, text=True, cwd=ROOT, env=_env(dev), timeout=600)
    assert r.returncode == 0 and "por codificar 0" in r.stdout, r.stdout + r.stderr     # reanudable: nada que codificar
    r = subprocess.run(_pool_cmd("info", str(out)), capture_output=True, text=True, cwd=ROOT, env=_env(dev))
    assert r.returncode == 0 and "AVISO support:priority" in r.stdout and "< 200" in r.stdout


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _get(port: int) -> dict:
    return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5).read())


def _post(port: int, body: dict) -> dict:
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=300).read())


def _requests() -> list[dict]:
    rows = [json.loads(l) for l in EXAMPLE.read_text().splitlines()]
    pri = next(r for r in rows if r["question"]["question_key"] == "support:priority")["question"]
    esc = next(r for r in rows if r["question"]["question_key"] == "support:escalate")["question"]
    odd = {**pri, "question_key": "support:unknown", "instructions": "Pick a label not in the pool."}
    st = ["Checkout is failing with error 500 for every customer since the deploy an hour ago.",
          "The help-center search returns nothing for any query; customers can still open tickets by email.",
          "Where can I download last month's invoice?"]
    return [{"state": st[0], "model": "dec", "questions": {"decision": pri}},
            {"state": st[1], "model": "dec", "questions": {"decision": esc}},
            {"state": st[2], "model": "dec", "questions": {"decision": odd}}]


@pytest.mark.timeout(1800)
def test_serve_pool_dir(pool, tmp_path):
    dev, out, _ = pool
    port = _free_port(); logf = (tmp_path / "server.log").open("w")
    def _tail() -> str:
        logf.flush(); return "\n".join((tmp_path / "server.log").read_text().splitlines()[-40:])
    srv = subprocess.Popen([sys.executable, "-m", "dec.serve.app", "--pool-dir", str(out), "--device", dev, "--port", str(port)],
                           cwd=ROOT, env=_env(dev), stdout=logf, stderr=subprocess.STDOUT)
    try:
        for _ in range(240):
            time.sleep(2)
            assert srv.poll() is None, "dec-serve murió:\n" + _tail()
            try:
                if _get(port)["ok"]: break
            except Exception:
                pass
        else:
            pytest.fail("dec-serve no arrancó:\n" + _tail())
        h = _get(port)
        assert h["mode"] == "memory" and h["pool_size"] == 18 and h["pool_dir"] == str(out) and h["config"] == "ours3cal", h
        assert h["features"] == "minilm" and h["post_temperature"] is True and h["device"] == dev
        assert sorted(h["question_keys"]) == ["support:escalate", "support:priority"]
        hits = unm = 0; lat = []
        for body in _requests():
            t0 = time.time(); resp = _post(port, body); lat.append(time.time() - t0)
            a = resp["answers"]["decision"]; q = body["questions"]["decision"]
            assert a["type"] == q["type"]
            if q["type"] == "noul":
                assert 0.0 <= a["noul"] <= 1.0
            else:
                p = np.array(list(a["probabilities"].values()))
                assert set(a["probabilities"]) == set(q["criteria"]) and np.all(p >= 0) and abs(p.sum() - 1) < 1e-3, a
                assert a["choice"] in q["criteria"]
            hits += resp["usage"]["memory_hits"]; unm += resp["usage"]["unmapped"]
        assert hits >= 1 and unm == 1
        h2 = _get(port); assert h2["memory_hits"] == hits and h2["unmapped"] == 1 and h2["requests"] == 3
        print(f"\nlatencia {dev} (Qwen3-0.6B, s/petición): " + ", ".join(f"{x:.2f}" for x in lat))
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill(); srv.wait()
        logf.close()
