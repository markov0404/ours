"""Contrato: el harness real de JevBench (adaptador typesafe) contra dec-serve."""
import json, os, socket, subprocess, sys, time, urllib.request
from pathlib import Path
import pytest, torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
ROOT = Path(__file__).resolve().parents[1]
JEV = ROOT / "vendor" / "jevbench"


def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _up(port: int) -> bool:
    try:
        return urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2).status == 200
    except Exception:
        return False


@pytest.mark.timeout(1200)
def test_jevbench_harness_runs_against_dec_serve(tmp_path):
    if not JEV.exists():
        pytest.skip("vendor/jevbench ausente")
    mini = ROOT / "tests/fixtures/jevbench_mini.jsonl"  # fixture congelado: no se modifica
    assert len(mini.read_text().splitlines()) == 3
    port = _free_port()
    errlog = tmp_path / "server.log"; errf = errlog.open("w")
    def _tail() -> str:
        errf.flush(); return "\n".join(errlog.read_text().splitlines()[-30:])
    srv = subprocess.Popen([sys.executable, "-m", "dec.serve.app", "--backbone", "Qwen/Qwen3-0.6B", "--style", "numeric",
                            "--port", str(port), "--mode", "benchmark"], cwd=ROOT, stdout=errf, stderr=subprocess.STDOUT)
    try:
        for _ in range(120):
            time.sleep(5)
            assert srv.poll() is None, "dec-serve murió:\n" + _tail()
            if _up(port): break
        else:
            pytest.fail("dec-serve no arrancó:\n" + _tail())
        run = tmp_path / "RUN"; run.mkdir()
        env = {**os.environ, "TYPESAFE_API_KEY": "dummy"}
        r = subprocess.run([sys.executable, "-m", "jevbench.cli", "run", "--tasks", str(mini), "--adapter", "typesafe",
                            "--endpoint", f"http://127.0.0.1:{port}", "--key-env", "TYPESAFE_API_KEY", "--model", "dec-dev",
                            "--price-in-per-m", "0", "--price-out-per-m", "0", "--results", str(run / "results.jsonl"),
                            "--raw-dir", str(run / "raw"), "--ledger", str(run / "ledger.jsonl"), "--cap-usd", "1",
                            "--manifest", str(run / "manifest.json")], cwd=JEV, env=env, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout + r.stderr
        rows = [json.loads(l) for l in (run / "results.jsonl").read_text().splitlines()]
        assert len(rows) == 3 and all(not row.get("error") for row in rows), rows
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill(); srv.wait()
        errf.close()
