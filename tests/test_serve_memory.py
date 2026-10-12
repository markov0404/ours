"""Modo memory de dec-serve: mapeo de question_key (CPU) y contrato HTTP vs camino de la escalera (GPU)."""
import json, socket, subprocess, sys, time, urllib.request
from pathlib import Path
import numpy as np
import pytest
from dec.schema import DecisionItem, Option
from dec.serve.memory_pool import map_question_key

ROOT = Path(__file__).resolve().parents[1]


def _it(qtype="choice", instr="color?", qk="serve:choice"):
    opts = [Option("yes"), Option("no")] if qtype == "noul" else [Option("a"), Option("b")]
    return DecisionItem(id="x", dataset="serve", state_id="s", state="e", qtype=qtype, instructions=instr, options=opts, gold=None, question_key=qk)


def test_qkey_mapping_strategies():
    by_instr = {"color?": "syn:color", "forma?": "syn:forma"}; by_type = {"choice": None, "noul": "syn:flag"}
    f = lambda it, strat="instructions", m=None: map_question_key(it, strat, m or {}, by_instr, by_type)
    assert f(_it()) == "syn:color" and f(_it(instr="otra")) is None
    assert f(_it(qtype="noul", instr="zzz"), "type") == "syn:flag" and f(_it(instr="zzz"), "type") is None   # choice ambiguo
    assert f(_it(), "none") is None
    assert f(_it(instr="zzz"), "none", {"serve:choice": "syn:forma"}) == "syn:forma"
    assert f(_it(instr="forma? detalle"), "none", {"forma": "syn:forma", "forma? d": "syn:color"}) == "syn:color"   # prefijo más largo


def test_pool_matches_ladder_on_synthetic():
    pytest.importorskip("faiss")
    from tests.test_ladder_synthetic import _synthetic
    from dec.eval.ladder import run_config
    from dec.serve.memory_pool import MemoryPool
    prep = _synthetic(n_pool=300, n_eval=60)
    pool = MemoryPool("syn", items=prep.labeled, enc=prep.enc, device="cpu", seed=0)
    assert pool.question_keys == ["syn:q"] and pool.map_key(prep.items_eval[0]) == "syn:q"
    out, ex = run_config(prep, "ours1", return_probs=True)
    for j, it in enumerate(prep.items_eval):
        e = prep.enc[it.id]; P = pool.fuse(e.p0, pool.hit(e.h, it.question_key), it.option_keys())
        assert np.allclose(P, ex["P"][j][:len(P)], atol=1e-5)
    assert pool.risk_state.theta == out["risk"]["theta"] or (np.isinf(pool.risk_state.theta) and np.isinf(out["risk"]["theta"]))


# ---- contrato GPU ----

def _free_port() -> int:
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def _get(port: int, path: str = "/health"):
    return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=5).read())


def _post(port: int, body: dict) -> dict:
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(r, timeout=120).read())


def _body(it: DecisionItem) -> dict:
    if it.qtype == "noul":
        crit = {"true": it.options[0].desc, "false": it.options[1].desc}
    elif it.qtype == "score":
        crit = [o.desc for o in it.options]
    else:
        crit = {o.key: o.desc for o in it.options}
    return {"state": it.state, "model": "dec-dev", "questions": {"decision": {"type": it.qtype, "instructions": it.instructions, "criteria": crit,
                                                                          "question_key": it.question_key}}}


def _probs(ans: dict, it: DecisionItem) -> np.ndarray:
    a = ans["answers"]["decision"]
    if it.qtype == "noul":
        return np.array([a["noul"], 1 - a["noul"]])
    return np.array([a["probabilities"][k] for k in it.option_keys()])


@pytest.mark.timeout(2400)
def test_http_memory_reproduces_ladder_path(tmp_path):
    import torch
    if not torch.cuda.is_available():
        pytest.skip("GPU")
    from dec.data import load
    from dec.backbone.model import Backbone
    from dec.serve.memory_pool import MemoryPool
    port = _free_port(); log = (tmp_path / "server.log").open("w")
    def _tail() -> str:
        log.flush(); return "\n".join((tmp_path / "server.log").read_text().splitlines()[-30:])
    srv = subprocess.Popen([sys.executable, "-m", "dec.serve.app", "--backbone", "Qwen/Qwen3-0.6B", "--style", "numeric", "--port", str(port),
                            "--mode", "memory", "--memory-pool", "plumb:train", "--memory-n", "200", "--memory-seed", "0",
                            "--qkey-strategy", "instructions"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    try:
        for _ in range(360):
            time.sleep(5)
            assert srv.poll() is None, "dec-serve murió:\n" + _tail()
            try:
                if _get(port)["ok"]: break
            except Exception:
                pass
        else:
            pytest.fail("dec-serve no arrancó:\n" + _tail())
        h = _get(port); assert h["mode"] == "memory" and h["pool_size"] > 0 and h["question_keys"], h
        # referencia en proceso: mismo pool, mismo backbone, ítem original (question_key propio, sin pasar por el cable)
        bb = Backbone("Qwen/Qwen3-0.6B", dtype="bf16", style="numeric", temperature=1.0)
        pool = MemoryPool(bb.backbone_id, bb=bb, pool_from=("plumb", "train"), n=200, seed=0, qkey_strategy="instructions")
        test = load("plumb", "test")
        mapped = [it for it in test if pool.map_key(it) is not None][:5]
        assert mapped, "ningún ítem de plumb test se mapea a una question_key del pool"
        n_hits = 0
        for it in mapped:
            resp = _post(port, _body(it)); p_srv = _probs(resp, it)
            assert np.all(p_srv >= 0) and abs(p_srv.sum() - 1) < 1e-3, p_srv
            e = bb.encode([it], log_every=10**9)[0]; hit = pool.hit(e.h, pool.map_key(it)); ref = pool.fuse(e.p0, hit, it.option_keys())
            assert resp["usage"]["unmapped"] == 0
            n_hits += resp["usage"]["memory_hits"]
            np.testing.assert_allclose(p_srv, ref, atol=1e-3)
        assert n_hits >= 1
        # sin mapeo: instrucciones desconocidas -> camino solo-backbone, contado en usage y /health
        odd = _body(mapped[0]); odd["questions"]["decision"]["instructions"] = "instrucción inexistente en el pool"; odd["questions"]["decision"].pop("question_key")
        r2 = _post(port, odd); assert r2["usage"]["unmapped"] == 1 and r2["usage"]["memory_hits"] == 0
        h2 = _get(port); assert h2["memory_hits"] == n_hits and h2["unmapped"] == 1
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except subprocess.TimeoutExpired:
            srv.kill(); srv.wait()
        log.close()
