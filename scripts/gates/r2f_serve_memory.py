"""R2-F (hip-16): dec-serve en modo memoria reproduce la escalera por HTTP.

Levanta dec-serve con JevK5 en modo benchmark y en modo memoria (pool plumb:train, n=1000, seed 0), envía los 131 ítems de plumb test
por HTTP con su `question_key` declarada (extensión dec del contrato; el harness de JevBench no la envía y plumb no es formato JevBench,
así que se usa un cliente HTTP propio con el mismo cuerpo que el adaptador typesafe) y compara con run_config(ours1, n=1000, seed 0)
restringido a plumb test. Regla: |acc_http - acc_ladder| <= 0.01, 0 errores de cable, p50(memoria) <= 1.3 x p50(benchmark).
"""
from __future__ import annotations
import argparse, json, subprocess, sys, time, urllib.request
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.gates._util import finish, default_backbone


def _post(port, body):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json", "Authorization": "Bearer x"})
    t = time.perf_counter(); r = json.loads(urllib.request.urlopen(req, timeout=120).read()); return r, time.perf_counter() - t


def _body(it):
    crit = {"true": it.options[0].desc, "false": it.options[1].desc} if it.qtype == "noul" else ([o.desc for o in it.options] if it.qtype == "score" else {o.key: o.desc for o in it.options})
    return {"state": it.state, "model": "dec", "questions": {"decision": {"type": it.qtype, "instructions": it.instructions, "criteria": crit, "question_key": it.question_key}}}


def _pred(resp, it):
    a = resp["answers"]["decision"]
    if it.qtype == "noul":
        return "yes" if float(a["noul"]) >= 0.5 else "no"
    pr = a["probabilities"]; return max(pr, key=pr.get)


def _serve(args, port, log):
    p = subprocess.Popen([sys.executable, "-m", "dec.serve.app", *args, "--port", str(port)], stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2); return p
        except Exception:  # noqa: BLE001
            if p.poll() is not None: raise RuntimeError("dec-serve murió")
            time.sleep(5)
    p.kill(); raise RuntimeError("dec-serve no arrancó")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--results-dir", default="results"); ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args(argv)
    from dec.data import load
    from dec.eval.ladder import prepare, run_config
    bb = default_backbone(a.results_dir); test = load("plumb", "test"); tid = {it.id for it in test}
    out, ex = run_config(prepare("plumb", bb, a.n, 0), "ours1", return_probs=True)
    idx = [i for i, it in enumerate(ex["items"]) if it.id in tid]
    acc_ladder = float(np.mean([ex["P"][i].argmax() == ex["G"][i] for i in idx]))
    base = ["--backbone", "alibiserikbay/JevK5", "--style", "jevk5", "--T", "1.22", "--dtype", "bf16"]
    res = {}; Path("results/gates").mkdir(parents=True, exist_ok=True)
    for mode, extra, port in (("benchmark", ["--mode", "benchmark"], 8093),
                              ("memory", ["--mode", "memory", "--memory-pool", "plumb:train", "--memory-n", str(a.n), "--memory-seed", "0"], 8094)):
        log = open(f"results/gates/R2-F_{mode}.log", "w"); srv = _serve(base + extra, port, log)
        try:
            preds, lat, errs, hits = [], [], 0, 0
            for it in test:
                try:
                    r, dt = _post(port, _body(it)); preds.append(_pred(r, it) == it.gold); lat.append(dt); hits += r.get("usage", {}).get("memory_hits", 0)
                except Exception:  # noqa: BLE001
                    errs += 1; preds.append(False)
            res[mode] = {"acc": float(np.mean(preds)), "p50_s": float(np.median(lat)) if lat else None, "errors": errs, "memory_hits": hits}
        finally:
            srv.terminate()
            try: srv.wait(timeout=30)
            except subprocess.TimeoutExpired: srv.kill()
            log.close()
    m, b = res["memory"], res["benchmark"]
    ok = abs(m["acc"] - acc_ladder) <= 0.01 and m["errors"] == 0 and m["p50_s"] <= 1.3 * b["p50_s"]
    v = {"zone": "A" if ok else "C", "detail": {"acc_http_memory": m["acc"], "acc_ladder_ours1": acc_ladder, "acc_http_benchmark": b["acc"],
                                               "p50_memory_s": m["p50_s"], "p50_benchmark_s": b["p50_s"], "errors": m["errors"], "memory_hits": m["memory_hits"]}}
    return finish("R2-F", v, res, {"backbone_id": bb, "n": a.n, "n_test": len(test), "client": "HTTP propio con question_key declarada"},
                  numbers=f"acc HTTP memoria {m['acc']:.3f} vs escalera {acc_ladder:.3f} (benchmark {b['acc']:.3f}); p50 {m['p50_s']*1e3:.0f} vs {b['p50_s']*1e3:.0f} ms; errores {m['errors']}")


if __name__ == "__main__":
    raise SystemExit(main())
