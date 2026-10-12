"""Microbenchmark de latencia de la capa de decisión en CPU (G7 local).

Camino completo por decisión (3 preguntas): phi.mask -> memory.search -> p_mem/fuse/reason_code
-> joint_decode -> gap_signals/route -> decide. n_mem es el total de precedentes, repartido entre las 3 preguntas.
"""
from __future__ import annotations
import argparse, json, os, platform, time
from pathlib import Path
import numpy as np
from dec.memory import PrecedentMemory
from dec.phi import GlobalDiagonalMetric, PhiMask
from dec.fusion import FusionParams, fuse, reason_code, p_mem_vector
from dec.coherence import hierarchy, joint_decode
from dec.router import gap_signals, route, RouterThresholds
from dec.policy import decide, RiskState, LoopDetector
from dec.schema import DecisionItem, Option, CostMatrix

NQ_LABELS = 5
COMPONENTS = ("phi", "memory", "fusion", "coherence", "router_policy")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=2560)
    ap.add_argument("--n-mem", type=int, default=200000)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--questions", type=int, default=3)
    ap.add_argument("--out", default="results/gates/G7_local.json")
    ap.add_argument("--device", default="cpu", help="cpu | cuda: dispositivo de la búsqueda de precedentes")
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    d, nq = a.d, a.questions
    d_q, J = 384, min(256, a.d)
    qkeys = [f"q{i}" for i in range(nq)]
    labels = {q: [f"{q}_l{j}" for j in range(NQ_LABELS)] for q in qkeys}

    mem = PrecedentMemory(d, k=a.k, candidates=200, device=a.device)
    per = a.n_mem // nq
    for q in qkeys:
        H = rng.standard_normal((per, d)).astype(np.float32)
        lab = [labels[q][j] for j in rng.integers(0, NQ_LABELS, per)]
        mem.add_many(H, lab, [q] * per, [f"{q}_{i}" for i in range(per)])
        del H
    mem.build()
    gmask = GlobalDiagonalMetric(d).mask()
    phi = PhiMask(d, d_q=d_q, d_p=128, J=J)
    Q = rng.standard_normal((nq, d_q)).astype(np.float32)
    params = FusionParams(T=1.0, a=1.0, b=0.0, weak_mass=0.05, strong_mass=0.5)
    thr, risk, costs = RouterThresholds(), RiskState(theta=0.5, alpha=0.1, bound=0.2), CostMatrix(c_escalate=1.0)
    cons = [hierarchy(qkeys[1], qkeys[2], {labels[qkeys[1]][j]: labels[qkeys[2]][j % NQ_LABELS] for j in range(NQ_LABELS)})] if nq >= 3 else []

    def item(i: int, q: str) -> DecisionItem:
        return DecisionItem(f"it{i}_{q}", "synthetic", f"s{i}", "El paciente presenta fiebre y tos desde hace tres dias.", "choice",
                            "Cual es el diagnostico mas probable segun el cuadro clinico?", [Option(l) for l in labels[q]], None, q)

    # calentamiento (FAISS/torch)
    gm = gmask
    mem.search(rng.standard_normal(d).astype(np.float32), qkeys[0], gm, a.k)

    tot, comp = [], {c: [] for c in COMPONENTS}
    loop = LoopDetector()
    for i in range(a.iters):
        hs = rng.standard_normal((nq, d)).astype(np.float32)
        p0s = rng.dirichlet(np.ones(NQ_LABELS), nq).astype(np.float32)
        items = [item(i, q) for q in qkeys]
        t = dict.fromkeys(COMPONENTS, 0.0)
        T0 = time.perf_counter()
        hits, fused, reasons, masks = [], [], [], []
        for j, q in enumerate(qkeys):
            s = time.perf_counter(); m = phi.mask(Q[j]); t["phi"] += time.perf_counter() - s
            s = time.perf_counter(); hit = mem.search(hs[j], q, m, a.k); t["memory"] += time.perf_counter() - s
            s = time.perf_counter()
            pm = p_mem_vector(hit, items[j].option_keys())
            f = fuse(p0s[j], pm, hit.mass, params); rc = reason_code(p0s[j], pm, hit.mass, params)
            t["fusion"] += time.perf_counter() - s
            hits.append(hit); fused.append(f); reasons.append(rc); masks.append(m)
        s = time.perf_counter()
        probs_by_q = {q: {k: float(v) for k, v in zip(items[j].option_keys(), fused[j])} for j, q in enumerate(qkeys)}
        _, info = joint_decode(probs_by_q, cons)
        t["coherence"] += time.perf_counter() - s
        s = time.perf_counter()
        for j in range(nq):
            g = gap_signals(items[j], fused[j], hits[j].mass)
            v = route(g, costs, thr, bool(cons))
            decide(items[j], p0s[j], hits[j], fused[j], v, risk, costs, "escalation", masks[j], info, reasons[j], loop)
        t["router_policy"] += time.perf_counter() - s
        tot.append(time.perf_counter() - T0)
        for c in COMPONENTS: comp[c].append(t[c])

    ms = lambda x, p: float(np.percentile(np.asarray(x) * 1e3, p))
    res = {
        "p50_ms": ms(tot, 50), "p95_ms": ms(tot, 95), "mean_ms": float(np.mean(tot) * 1e3),
        "components": {f"{c}_ms_p50": ms(comp[c], 50) for c in COMPONENTS},
        "components_p95": {f"{c}_ms_p95": ms(comp[c], 95) for c in COMPONENTS},
        "n_mem": a.n_mem, "d": d, "k": a.k, "iters": a.iters, "questions": nq, "candidates": mem.candidates, "device": a.device,
        "machine": {"processor": platform.processor(), "platform": platform.platform(), "cpu_count": os.cpu_count()},
    }
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=2))
    print(f"{'componente':<16}{'p50 ms':>10}{'p95 ms':>10}")
    for c in COMPONENTS: print(f"{c:<16}{ms(comp[c], 50):>10.3f}{ms(comp[c], 95):>10.3f}")
    print(f"{'TOTAL':<16}{res['p50_ms']:>10.3f}{res['p95_ms']:>10.3f}   (n_mem={a.n_mem}, d={d}, k={a.k}, p50<2ms: {res['p50_ms'] < 2.0})")


if __name__ == "__main__":
    main()
