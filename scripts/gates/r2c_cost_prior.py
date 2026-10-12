"""R2-C (hip-13): ¿corregir el voto de la memoria por la frecuencia de clase y decidir por coste recupera el recall de clases raras?

civil (n=1000, seed 0), preguntas threat e identity_attack, matriz A (perder un positivo cuesta 10, sin escalar). Compara
B0 argmax, Ours-1 argmax/Bayes y Ours-1 con prior argmax/Bayes. Regla pre-registrada: con prior + Bayes, recall >= recall de B0 y
falsas alarmas <= las de B0, y coste realizado < coste del argmax de B0, en AMBAS preguntas.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.gates._util import finish, default_backbone

QUESTIONS = ("civil:threat", "civil:identity_attack")
LOSS = {("no", "yes"): 10.0}   # matriz A: predecir "no" cuando es "yes" cuesta 10


def _stats(preds: list[str], truths: list[str], cm) -> dict:
    from dec.metrics import realized_cost
    tp = sum(p == "yes" and t == "yes" for p, t in zip(preds, truths)); pos = sum(t == "yes" for t in truths)
    fp = sum(p == "yes" and t != "yes" for p, t in zip(preds, truths))
    return {"recall": tp / max(pos, 1), "detected": tp, "positives": pos, "false_pos": fp, "cost": realized_cost(preds, truths, cm)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--results-dir", default="results")
    ap.add_argument("--backbone", default=None); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default="ours1")
    a = ap.parse_args(argv)
    from dec.eval.ladder import prepare, run_config
    from dec.policy import bayes_action
    from dec.schema import CostMatrix
    bb = a.backbone or default_backbone(a.results_dir); cm = CostMatrix(loss=dict(LOSS), c_escalate=None)
    prep = prepare("civil", bb, a.n, a.seed)
    runs = {"ours1": run_config(prep, a.config, return_probs=True)[1], "ours1_prior": run_config(prep, a.config, return_probs=True, prior_correction=True)[1]}
    items = runs["ours1"]["items"]; out: dict[str, dict] = {}; ok_all = True
    for q in QUESTIONS:
        idx = [i for i, it in enumerate(items) if it.question_key == q]
        truths = [items[i].gold for i in idx]; res = {}
        p0 = {i: dict(zip(items[i].option_keys(), map(float, prep.enc[items[i].id].p0))) for i in idx}
        res["B0_argmax"] = _stats([max(p0[i], key=p0[i].get) for i in idx], truths, cm)
        res["B0_bayes"] = _stats([bayes_action(p0[i], cm, allow_escalate=False)[0] for i in idx], truths, cm)
        for name, ex in runs.items():
            P = {i: dict(zip(items[i].option_keys(), map(float, ex["P"][i, :len(items[i].options)]))) for i in idx}
            res[f"{name}_argmax"] = _stats([max(P[i], key=P[i].get) for i in idx], truths, cm)
            res[f"{name}_bayes"] = _stats([bayes_action(P[i], cm, allow_escalate=False)[0] for i in idx], truths, cm)
        b0, best = res["B0_argmax"], res["ours1_prior_bayes"]
        ok = best["recall"] >= b0["recall"] - 1e-9 and best["false_pos"] <= b0["false_pos"] and best["cost"] < b0["cost"]
        res["ok"] = ok; ok_all &= ok; out[q] = res
    v = {"zone": "A" if ok_all else "C", "detail": {q: {"ok": r["ok"], "B0_argmax": r["B0_argmax"], "ours1_prior_bayes": r["ours1_prior_bayes"]} for q, r in out.items()}}
    nums = "; ".join(f"{q.split(':')[1]}: B0 recall {r['B0_argmax']['recall']:.2f} FP {r['B0_argmax']['false_pos']} coste {r['B0_argmax']['cost']:.0f} | prior+bayes recall {r['ours1_prior_bayes']['recall']:.2f} FP {r['ours1_prior_bayes']['false_pos']} coste {r['ours1_prior_bayes']['cost']:.0f}" for q, r in out.items())
    return finish("R2-C", v, out, {"backbone_id": bb, "n": a.n, "seed": a.seed, "matrix": "A: (no|yes)=10, sin escalar"}, numbers=nums)


if __name__ == "__main__":
    raise SystemExit(main())
