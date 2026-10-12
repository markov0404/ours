"""G5: coste. Compara coste realizado de bayes_action vs argmax vs umbral fijo 0.5 con las probabilidades de ours1 y 3 matrices fijas.

Matrices (claves = (acción, verdad)): A = {("no","yes"): 10} perder una amenaza; B = {("yes","no"): 10} falso positivo caro;
C = {("no","yes"): 5} con c_escalate = 1 (solo Bayes puede escalar). Las matrices sobre yes/no solo se evalúan en grupos con opciones {yes, no}.
Grupos: civil:threat, civil:toxicity (pregunta) y plumb routing/judge (familia).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.gates.rules import g5_rule
from scripts.gates._util import finish, default_backbone

MATRICES = {"A": ({("no", "yes"): 10.0}, None), "B": ({("yes", "no"): 10.0}, None), "C": ({("no", "yes"): 5.0}, 1.0)}
GROUPS = {"civil": [("civil:threat", lambda it: it.question_key == "civil:threat"), ("civil:toxicity", lambda it: it.question_key == "civil:toxicity")],
          "plumb": [("plumb:routing", lambda it: it.family == "routing"), ("plumb:judge", lambda it: it.family == "judge")]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--results-dir", default="results")
    ap.add_argument("--backbone", default=None); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    import numpy as np
    from dec.eval.ladder import prepare, run_config  # import perezoso: --help no necesita caches
    from dec.policy import bayes_action
    from dec.metrics import realized_cost
    from dec.schema import CostMatrix
    bb = a.backbone or default_backbone(a.results_dir)
    scenarios: dict[str, dict] = {}; skipped: list[str] = []
    for ds, groups in GROUPS.items():
        _, ex = run_config(prepare(ds, bb, a.n, a.seed), "ours1", return_probs=True)
        for gname, pred in groups:
            sel = [i for i, it in enumerate(ex["items"]) if pred(it)]
            if not sel:
                skipped.append(f"{gname}: sin ítems"); continue
            if any({o.key for o in ex["items"][i].options} != {"yes", "no"} for i in sel):
                skipped.append(f"{gname}: opciones no yes/no (matrices A/B/C omitidas)"); continue
            for mname, (loss, c_esc) in MATRICES.items():
                cm = CostMatrix(loss=dict(loss), c_escalate=c_esc)
                bay, arg, thr_a, truths = [], [], [], []
                for i in sel:
                    it = ex["items"][i]; keys = it.option_keys(); p = dict(zip(keys, map(float, ex["P"][i])))
                    bay.append(bayes_action(p, cm, allow_escalate=c_esc is not None)[0])
                    arg.append(max(p, key=p.get)); thr_a.append("yes" if p["yes"] >= 0.5 else "no"); truths.append(it.gold)
                scenarios[f"{mname}/{gname}"] = {"bayes": realized_cost(bay, truths, cm), "argmax": realized_cost(arg, truths, cm),
                                                 "threshold": realized_cost(thr_a, truths, cm), "n": len(sel)}
    need = [f"{m}/{g}" for g in ("civil:threat", "civil:toxicity") for m in MATRICES]
    if any(k not in scenarios for k in need):
        raise RuntimeError(f"G5 requiere los escenarios civil:threat y civil:toxicity; faltan {[k for k in need if k not in scenarios]}; omitidos: {skipped}")
    v = g5_rule(scenarios)
    nums = "; ".join(f"{k}: bayes {x['bayes']:.1f} argmax {x['argmax']:.1f} umbral {x['threshold']:.1f}" for k, x in scenarios.items())
    return finish("G5", v, scenarios, {"backbone_id": bb, "skipped": skipped}, numbers=nums)


if __name__ == "__main__":
    raise SystemExit(main())
