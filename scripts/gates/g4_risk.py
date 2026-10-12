"""G4: riesgo selectivo y OOS. Corre ours1 sobre clinc con alpha en {0.02, 0.05} y varias semillas (requiere caches)."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.gates.rules import g4_rule
from scripts.gates._util import finish, default_backbone

ALPHAS = (0.02, 0.05)


def _mean(xs: list[float]) -> float:
    """Media ignorando NaN (riesgo indefinido si no se automatizó nada); NaN si todo es NaN."""
    v = [x for x in xs if x == x]
    return float(np.mean(v)) if v else float("nan")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--results-dir", default="results")
    ap.add_argument("--backbone", default=None); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    a = ap.parse_args(argv)
    from dec.eval.ladder import prepare, run_config  # import perezoso: --help no necesita caches
    bb = a.backbone or default_backbone(a.results_dir)
    runs: dict[float, list[dict]] = {al: [] for al in ALPHAS}; oos: list[dict] = []
    for s in a.seeds:
        prep = prepare("clinc", bb, a.n, s)
        for al in ALPHAS:
            r = run_config(prep, "ours1", alpha=al); runs[al].append(r["risk"])
            if al == ALPHAS[-1]:
                oos.append(r["oos"])
    bad = {str(al): [s for s, x in zip(a.seeds, rs) if x["insufficient"] or x["realized_risk"] != x["realized_risk"]] for al, rs in runs.items()}
    bad = {k: v for k, v in bad.items() if v}
    per_seed = {str(al): [{"seed": s, **x} for s, x in zip(a.seeds, rs)] for al, rs in runs.items()}
    by_alpha = {al: {"risk": {"realized_risk": _mean([x["realized_risk"] for x in rs]), "automation": _mean([x["automation"] for x in rs]),
                              "insufficient": [x["insufficient"] for x in rs]}} for al, rs in runs.items()}
    oos_m = {k: _mean([o[k] for o in oos]) for k in ("auroc_conf", "auroc_conf_mass")}
    v = g4_rule(by_alpha, oos_m)
    if bad:
        # theta=inf con insufficient=False significa "sin umbral factible" (ninguna cobertura de la rejilla cumple la cota), no falta de datos
        infeasible = all(not x["insufficient"] for rs in runs.values() for x in rs)
        v = {"zone": "C", "detail": {**v["detail"], "insufficient_seeds": bad,
                                     "reason": "sin umbral factible: riesgo ≤ α no alcanzable con la confianza 1−δ" if infeasible else "calibración insuficiente"}}
    nums = "; ".join(f"alpha {al}: riesgo {x['risk']['realized_risk']:.4f} aut {x['risk']['automation']:.3f}" for al, x in by_alpha.items()) \
        + f"; AUROC conf {oos_m['auroc_conf']:.3f} conf+masa {oos_m['auroc_conf_mass']:.3f}"
    return finish("G4", v, {"by_alpha": {str(k): x for k, x in by_alpha.items()}, "oos": oos_m, "per_seed": per_seed}, {"backbone_id": bb, "seeds": a.seeds, "n": a.n}, numbers=nums)


if __name__ == "__main__":
    raise SystemExit(main())
