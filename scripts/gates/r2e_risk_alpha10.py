"""R2-E (hip-15, informativo): ¿el umbral selectivo automatiza >= 50 % de clinc con riesgo realizado <= 0.10? No modifica G4."""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.gates._util import finish, default_backbone


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--results-dir", default="results")
    ap.add_argument("--backbone", default=None); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--config", default="ours1"); ap.add_argument("--alpha", type=float, default=0.10)
    a = ap.parse_args(argv)
    from dec.eval.ladder import prepare, run_config
    bb = a.backbone or default_backbone(a.results_dir)
    per = []
    for s in a.seeds:
        r = run_config(prepare("clinc", bb, a.n, s), a.config, alpha=a.alpha)["risk"]; per.append({"seed": s, **r})
    ok = all((x["automation"] >= 0.5) and (x["realized_risk"] == x["realized_risk"]) and (x["realized_risk"] <= a.alpha) for x in per)
    v = {"zone": "A" if ok else "C", "detail": {"alpha": a.alpha, "config": a.config, "per_seed": [{k: x[k] for k in ("seed", "theta", "automation", "realized_risk", "bound")} for x in per]}}
    nums = "; ".join(f"s{x['seed']}: aut {x['automation']:.3f} riesgo {x['realized_risk']:.4f} theta {x['theta']:.3f}" for x in per)
    return finish("R2-E", v, {"per_seed": per}, {"backbone_id": bb, "n": a.n}, numbers=nums)


if __name__ == "__main__":
    raise SystemExit(main())
