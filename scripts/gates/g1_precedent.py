"""G1: precedente. Lee results/curves/{plumb,massive,clinc}.json (de `dec-eval curves`)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.gates.rules import g1_rule
from scripts.gates._util import finish


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--backbone", default=None, help="solo informativo; las curvas ya fijan el backbone")
    a = ap.parse_args(argv)
    curves = {d: json.loads((Path(a.results_dir) / "curves" / f"{d}.json").read_text()) for d in ("plumb", "massive", "clinc")}
    v = g1_rule(curves, a.n)
    return finish("G1", v, {d: c["agg"] for d, c in curves.items()}, {"backbone_id": {d: c.get("backbone_id") for d, c in curves.items()}},
                  numbers="; ".join(f"{d}: ours1 acc {c['agg'][f'ours1@{a.n}']['acc'][0]:.3f} ece {c['agg'][f'ours1@{a.n}']['ece'][0]:.3f} vs B1 acc {c['agg'][f'B1@{a.n}']['acc'][0]:.3f} ece {c['agg'][f'B1@{a.n}']['ece'][0]:.3f}" for d, c in curves.items()))


if __name__ == "__main__":
    raise SystemExit(main())
