"""G2: métrica por pregunta (phi). Lee results/curves/{plumb,civil,massive,clinc}.json y compara ours2 con ours1."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.gates.rules import g2_rule
from scripts.gates._util import finish


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--backbone", default=None, help="solo informativo; las curvas ya fijan el backbone")
    a = ap.parse_args(argv)
    curves = {d: json.loads((Path(a.results_dir) / "curves" / f"{d}.json").read_text()) for d in ("plumb", "civil", "massive", "clinc")}
    v = g2_rule(curves, a.n)
    return finish("G2", v, {d: c["agg"] for d, c in curves.items()}, {"backbone_id": {d: c.get("backbone_id") for d, c in curves.items()}},
                  numbers="; ".join(f"{d}: ganancia ours2-ours1 {g:+.3f}" for d, g in v["detail"].items()))


if __name__ == "__main__":
    raise SystemExit(main())
