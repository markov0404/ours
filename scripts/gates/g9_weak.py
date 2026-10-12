"""G9 (borde): ¿la memoria ayuda más cuanto más débil es el backbone? Compara Δ(acc) a n=1000 de la config de memoria sobre MiniLM
(B2) entre un backbone débil (Qwen3-0.6B, curvas en --weak-dir) y el elegido en G0 (curvas en --results-dir), en plumb/massive/clinc."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.gates.rules import g9_rule
from scripts.gates._util import finish

DOMAINS = ("plumb", "massive", "clinc")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--results-dir", default="results")
    ap.add_argument("--weak-dir", default="results_qwen06"); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--config", default="B2")
    a = ap.parse_args(argv)
    load = lambda root: {d: json.loads((Path(root) / "curves" / f"{d}.json").read_text()) for d in DOMAINS if (Path(root) / "curves" / f"{d}.json").exists()}
    weak, strong = load(a.weak_dir), load(a.results_dir)
    v = g9_rule(weak, strong, a.n, a.config)
    nums = "; ".join(f"{d}: Δdébil {x['delta_weak']:+.3f} vs Δfuerte {x['delta_strong']:+.3f}" for d, x in v["detail"].items()) or "sin dominios comparables"
    return finish("G9", v, {"weak": {d: c["agg"] for d, c in weak.items()}, "strong": {d: c["agg"] for d, c in strong.items()}},
                  {"config": a.config, "n": a.n, "weak_backbone": {d: c.get("backbone_id") for d, c in weak.items()},
                   "strong_backbone": {d: c.get("backbone_id") for d, c in strong.items()}}, numbers=nums)


if __name__ == "__main__":
    raise SystemExit(main())
