"""G3: coherencia. Lee results/ladder/{massive,civil}/<config>_n1000_s0.json (ours1, o B3 si G1 cayó en zona C)."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.gates.rules import g3_rule
from scripts.gates._util import finish


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--results-dir", default="results")
    ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--config", default=None, help="por defecto ours1, o B3 si results/gates/G1.json tiene zona C")
    ap.add_argument("--backbone", default=None, help="solo informativo; los ladder ya fijan el backbone")
    a = ap.parse_args(argv)
    cfg = a.config
    if cfg is None:
        g1 = Path(a.results_dir) / "gates" / "G1.json"
        cfg = "B3" if g1.exists() and json.loads(g1.read_text()).get("zone") == "C" else "ours1"
    res = {d: json.loads((Path(a.results_dir) / "ladder" / d / f"{cfg}_n{a.n}_s{a.seed}.json").read_text()) for d in ("massive", "civil")}
    v = g3_rule(res)
    return finish("G3", v, {d: {"acc": r["acc"], "coherence": r["coherence"]} for d, r in res.items()}, {"config": cfg},
                  numbers="; ".join((f"{d}: incoherencia {r['coherence']['incoherence_before']:.3f}->{r['coherence']['incoherence_after']:.3f}, acc {r['acc']:.3f}->{r['coherence']['acc_after']:.3f}"
                                     if r.get("coherence") else f"{d}: sin restricciones (no evaluable)") for d, r in res.items()))


if __name__ == "__main__":
    raise SystemExit(main())
