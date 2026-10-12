"""R2-A (hip-11) y R2-B (hip-12) sobre las curvas de la ronda 2 (results_r2/curves/<d>.json, n=1000, 3 semillas).

R2-A: en los 5 dominios, ECE(ours1cal) <= ECE(B1) + 0.005 y acc(ours1cal) >= acc(ours1) - 0.002.
R2-B: acc(ours3) >= acc(B2) en massive, clinc, banking77; acc(ours3) >= acc(ours1) - 0.002 en plumb y civil; ECE(ours3cal) <= 0.06 en massive.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.gates._util import finish

DOMS = ("plumb", "massive", "clinc", "civil", "banking77")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--curves-dir", default="results_r2"); ap.add_argument("--n", type=int, default=1000)
    a = ap.parse_args(argv)
    C = {d: json.loads((Path(a.curves_dir) / "curves" / f"{d}.json").read_text())["agg"] for d in DOMS}
    acc = lambda d, c: C[d][f"{c}@{a.n}"]["acc"][0]; ece = lambda d, c: C[d][f"{c}@{a.n}"]["ece"][0]
    da = {d: {"ece_cal": ece(d, "ours1cal"), "ece_B1": ece(d, "B1"), "acc_cal": acc(d, "ours1cal"), "acc_ours1": acc(d, "ours1"),
              "ok": ece(d, "ours1cal") <= ece(d, "B1") + 0.005 and acc(d, "ours1cal") >= acc(d, "ours1") - 0.002} for d in DOMS}
    va = {"zone": "A" if all(x["ok"] for x in da.values()) else "C", "detail": da}
    finish("R2-A", va, {d: C[d] for d in DOMS}, {"n": a.n},
           numbers="; ".join(f"{d}: ECE {x['ece_cal']:.3f} vs B1 {x['ece_B1']:.3f}, acc {x['acc_cal']:.3f} vs {x['acc_ours1']:.3f}" for d, x in da.items()))
    db = {d: {"acc_ours3": acc(d, "ours3"), "ref": acc(d, "B2") if d in ("massive", "clinc", "banking77") else acc(d, "ours1") - 0.002,
              "ref_name": "B2" if d in ("massive", "clinc", "banking77") else "ours1-0.002", "ece_ours3cal": ece(d, "ours3cal")} for d in DOMS}
    for d, x in db.items():
        x["ok"] = x["acc_ours3"] >= x["ref"] - 1e-9 and (d != "massive" or x["ece_ours3cal"] <= 0.06)
    vb = {"zone": "A" if all(x["ok"] for x in db.values()) else "C", "detail": db}
    return finish("R2-B", vb, {d: C[d] for d in DOMS}, {"n": a.n},
                  numbers="; ".join(f"{d}: ours3 {x['acc_ours3']:.3f} vs {x['ref_name']} {x['ref']:.3f}, ECE ours3cal {x['ece_ours3cal']:.3f}" for d, x in db.items()))


if __name__ == "__main__":
    raise SystemExit(main())
