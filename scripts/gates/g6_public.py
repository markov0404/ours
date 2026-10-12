"""G6: benchmark público. B0 vs ours1 sobre JevBench con memoria poblada desde un pool EXTERNO declarado (banking77 + clinc + massive), modo benchmark.

Antes de evaluar audita contaminación (8-gramas y exactos) entre ese pool y los ítems públicos; aborta si hay solapamiento.
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.gates.rules import g6_rule
from scripts.gates._util import finish, default_backbone

POOL = ["banking77", "clinc", "massive"]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--results-dir", default="results")
    ap.add_argument("--backbone", default=None); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from dec.eval.ladder import prepare, run_config  # import perezoso: --help no necesita caches
    from dec.data.contamination import audit
    bb = a.backbone or default_backbone(a.results_dir)
    prep = prepare("jevbench", bb, 10 ** 9, a.seed, pool_from=POOL)   # n enorme = todo el pool externo como memoria
    rep = audit(prep.items_pool, prep.items_eval)   # informativo: prepare() ya EXCLUYÓ del pool los estados idénticos a eval (n_excluded)
    print(f"audit pool externo: solapamientos 8-gramas {len(rep.overlaps)} · exactos restantes {len(rep.exact)} · excluidos en prepare {prep.n_excluded}", file=sys.stderr)
    b0 = run_config(prep, "B0"); o1, ex = run_config(prep, "ours1", return_probs=True)
    cov = float((ex["mass"] > 0).mean()); mass_mean = float(ex["mass"].mean()); trivial = cov < 0.05
    pick = lambda r: {"acc": r["acc"], "ece": r["ece"], "brier": r["brier"]}
    v = g6_rule(pick(o1), pick(b0))
    text = f"G6 corrido: zona {v['zone']} TRIVIAL (ours1 ≈ B0 + T externa; cobertura de memoria {cov:.1%})" if trivial else f"G6 corrido: zona {v['zone']}"
    nums = f"B0 acc {b0['acc']:.3f} ece {b0['ece']:.3f}; ours1 acc {o1['acc']:.3f} ece {o1['ece']:.3f}; cobertura {cov:.1%}; masa media {mass_mean:.3f}; pool {len(prep.items_pool)}"
    return finish("G6", v, {"B0": pick(b0), "ours1": pick(o1)},
                  {"backbone_id": bb, "external_pool": POOL, "n_pool": len(prep.items_pool), "pool_size": len(prep.items_pool), "memory_coverage": cov,
                   "mass_mean": mass_mean, "trivial": trivial, "audit": {"overlaps": len(rep.overlaps), "exact": len(rep.exact), "excluded_in_prepare": prep.n_excluded}}, text=text, numbers=nums)


if __name__ == "__main__":
    raise SystemExit(main())
