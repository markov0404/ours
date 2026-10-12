"""dec-eval: run | curves | audit"""
from __future__ import annotations
import argparse
import json


def main() -> int:
    """Punto de entrada de `dec-eval`."""
    ap = argparse.ArgumentParser(prog="dec-eval"); sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run"); r.add_argument("--dataset", required=True); r.add_argument("--backbone", required=True); r.add_argument("--config", required=True)
    r.add_argument("--n", type=int, default=0); r.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("curves"); c.add_argument("--dataset", required=True); c.add_argument("--backbone", required=True)
    c.add_argument("--configs", nargs="+", default=["B0", "B1", "B2", "B3", "ours1", "ours2"]); c.add_argument("--ns", nargs="+", type=int, default=[0, 50, 200, 1000]); c.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    c.add_argument("--out", default="results", help="raíz de salida (results/ por defecto; otra por backbone, p. ej. results_qwen06 para G9)")
    a = sub.add_parser("audit"); a.add_argument("--dataset", required=True); a.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "run":
        from .ladder import prepare, run_config
        print(json.dumps(run_config(prepare(args.dataset, args.backbone, args.n, args.seed), args.config), indent=2))
    elif args.cmd == "curves":
        from .curves import run
        from pathlib import Path
        print(run(args.dataset, args.backbone, args.configs, args.ns, args.seeds, Path(args.out)))
    else:
        from dec.data.contamination import audit
        from .ladder import pool_and_eval
        pool, ev, _ = pool_and_eval(args.dataset, args.seed)
        rep = audit(pool, ev)
        # spec §5: los duplicados exactos se EXCLUYEN del pool en prepare() (n_excluded); aquí solo se informa
        print(f"solapamientos 8-gramas: {len(rep.overlaps)} · exactos (se excluirán del pool): {len(rep.exact)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
