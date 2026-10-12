"""Curvas n × seeds × configs -> results/curves/<dataset>.json"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from .ladder import prepare, run_config


def run(dataset: str, backbone_id: str, configs: list[str], ns: list[int], seeds: list[int], out_dir: Path = Path("results")) -> Path:
    """Corre cada config para cada (n, seed) y escribe filas y agregados (media, std) en results/curves/<dataset>.json."""
    rows = []
    for n in ns:
        for s in seeds:
            prep = prepare(dataset, backbone_id, n, s)
            for c in configs:
                if n == 0 and c != "B0":
                    continue
                r = run_config(prep, c); r["n_requested"] = n; rows.append(r)
                (out_dir / "ladder" / dataset).mkdir(parents=True, exist_ok=True)
                (out_dir / "ladder" / dataset / f"{c}_n{n}_s{s}.json").write_text(json.dumps(r, indent=2))
    agg = {}
    for c in configs:
        for n in ns:
            sel = [r for r in rows if r["config"] == c and r["n_requested"] == n]
            if sel:
                agg[f"{c}@{n}"] = {k: (float(np.mean([r[k] for r in sel])), float(np.std([r[k] for r in sel]))) for k in ("acc", "ece", "brier")}
    p = out_dir / "curves" / f"{dataset}.json"; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"dataset": dataset, "backbone_id": backbone_id, "rows": rows, "agg": agg}, indent=2)); return p
