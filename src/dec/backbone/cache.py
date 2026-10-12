"""Cache parquet de Encoded por dataset/split/backbone."""
from __future__ import annotations
import os
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
from dec.schema import Encoded

CACHE = Path("cache/enc")

def cache_path(dataset: str, split: str, backbone_id: str) -> Path:
    safe = backbone_id.replace("/", "_").replace("@", "_").replace(":", "_")
    return CACHE / f"{dataset}_{split}__{safe}.parquet"

def write_cache(path: Path, encoded: list[Encoded], backbone_id: str) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "item_id": [e.item_id for e in encoded],
        "p0": [e.p0.astype(np.float32).tolist() for e in encoded],
        "logits": [e.logits.astype(np.float32).tolist() for e in encoded],
        "h": [e.h.astype(np.float16).tobytes() for e in encoded],
        "h_last": [e.h_last.astype(np.float16).tobytes() for e in encoded],
        "d": [int(e.h.shape[0]) for e in encoded],
        "readout": [e.readout for e in encoded],
        "truncated": [e.truncated for e in encoded],
        "backbone_id": [backbone_id] * len(encoded),
    })
    tmp = path.with_name(path.name + ".tmp")   # escritura atómica: un corte a mitad no destruye los trozos ya cacheados
    pq.write_table(table, tmp); os.replace(tmp, path)

def read_cache(path: Path) -> dict[str, Encoded]:
    t = pq.read_table(path).to_pydict(); out = {}
    for i in range(len(t["item_id"])):
        d = t["d"][i]
        out[t["item_id"][i]] = Encoded(t["item_id"][i], np.array(t["p0"][i], np.float32), np.array(t["logits"][i], np.float32),
            np.frombuffer(t["h"][i], np.float16).astype(np.float32)[:d], np.frombuffer(t["h_last"][i], np.float16).astype(np.float32)[:d],
            t["readout"][i], bool(t["truncated"][i]))
    return out
