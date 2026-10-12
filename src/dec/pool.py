"""dec-pool: pool de precedentes a partir del historial de decisiones del propio cliente.

Historial (JSONL, una decisión por línea; mismo esquema de pregunta que acepta dec-serve + la decisión correcta):

    {"id": "t-001", "state": "...", "question": {"type": "choice", "instructions": "...",
     "criteria": {"low": "...", "high": "..."}, "question_key": "support:priority"}, "decision": "high"}

- `type`: `noul` (criteria {"true", "false"}; decision "yes"/"no" o true/false), `choice` (criteria {opción: descripción}
  o lista de opciones; decision = clave de la opción) o `score` (criteria = lista de niveles; decision = índice del nivel).
- `question_key` es obligatoria: la memoria se particiona por ella y las peticiones la declaran para usar el pool.
- `id` es opcional (si falta se deriva de un hash del contenido); debe ser única.

Directorio del pool (`--out`):
    items.jsonl     DecisionItems normalizados (orden del historial)
    enc.parquet     codificaciones del backbone (`dec.backbone.cache.write_cache`); reanudable: se saltean los ids presentes
    minilm.npy      embeddings MiniLM de los estados, alineados con items.jsonl (features=minilm)
    pool.json       backbone, estilo, T, dtype, features, config, conteos por question_key, fecha y versión de dec
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from dec.schema import DecisionItem, Encoded, state_text

POOL_FORMAT = 1
MIN_PER_KEY = 200     # README §4: con menos la memoria puede empeorar
MIN_PER_CLASS = 50    # README §4: clases raras
CONFIG_BY_FEATURES = {"minilm": "ours3cal", "h": "ours1cal"}


class HistoryError(ValueError):
    """Historial inválido; el mensaje lista los errores con su número de línea."""


def dec_version() -> str:
    try:
        from importlib.metadata import version
        return version("dec")
    except Exception:
        return "desconocida"


def _gold(qtype: str, decision, keys: list[str]) -> str:
    """Normaliza `decision` a la clave de opción del DecisionItem."""
    if qtype == "noul":
        if isinstance(decision, bool):
            return "yes" if decision else "no"
        g = str(decision).strip().lower()
        g = {"true": "yes", "false": "no"}.get(g, g)
        if g not in ("yes", "no"):
            raise ValueError(f"noul requiere decision 'yes'/'no' (o true/false), no {decision!r}")
        return g
    if qtype == "score":
        if isinstance(decision, bool) or not (isinstance(decision, int) or (isinstance(decision, str) and decision.strip().isdigit())):
            raise ValueError(f"score requiere decision = índice entero del nivel (0..{len(keys) - 1}), no {decision!r}")
        g = str(int(decision))
        if g not in keys:
            raise ValueError(f"nivel {g} fuera de rango: criteria tiene {len(keys)} niveles (0..{len(keys) - 1})")
        return g
    if not isinstance(decision, str):
        raise ValueError(f"choice requiere decision = clave de opción (texto), no {decision!r}")
    if decision not in keys:
        raise ValueError(f"decision {decision!r} no está entre las opciones {keys}")
    return decision


def parse_history_row(row, lineno: int) -> DecisionItem:
    """Valida una fila del historial y la convierte en DecisionItem (lanza ValueError con el motivo)."""
    from dec.serve.wire import parse_request
    if not isinstance(row, dict):
        raise ValueError("cada línea debe ser un objeto JSON")
    for k in ("state", "question", "decision"):
        if k not in row:
            raise ValueError(f"falta el campo '{k}'")
    state, q = row["state"], row["question"]
    if not isinstance(state, (str, dict, list)) or (isinstance(state, str) and not state.strip()):
        raise ValueError("'state' debe ser texto no vacío o un objeto/lista JSON")
    if not isinstance(q, dict):
        raise ValueError("'question' debe ser un objeto {type, instructions, criteria, question_key}")
    if q.get("type") not in ("noul", "choice", "score"):
        raise ValueError(f"question.type debe ser noul, choice o score, no {q.get('type')!r}")
    qk = q.get("question_key")
    if not isinstance(qk, str) or not qk.strip():
        raise ValueError("falta question.question_key (obligatoria en el historial: la memoria se particiona por ella)")
    if not isinstance(q.get("instructions", ""), str):
        raise ValueError("question.instructions debe ser texto")
    if q["type"] in ("choice", "score") and q.get("criteria") is None:
        raise ValueError(f"{q['type']} requiere question.criteria")
    _, items = parse_request({"state": state, "questions": {"decision": q}})
    it = items[0]
    gold = _gold(it.qtype, row["decision"], it.option_keys())
    rid = row.get("id")
    if rid is not None and (not isinstance(rid, (str, int)) or isinstance(rid, bool) or not str(rid).strip()):
        raise ValueError("'id' debe ser texto o entero no vacío")
    if rid is None:
        rid = "h-" + hashlib.sha1(json.dumps({k: row[k] for k in ("state", "question", "decision")}, sort_keys=True, ensure_ascii=False)
                                  .encode()).hexdigest()[:16]
    st = state_text(state)
    return DecisionItem(id=str(rid), dataset="client", state_id="s-" + hashlib.sha1(st.encode()).hexdigest()[:16], state=state,
                        qtype=it.qtype, instructions=it.instructions, options=it.options, gold=gold, question_key=qk.strip(),
                        meta={"line": lineno})


def load_history(path: str | Path, max_errors: int = 20) -> list[DecisionItem]:
    """Lee y valida el historial completo; ante errores lanza HistoryError con 'línea N: motivo' (hasta `max_errors`)."""
    path = Path(path)
    if not path.exists():
        raise HistoryError(f"no existe el historial {path}")
    items: list[DecisionItem] = []; errors: list[str] = []; seen: dict[str, int] = {}; n_err = 0
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                it = parse_history_row(json.loads(line), lineno)
                if it.id in seen:
                    raise ValueError(f"id duplicado {it.id!r} (ya usado en la línea {seen[it.id]})")
                seen[it.id] = lineno; items.append(it)
            except json.JSONDecodeError as e:
                n_err += 1; errors.append(f"línea {lineno}: JSON inválido ({e.msg}, columna {e.colno})")
            except (ValueError, KeyError, TypeError) as e:
                n_err += 1; errors.append(f"línea {lineno}: {e}")
    if n_err:
        more = f"\n... y {n_err - max_errors} errores más" if n_err > max_errors else ""
        raise HistoryError(f"{path}: {n_err} filas inválidas\n" + "\n".join(errors[:max_errors]) + more)
    if not items:
        raise HistoryError(f"{path}: historial vacío")
    return items


def _item_to_json(it: DecisionItem) -> dict:
    from dec.data import to_json
    return to_json(it)


def _item_from_json(d: dict) -> DecisionItem:
    from dec.data import from_json
    return from_json(d)


def summarize(items: list[DecisionItem]) -> dict:
    """Conteos por question_key: total y por clase (incluye opciones nunca vistas con 0)."""
    out: dict[str, dict] = {}
    for it in items:
        e = out.setdefault(it.question_key, {"n": 0, "qtype": it.qtype, "classes": {k: 0 for k in it.option_keys()}})
        e["n"] += 1
        for k in it.option_keys():
            e["classes"].setdefault(k, 0)
        e["classes"][it.gold] = e["classes"].get(it.gold, 0) + 1
    return dict(sorted(out.items()))


def pool_warnings(meta: dict) -> list[str]:
    """Avisos de la guía del README: < 200 precedentes por question_key o alguna clase con < 50 ejemplos."""
    w = []
    for qk, e in meta["question_keys"].items():
        if e["n"] < MIN_PER_KEY:
            w.append(f"{qk}: {e['n']} precedentes (< {MIN_PER_KEY}); con pocos precedentes la memoria puede empeorar al backbone solo")
        low = {c: n for c, n in e["classes"].items() if n < MIN_PER_CLASS}
        if low:
            w.append(f"{qk}: clases con < {MIN_PER_CLASS} ejemplos: " + ", ".join(f"{c}={n}" for c, n in low.items())
                     + " (la memoria tiende a la tasa base y deja de detectarlas)")
    return w


def read_pool_meta(pool_dir: str | Path) -> dict:
    p = Path(pool_dir) / "pool.json"
    if not p.exists():
        raise FileNotFoundError(f"{pool_dir} no es un pool de dec (falta pool.json); crearlo con `dec-pool build`")
    return json.loads(p.read_text())


def load_pool_dir(pool_dir: str | Path) -> tuple[dict, list[DecisionItem], dict[str, Encoded], np.ndarray | None]:
    """(meta, items, enc, H) de un pool construido con `dec-pool build`. H = minilm.npy (None si features=h)."""
    from dec.backbone.cache import read_cache
    d = Path(pool_dir); meta = read_pool_meta(d)
    items = [_item_from_json(json.loads(l)) for l in (d / "items.jsonl").open(encoding="utf-8") if l.strip()]
    enc = read_cache(d / "enc.parquet")
    miss = [it.id for it in items if it.id not in enc]
    if miss:
        raise ValueError(f"{len(miss)} ítems de items.jsonl sin codificar en enc.parquet (ej. {miss[0]}); reanudar con `dec-pool build`")
    H = None
    if meta["features"] == "minilm":
        H = np.load(d / "minilm.npy")
        if H.shape[0] != len(items):
            raise ValueError(f"minilm.npy tiene {H.shape[0]} filas para {len(items)} ítems; reconstruir con `dec-pool build`")
    return meta, items, enc, H


def _cached_backbone_id(encp: Path) -> str | None:
    """backbone_id guardado en enc.parquet (columna `backbone_id`), o None si no hay filas."""
    import pyarrow.parquet as pq
    if not encp.exists():
        return None
    col = pq.read_table(encp, columns=["backbone_id"]).column("backbone_id").to_pylist()
    return col[0] if col else None


def build_pool(history: str | Path, out: str | Path, backbone: str, style: str = "jevk5", T: float = 1.0, device: str = "auto",
               features: str = "minilm", dtype: str = "auto", revision: str | None = None, chunk: int = 64,
               token_budget: int | None = None, bb=None, log=print) -> dict:
    """Valida el historial, codifica con el backbone (reanudable por id) y escribe el directorio del pool. Devuelve pool.json."""
    from dec.backbone.cache import read_cache, write_cache
    if features not in CONFIG_BY_FEATURES:
        raise ValueError(f"features debe ser minilm o h, no {features!r}")
    items = load_history(history)
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    old = json.loads((out / "pool.json").read_text()) if (out / "pool.json").exists() else None
    if old and (old["backbone"], old["style"], float(old["T"]), old.get("revision")) != (backbone, style, float(T), revision):
        raise ValueError(f"{out} ya tiene un pool de {old['backbone']} (style={old['style']}, T={old['T']}); usar otro --out "
                         f"o borrar el directorio para reconstruir con {backbone} (style={style}, T={T})")
    encp = out / "enc.parquet"
    enc = read_cache(encp) if encp.exists() else {}
    ids = {it.id for it in items}
    stale = [k for k in enc if k not in ids]
    for k in stale:
        enc.pop(k)
    todo = [it for it in items if it.id not in enc]
    log(f"historial: {len(items)} decisiones válidas; ya codificadas {len(items) - len(todo)}; por codificar {len(todo)}"
        + (f"; descartadas {len(stale)} que ya no están en el historial" if stale else ""))
    if todo:
        if bb is None:
            from dec.backbone.model import Backbone
            bb = Backbone(backbone, dtype=dtype, style=style, temperature=T, revision=revision, device=device)
        tb = token_budget or bb.default_token_budget(); t0 = time.time()
        for s in range(0, len(todo), chunk):
            part = todo[s:s + chunk]
            enc.update({e.item_id: e for e in bb.encode(part, log_every=10**9, token_budget=tb)})
            write_cache(encp, [enc[it.id] for it in items if it.id in enc], bb.backbone_id)   # checkpoint: un corte no pierde lo hecho
            el = time.time() - t0; done = s + len(part)
            log(f"  codificadas {done}/{len(todo)} · {el:.0f} s · ETA {(len(todo) - done) * el / done:.0f} s")
        bb_id, used_dtype = bb.backbone_id, bb.dtype
        prev = (old or {}).get("backbone_id") or (_cached_backbone_id(encp) if len(enc) > len(todo) else None)
        if prev and prev != bb_id:
            log(f"AVISO: el pool se empezó con {prev} y se completó con {bb_id} (otro dtype/revisión): conviene reconstruirlo entero")
    else:
        bb_id = (old or {}).get("backbone_id") or _cached_backbone_id(encp)
        used_dtype = (old or {}).get("dtype") or (bb_id.rsplit(":", 1)[-1] if bb_id else None)
        if stale:
            write_cache(encp, [enc[it.id] for it in items], bb_id or backbone)
    with (out / "items.jsonl").open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(_item_to_json(it), ensure_ascii=False) + "\n")
    if features == "minilm":
        from dec.eval.minilm import embed_texts, set_device
        from dec.backbone.model import resolve_device
        dev = resolve_device(device)
        set_device("cpu" if dev == "mps" else dev)
        np.save(out / "minilm.npy", embed_texts([state_text(it.state) for it in items]))
    elif (out / "minilm.npy").exists():
        (out / "minilm.npy").unlink()
    meta = {"format": POOL_FORMAT, "backbone": backbone, "revision": revision, "style": style, "T": float(T), "dtype": used_dtype,
            "backbone_id": bb_id, "features": features, "config": CONFIG_BY_FEATURES[features], "n_items": len(items),
            "question_keys": summarize(items), "history": str(history),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds"), "dec_version": dec_version()}
    tmp = out / "pool.json.tmp"; tmp.write_text(json.dumps(meta, indent=1, ensure_ascii=False)); tmp.replace(out / "pool.json")
    return meta


def format_info(meta: dict) -> str:
    lines = [f"pool: {meta['n_items']} precedentes · backbone {meta['backbone']} (style={meta['style']}, T={meta['T']}, dtype={meta['dtype']})",
             f"features: {meta['features']} · config: {meta['config']} · creado {meta['created']} · dec {meta['dec_version']}",
             "question_keys:"]
    for qk, e in meta["question_keys"].items():
        lines.append(f"  {qk} ({e['qtype']}): {e['n']} · " + ", ".join(f"{c}={n}" for c, n in e["classes"].items()))
    w = pool_warnings(meta)
    if w:
        lines.append("avisos:")
        lines += [f"  AVISO {x}" for x in w]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI dec-pool: `build` (historial -> pool) e `info` (resumen y avisos)."""
    ap = argparse.ArgumentParser(prog="dec-pool", description="Pool de precedentes a partir del historial de decisiones del cliente")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="valida el historial, codifica con el backbone (reanudable) y escribe el pool")
    b.add_argument("--history", required=True, help="JSONL: una decisión por línea (ver examples/history_example.jsonl)")
    b.add_argument("--backbone", required=True, help="id de Hugging Face (ej. alibiserikbay/JevK5)")
    b.add_argument("--style", default="jevk5", choices=["jevk5", "numeric"]); b.add_argument("--T", type=float, default=1.0)
    b.add_argument("--out", required=True, help="directorio del pool")
    b.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu", "mps"])
    b.add_argument("--dtype", default="auto", choices=["auto", "bf16", "fp32"])
    b.add_argument("--features", default="minilm", choices=["minilm", "h"], help="minilm = Ours-3-cal (recomendada); h = estado oculto")
    b.add_argument("--threads", type=int, default=None, help="hilos de CPU de torch (por defecto: la cuota de CPU del contenedor si la hay)")
    b.add_argument("--revision", default=None); b.add_argument("--chunk", type=int, default=64, help="ítems por checkpoint")
    i = sub.add_parser("info", help="resumen del pool y avisos (< 200 precedentes por question_key, clases con < 50)")
    i.add_argument("pool_dir")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "build":
            from dec._compat import set_cpu_threads
            set_cpu_threads(a.threads)
            meta = build_pool(a.history, a.out, a.backbone, a.style, a.T, a.device, a.features, a.dtype, a.revision, a.chunk)
            print(format_info(meta)); print(f"pool listo en {a.out}; servir con: dec-serve --pool-dir {a.out}")
        else:
            print(format_info(read_pool_meta(a.pool_dir)))
    except (HistoryError, FileNotFoundError, ValueError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
