"""Pool de memoria para dec-serve: reproduce el camino de la escalera (`ladder._fit_memory` + `run_config`).

Dos orígenes: un dataset de investigación (`--memory-pool dataset:split`, features h sin recalibrar = Ours-1) o el directorio de un
pool del cliente (`--pool-dir`, `MemoryPool.from_dir`; por defecto features MiniLM + métrica global + T post-fusión = Ours-3-cal).
"""
from __future__ import annotations
import logging
import time
import numpy as np
from dec.schema import DecisionItem, Encoded, state_text
from dec.memory import MemoryHit
from dec.fusion import fuse, p_mem_vector, temper
from dec.policy import RiskState
from dec.risk import sgr_threshold
from dec.eval.ladder import Prepared, FusionBank, SGR_DELTA, _fit_memory, _mask_for, _norm_state, pool_and_eval, _limit_states

log = logging.getLogger("dec.serve.memory")


def load_pool_items(dataset: str, split: str, n: int, seed: int, limit_states: int | None = 3000) -> tuple[list[DecisionItem], list[DecisionItem]]:
    """(pool, etiquetados). Para `train` replica `ladder.prepare`: pool de `pool_and_eval` sin estados idénticos a evaluación y
    muestra de `n` con `random.Random(seed)`; para otros splits, el split completo (con límite de estados)."""
    import random
    if split == "train":
        pool, ev, _ = pool_and_eval(dataset, seed, limit_states)
        evs = {_norm_state(it) for it in ev}; pool = [it for it in pool if _norm_state(it) not in evs]
    else:
        from dec.data import load
        pool = _limit_states(load(dataset, split), limit_states, 0)
    return pool, (random.Random(seed).sample(pool, min(n, len(pool))) if n > 0 else [])


def map_question_key(item: DecisionItem, strategy: str, qkey_map: dict[str, str], by_instr: dict[str, str], by_type: dict[str, str | None],
                     pool_keys: set[str] | frozenset[str] = frozenset()) -> str | None:
    """Clave de pregunta del pool para un ítem de la petición, o None si no hay mapeo.

    Orden: `qkey_map` (prefijo de `instructions` más largo, luego la clave `serve:<qtype>` exacta), luego la estrategia
    (`instructions`: texto idéntico en el pool; `type`: clave única del pool para ese qtype; `none`: sin mapeo).
    """
    if item.question_key in pool_keys:   # el cliente declaró una question_key del pool (campo `question_key` de la pregunta)
        return item.question_key
    pref = [p for p in qkey_map if not p.startswith("serve:") and item.instructions.startswith(p)]
    if pref:
        return qkey_map[max(pref, key=len)]
    if item.question_key in qkey_map:
        return qkey_map[item.question_key]
    if strategy == "instructions":
        return by_instr.get(item.instructions)
    if strategy == "type":
        return by_type.get(item.qtype)
    return None


class MemoryPool:
    """Memoria de precedentes + métrica global + fusión + umbral SGR ajustados sobre un pool etiquetado."""

    def __init__(self, backbone_id: str, bb=None, pool_from: tuple[str, str] = ("plumb", "train"), n: int = 1000, seed: int = 0,
                 limit_states: int | None = 3000, alpha: float = 0.05, device: str | None = None, qkey_strategy: str = "instructions",
                 qkey_map: dict[str, str] | None = None, items: list[DecisionItem] | None = None, enc: dict[str, Encoded] | None = None,
                 features: str = "h", H: np.ndarray | None = None, post: bool = False, source: dict | None = None):
        """`bb` (Backbone) codifica los etiquetados sin cache; `items`/`enc` permiten inyectar un pool ya cargado (tests, pool del cliente).

        `features`: "h" (estado oculto del backbone) o "minilm" (embedding MiniLM del estado; `H` = matriz precalculada alineada con
        `items`). `post=True` aplica la temperatura post-fusión (configs *cal).
        """
        import torch
        t0 = time.time()
        if features not in ("h", "minilm"):
            raise ValueError(f"features desconocidas: {features}")
        self.alpha, self.qkey_strategy, self.qkey_map = alpha, qkey_strategy, dict(qkey_map or {})
        self.features, self.post, self.source = features, post, dict(source or {})
        dataset, split = pool_from
        if items is None:
            pool, labeled = load_pool_items(dataset, split, n, seed, limit_states)
        else:
            pool, labeled = items, items
        enc = dict(enc or {})
        if not enc:
            from dec.backbone.cache import read_cache, cache_path
            p = cache_path(dataset, split, backbone_id)
            if p.exists():
                enc.update(read_cache(p))
        missing = [it for it in labeled if it.id not in enc]
        if missing:
            if bb is None:
                raise FileNotFoundError(f"{len(missing)} etiquetados sin cache y sin Backbone para codificarlos")
            log.warning("codificando %d precedentes sin cache con el backbone del servidor", len(missing))
            enc.update({e.item_id: e for e in bb.encode(missing, log_every=100)})
        if features == "minilm" and H is None:
            from dec.eval.minilm import embed_texts
            H = embed_texts([state_text(it.state) for it in labeled])
        prep = Prepared(pool, [], enc, labeled, [], {}, dataset, seed, backbone_id)
        self._prep = prep
        self.mem, self._masks, self.bank = _fit_memory(prep, features, "global", post=post, H=H)
        if self.mem is None:
            raise ValueError("pool de memoria vacío")
        self._H = np.stack([enc[it.id].h for it in labeled]) if H is None else np.asarray(H, np.float32)
        dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dev != "cpu":
            self.mem.device = dev; self.mem.build()   # misma búsqueda en GPU; τ ya calibrada se conserva
        self.risk_state = self._fit_risk(labeled)
        keys = sorted({it.question_key for it in labeled}); self.question_keys = keys
        ks = set(keys); by_instr: dict[str, set[str]] = {}; by_type: dict[str, set[str]] = {}
        for it in labeled:
            by_instr.setdefault(it.instructions, set()).add(it.question_key); by_type.setdefault(it.qtype, set()).add(it.question_key)
        self._by_instr = {t: next(iter(s)) for t, s in by_instr.items() if len(s) == 1}
        self._by_type = {t: (next(iter(s)) if len(s) == 1 else None) for t, s in by_type.items()}
        bad = [v for v in self.qkey_map.values() if v not in ks]
        if bad:
            raise ValueError(f"qkey-map apunta a question_keys fuera del pool: {bad}")
        self.size, self.fit_seconds, self.device = len(labeled), time.time() - t0, dev
        log.warning("memoria lista: %d precedentes, %d question_keys, features=%s, post=%s, theta=%.4f, ajuste %.1fs (device=%s)",
                    self.size, len(keys), features, post, self.risk_state.theta, self.fit_seconds, dev)

    @classmethod
    def from_dir(cls, pool_dir, bb=None, device: str | None = None, alpha: float = 0.05, qkey_strategy: str = "instructions",
                 qkey_map: dict[str, str] | None = None) -> "MemoryPool":
        """Carga un pool de `dec-pool build` y ajusta memoria + métrica + fusión (post si la config es *cal) + τ + umbral SGR.

        Si se pasa `bb`, se verifica que sea el mismo backbone/estilo/T con que se construyó el pool.
        """
        import torch
        from dec.pool import load_pool_dir
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        meta, items, enc, H = load_pool_dir(pool_dir)
        if bb is not None:
            got = (bb.model_id, bb.style, float(bb.T)); want = (meta["backbone"], meta["style"], float(meta["T"]))
            if got != want:
                raise ValueError(f"el pool {pool_dir} se construyó con backbone={want[0]} style={want[1]} T={want[2]}; "
                                 f"el servidor usa backbone={got[0]} style={got[1]} T={got[2]}")
            if meta.get("dtype") and meta["dtype"] != bb.dtype:
                log.warning("pool codificado en %s y servidor en %s: diferencias numéricas pequeñas en p0", meta["dtype"], bb.dtype)
        if meta["features"] == "minilm":
            from dec.eval.minilm import set_device
            set_device("cuda" if device.startswith("cuda") else "cpu")
            from dec.eval.minilm import embed_texts
            embed_texts(["warm-up"])   # carga MiniLM al arrancar, no en la primera petición
        src = {"pool_dir": str(pool_dir), "config": meta["config"], "features": meta["features"], "created": meta["created"]}
        return cls(meta.get("backbone_id") or meta["backbone"], items=items, enc=enc, H=H, features=meta["features"],
                   post=meta["config"].endswith("cal"), device=device, alpha=alpha, qkey_strategy=qkey_strategy, qkey_map=qkey_map,
                   pool_from=("client", "pool"), source=src)

    def _fit_risk(self, labeled: list[DecisionItem]) -> RiskState:
        """Umbral SGR sobre confianzas fusionadas LOO (igual que `run_config`, con la misma T post si `post`)."""
        conf, corr = [], []
        for it, h in zip(labeled, self._H):
            hit = self.mem.search(h, it.question_key, mask=_mask_for(self._masks, self._prep, it), exclude_id=it.id)
            f = self.fuse(self._prep.enc[it.id].p0, hit, it.option_keys())
            conf.append(f.max()); corr.append(int(f.argmax() == it.gold_index()))
        r = sgr_threshold(np.array(conf), np.array(corr), self.alpha, delta=SGR_DELTA)
        return RiskState(r.theta, self.alpha, r.bound)

    def query_vector(self, item: DecisionItem, e: Encoded) -> np.ndarray:
        """Vector de búsqueda de un ítem de petición: `e.h` o el embedding MiniLM del estado (sin cache en disco)."""
        if self.features == "h":
            return e.h
        from dec.eval.minilm import embed_texts
        return embed_texts([state_text(item.state)])[0]

    def map_key(self, item: DecisionItem) -> str | None:
        """Clave del pool para un ítem de petición (None = sin mapeo)."""
        return map_question_key(item, self.qkey_strategy, self.qkey_map, self._by_instr, self._by_type, frozenset(self.question_keys))

    def mask(self) -> np.ndarray | None:
        """Máscara de la métrica global (None sin métrica)."""
        return None if self._masks is None else self._masks[1].mask()

    def hit(self, h: np.ndarray, question_key: str, mask: np.ndarray | None = None) -> MemoryHit:
        """Búsqueda de precedentes; `mask=None` usa la métrica global ajustada."""
        return self.mem.search(h, question_key, mask=self.mask() if mask is None else mask)

    def fuse(self, p0: np.ndarray, hit: MemoryHit, option_keys: list[str]) -> np.ndarray:
        """Fusión prior + voto de memoria con los parámetros del grupo de K (y su T post-fusión si `post`)."""
        prm = self.bank.for_k(len(option_keys)); f = fuse(p0, p_mem_vector(hit, option_keys), hit.mass, prm)
        return temper(f, prm.T_post) if self.post else f

    def predict(self, item: DecisionItem, e: Encoded) -> tuple[np.ndarray, MemoryHit, RiskState, str, str | None]:
        """(fused, hit, riesgo, reason_code, question_key) de un ítem ya codificado; sin mapeo -> backbone solo (p0, riesgo sin umbral)."""
        from dec.fusion import reason_code
        qk = self.map_key(item)
        if qk is None:
            return e.p0, MemoryHit({}, 0.0, []), RiskState(float("inf"), self.alpha, float("nan")), "sin_precedentes", None
        hit = self.hit(self.query_vector(item, e), qk); fused = self.fuse(e.p0, hit, item.option_keys())
        reason = "sin_precedentes"
        if hit.mass > 0 and bool(hit.precedents):
            reason = reason_code(e.p0, p_mem_vector(hit, item.option_keys()), hit.mass, self.bank.for_k(len(item.options)))
        return fused, hit, self.risk_state, reason, qk

    def info(self) -> dict:
        """Resumen para /health."""
        return {"pool_size": self.size, "question_keys": self.question_keys, "theta": float(self.risk_state.theta) if np.isfinite(self.risk_state.theta) else None, "alpha": self.alpha,
                "qkey_strategy": self.qkey_strategy, "qkey_map": self.qkey_map, "fit_seconds": round(self.fit_seconds, 2),
                "features": self.features, "post_temperature": self.post, **self.source}
