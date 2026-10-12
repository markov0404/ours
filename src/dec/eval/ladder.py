"""Escalera de baselines B0..Ours-2 sobre caches; cada config devuelve el mismo dict de métricas."""
from __future__ import annotations
from dataclasses import dataclass
import random
import numpy as np
from dec.schema import DecisionItem, Encoded, state_text
from dec.data import load
from dec.memory import PrecedentMemory, MemoryHit
from dec.phi import GlobalDiagonalMetric, PhiMask, fit_nca, embed_questions
from dec.fusion import FusionRecord, FusionParams, fit_fusion, fit_post_temperature, fuse, p_mem_vector, temper
from dec.coherence import Constraint, hierarchy, implies, joint_decode, incoherence_rate
from dec.risk import sgr_threshold
from dec.metrics import accuracy, ece, brier, auroc, qwk

CONFIGS = ["B0", "B1", "B2", "B3", "ours1", "ours2", "ours1cal", "ours3", "ours3cal"]
# ronda 2: ours1cal = ours1 + temperatura post-fusión (hip-11); ours3 = memoria sobre MiniLM + métrica global (hip-12); ours3cal = ambas
_KIND = {"B2": "minilm", "ours3": "minilm", "ours3cal": "minilm"}
_METRIC = {"ours1": "global", "ours1cal": "global", "ours2": "phi", "ours3": "global", "ours3cal": "global"}
SGR_DELTA = 0.05   # confianza 95 % de la cota SGR: con ~1000 LOO de calibración, 0.001 (Geifman, n ~ 1e4-1e5) era infactible


@dataclass
class Prepared:
    """Datos listos para correr la escalera: pool, evaluación, caches, etiquetados, restricciones y embeddings de preguntas."""
    items_pool: list[DecisionItem]
    items_eval: list[DecisionItem]
    enc: dict[str, Encoded]
    labeled: list[DecisionItem]
    constraints: list[Constraint]
    qemb: dict[str, np.ndarray]
    dataset: str
    seed: int
    backbone_id: str
    n_excluded: int = 0   # ítems del pool excluidos por estado idéntico a uno de evaluación (spec §5)


class FusionBank:
    """Parámetros de fusión por número de opciones K (`fit_fusion` exige K homogéneo): plumb, massive y jevbench mezclan K."""

    def __init__(self, by_k: dict[int, FusionParams], default: FusionParams):
        self.by_k, self.default = by_k, default

    def for_k(self, K: int) -> FusionParams:
        return self.by_k.get(K, self.default)

    def to_json(self) -> dict:
        return {"default": self.default.__dict__, "by_k": {str(k): p.__dict__ for k, p in sorted(self.by_k.items())}}


def fit_fusion_grouped(recs: list[FusionRecord], min_n: int = 20, post: bool = False) -> FusionBank:
    """Ajusta la fusión por grupo de K; grupos con menos de `min_n` registros heredan los parámetros del grupo mayor.

    `post=True` (hip-11) ajusta además T_post por grupo sobre las salidas fusionadas LOO de ese grupo.
    """
    groups: dict[int, list[FusionRecord]] = {}
    for r in recs:
        groups.setdefault(int(r.p0.shape[0]), []).append(r)
    by_k = {k: fit_fusion(g) for k, g in groups.items() if len(g) >= min_n}
    if post:
        for k, prm in by_k.items():
            P = np.stack([fuse(r.p0, r.p_mem, r.mass, prm) for r in groups[k]]); G = np.array([r.gold_idx for r in groups[k]])
            prm.T_post = fit_post_temperature(P, G)
    if not by_k and groups:
        k = max(groups, key=lambda k: len(groups[k])); by_k = {k: fit_fusion(groups[k])}
    default = by_k[max(by_k, key=lambda k: len(groups[k]))] if by_k else FusionParams()
    return FusionBank(by_k, default)


def _norm_state(it: DecisionItem) -> str:
    """Misma normalización que `dec.data.contamination` (duplicado exacto = mismos tokens normalizados)."""
    from dec.data.contamination import _norm
    return " ".join(_norm(state_text(it.state)))


def _split_states(items: list[DecisionItem], frac: float, seed: int) -> tuple[list[DecisionItem], list[DecisionItem]]:
    """Parte los ítems por state_id: fracción `frac` (barajada con seed) y el resto."""
    sids = sorted({it.state_id for it in items}); rng = random.Random(seed); rng.shuffle(sids)
    keep = set(sids[: int(len(sids) * frac)])
    return [it for it in items if it.state_id in keep], [it for it in items if it.state_id not in keep]


def _limit_states(items: list[DecisionItem], limit: int | None, seed: int) -> list[DecisionItem]:
    """Limita a `limit` estados elegidos al azar con seed (None = sin límite)."""
    if limit is None:
        return items
    sids = sorted({it.state_id for it in items}); rng = random.Random(seed); rng.shuffle(sids); keep = set(sids[:limit])
    return [it for it in items if it.state_id in keep]


def pool_and_eval(dataset: str, seed: int, limit_states: int | None = 3000) -> tuple[list[DecisionItem], list[DecisionItem], list[Constraint]]:
    """Pool etiquetable, evaluación y restricciones de un dataset, sin necesidad de caches.

    El límite de estados del pool usa SIEMPRE seed 0: el pool es la definición del dataset y la campaña de caches lo
    codifica con seed 0; la semilla de la corrida solo gobierna la muestra etiquetada y el split de plumb.
    """
    constraints: list[Constraint] = []
    if dataset == "plumb":
        tr = load("plumb", "train"); te = load("plumb", "test"); pool, held = _split_states(tr, 0.8, seed); ev = te + held
    elif dataset == "massive":
        pool = _limit_states(load("massive", "train"), limit_states, 0); ev = load("massive", "test")
        intent = next(it for it in ev if it.question_key == "massive:intent")   # el ítem :scenario no lleva meta
        constraints = [hierarchy("massive:intent", "massive:scenario", intent.meta["scenario_of_intent"])]
    elif dataset == "clinc":
        pool = _limit_states(load("clinc", "train"), limit_states, 0); ev = load("clinc", "test")
    elif dataset == "civil":
        pool = load("civil", "train", n=3000, seed=0); ev = load("civil", "test", n=3000, seed=0)
        # el cache JSON convierte las tuplas en listas
        constraints = [implies(a, "yes", b, {"yes"}) for a, b in (tuple(p) for p in (ev[0].meta.get("implications") or []))]
    elif dataset == "banking77":
        pool = _limit_states(load("banking77", "train"), limit_states, 0); ev = load("banking77", "test")
    elif dataset == "jevbench":
        pool = []; ev = load("jevbench", "public")
    else:
        raise KeyError(dataset)
    return pool, ev, constraints


def prepare(dataset: str, backbone_id: str, n: int, seed: int, limit_states: int | None = 3000, pool_from: list[str] | None = None) -> Prepared:
    """Carga pool/eval, lee los caches del backbone y muestrea `n` etiquetados con la semilla.

    `pool_from` (opcional): datasets cuyo split train reemplaza al pool (pool externo declarado, p. ej. G6 sobre jevbench).
    """
    from dec.backbone.cache import read_cache, cache_path  # import perezoso: el módulo importa sin el cache
    pool, ev, constraints = pool_and_eval(dataset, seed, limit_states)
    enc: dict[str, Encoded] = {}
    for split in ({"train", "test"} if dataset != "jevbench" else {"public"}):
        p = cache_path(dataset, split, backbone_id)
        if p.exists():
            enc.update(read_cache(p))
    if pool_from:
        pool = []
        for d in pool_from:
            pool += _limit_states(load(d, "train"), limit_states, 0)
            p = cache_path(d, "train", backbone_id)
            if p.exists():
                enc.update(read_cache(p))
    evs = {_norm_state(it) for it in ev}; n_pool = len(pool)
    pool = [it for it in pool if _norm_state(it) not in evs]   # exclusión (no aborto) de estados idénticos pool/eval, spec §5
    n_excluded = n_pool - len(pool)
    missing = [it.id for it in pool + ev if it.id not in enc]
    if missing:
        raise FileNotFoundError(f"{len(missing)} ítems sin cache para {backbone_id}; corré scripts/cache_dataset.py (ej. {missing[0]})")
    rng = random.Random(seed); labeled = rng.sample(pool, min(n, len(pool))) if n > 0 else []
    qtexts = sorted({it.instructions for it in pool + ev}); Q = embed_questions(qtexts) if qtexts else np.zeros((0, 384))
    return Prepared(pool, ev, enc, labeled, constraints, dict(zip(qtexts, Q)), dataset, seed, backbone_id, n_excluded)


def _features(prep: Prepared, items: list[DecisionItem], kind: str) -> np.ndarray:
    """Matriz de features de los ítems: `h` del backbone o embeddings MiniLM (B2)."""
    if kind == "h":
        return np.stack([prep.enc[it.id].h for it in items])
    from .minilm import embed_states
    return embed_states(items)


def _mask_for(masks, prep: Prepared, it: DecisionItem):
    """Máscara de métrica para un ítem (None = sin métrica aprendida)."""
    if masks is None:
        return None
    kind, obj = masks
    return obj.mask() if kind == "global" else obj.mask(prep.qemb[it.instructions])


def _fit_memory(prep: Prepared, kind: str, metric: str, post: bool = False, prior_correction: bool = False, H: np.ndarray | None = None):
    """Construye la memoria con los etiquetados, ajusta la métrica y la fusión por LOO.

    `H` (opcional): features ya calculadas de `prep.labeled` (mismo orden); el pool de un cliente trae `minilm.npy` precalculado.
    """
    labeled = prep.labeled
    if not labeled:
        return None, None, None
    H = _features(prep, labeled, kind) if H is None else np.asarray(H, np.float32); d = H.shape[1]
    if H.shape[0] != len(labeled):
        raise ValueError(f"H tiene {H.shape[0]} filas para {len(labeled)} etiquetados")
    golds = [it.gold for it in labeled]; qkeys = [it.question_key for it in labeled]
    ids = [it.id for it in labeled]
    mem = PrecedentMemory(d=d); mem.add_many(H, golds, qkeys, ids); mem.build()
    masks = None
    if metric == "global":
        m = GlobalDiagonalMetric(d); fit_nca(m, H, golds, qkeys, None, epochs=15); masks = ("global", m)
    elif metric == "phi":
        Q = np.stack([prep.qemb[it.instructions] for it in labeled])
        phi = PhiMask(d, Q.shape[1], d_p=128, J=min(d, max(16, d // 10)))
        fit_nca(phi, H, golds, qkeys, Q, epochs=20); masks = ("phi", phi)
    # τ fija por memoria (mediana de distancias LOO con la métrica ajustada): sin esto la masa es ~constante (τ por consulta).
    mem.calibrate_tau(H, qkeys, ids, None if masks is None else [_mask_for(masks, prep, it) for it in labeled])
    recs = []
    for it, h in zip(labeled, H):
        hit = mem.search(h, it.question_key, mask=_mask_for(masks, prep, it), exclude_id=it.id)
        pri = mem.class_freq(it.question_key) if prior_correction else None
        recs.append(FusionRecord(prep.enc[it.id].p0, p_mem_vector(hit, it.option_keys(), pri), hit.mass, it.gold_index()))
    return mem, masks, (fit_fusion_grouped(recs, post=post) if recs else FusionBank({}, FusionParams()))


def _coherence(prep: Prepared, ev: list[DecisionItem], probs: dict[str, np.ndarray], G: np.ndarray) -> dict:
    """Incoherencia antes/después de la decodificación conjunta y accuracy tras ella."""
    by_state: dict[str, list[DecisionItem]] = {}
    for it in ev:
        by_state.setdefault(it.state_id, []).append(it)
    before, after, fallbacks = [], [], 0
    pred_after: dict[str, int] = {}
    for grp in by_state.values():
        before.append({it.question_key: it.option_keys()[int(np.argmax(probs[it.id]))] for it in grp})
        pb = {it.question_key: dict(zip(it.option_keys(), map(float, probs[it.id]))) for it in grp}
        adj, info = joint_decode(pb, prep.constraints)
        if info["map_consistent"]:
            act = dict(info["map"])
        else:
            fallbacks += 1
            act = {q: max(m, key=m.get) for q, m in adj.items()}
        after.append(act)
        for it in grp:
            pred_after[it.id] = it.option_keys().index(act[it.question_key])
    pa = np.array([pred_after[it.id] for it in ev])
    return {"incoherence_before": incoherence_rate(before, prep.constraints),
            "incoherence_after": incoherence_rate(after, prep.constraints),
            "acc_after": float((pa == G).mean()), "map_fallbacks": fallbacks}


def run_config(prep: Prepared, config: str, alpha: float = 0.05, return_probs: bool = False, prior_correction: bool = False):
    """Corre una config de la escalera sobre `prep` y devuelve el dict de métricas común.

    Con `return_probs=True` devuelve `(dict, {"P", "G", "items", "mass"})`: probabilidades fusionadas, índices gold, ítems y masa de memoria por ítem de evaluación.
    """
    if config not in CONFIGS:
        raise KeyError(config)
    ev = prep.items_eval; p0 = {it.id: prep.enc[it.id].p0 for it in ev}
    kind = _KIND.get(config, "h"); metric = _METRIC.get(config, "none"); post = config.endswith("cal")
    pmv = (lambda hit, it: p_mem_vector(hit, it.option_keys(), mem.class_freq(it.question_key))) if prior_correction else (lambda hit, it: p_mem_vector(hit, it.option_keys()))
    mem = masks = params = None
    if config == "B1":
        recs = [FusionRecord(prep.enc[it.id].p0, np.zeros(len(it.options)), 0.0, it.gold_index()) for it in prep.labeled]
        params = fit_fusion_grouped(recs) if recs else FusionBank({}, FusionParams())
    elif config in ("B2", "B3", "ours1", "ours2", "ours1cal", "ours3", "ours3cal"):
        mem, masks, params = _fit_memory(prep, kind, metric, post=post, prior_correction=prior_correction)
    params = params or FusionBank({}, FusionParams())
    Heval = _features(prep, ev, kind) if mem is not None else None
    probs: dict[str, np.ndarray] = {}; hits: dict[str, MemoryHit] = {}
    for j, it in enumerate(ev):
        if mem is None:
            hit = MemoryHit({}, 0.0, []); fused = temper(p0[it.id], params.for_k(len(it.options)).T)
        else:
            hit = mem.search(Heval[j], it.question_key, mask=_mask_for(masks, prep, it))
            prm = params.for_k(len(it.options)); fused = fuse(p0[it.id], pmv(hit, it), hit.mass, prm)
            if post: fused = temper(fused, prm.T_post)
        probs[it.id] = fused; hits[it.id] = hit
    Kmax = max(len(it.options) for it in ev)   # K heterogéneo (plumb, massive, jevbench): relleno con ceros, no altera argmax/ECE/Brier
    P = np.stack([np.pad(probs[it.id], (0, Kmax - len(probs[it.id]))) for it in ev]); G = np.array([it.gold_index() for it in ev])
    coh = _coherence(prep, ev, probs, G) if prep.constraints else {}
    risk = {"theta": float("inf"), "automation": 0.0, "realized_risk": float("nan"), "bound": float("nan"), "alpha": alpha, "delta": SGR_DELTA, "insufficient": True}
    if prep.labeled and mem is not None:
        conf, corr = [], []
        Hl = _features(prep, prep.labeled, kind)
        for it, h in zip(prep.labeled, Hl):
            hit = mem.search(h, it.question_key, mask=_mask_for(masks, prep, it), exclude_id=it.id)
            prm = params.for_k(len(it.options)); f = fuse(prep.enc[it.id].p0, pmv(hit, it), hit.mass, prm)
            if post: f = temper(f, prm.T_post)
            conf.append(f.max()); corr.append(int(f.argmax() == it.gold_index()))
        r = sgr_threshold(np.array(conf), np.array(corr), alpha, delta=SGR_DELTA); sel = P.max(1) >= r.theta
        risk = {"theta": r.theta, "automation": float(sel.mean()),
                "realized_risk": float(1 - (P.argmax(1) == G)[sel].mean()) if sel.any() else float("nan"),
                "bound": r.bound, "alpha": alpha, "delta": SGR_DELTA, "insufficient": r.insufficient}
    out = {"dataset": prep.dataset, "config": config, "seed": prep.seed, "backbone_id": prep.backbone_id,
           "n_labeled": 0 if config == "B0" else len(prep.labeled), "n_eval": len(ev),
           "acc": accuracy(P, G), "ece": ece(P, G), "brier": brier(P, G), "risk": risk, "coherence": coh, "fusion": params.to_json(), "n_excluded": prep.n_excluded}
    if prep.dataset == "clinc":
        oos = np.array([it.meta["is_oos"] for it in ev]); conf = P.max(1); mass = np.array([hits[it.id].mass for it in ev])
        out["oos"] = {"auroc_conf": auroc(1 - conf, oos), "auroc_conf_mass": auroc((1 - conf) + (1 - mass / (mass.max() + 1e-9)), oos)}
    def _group(keyfn, min_n: int) -> dict:
        res = {}
        for g in sorted({keyfn(it) for it in ev if keyfn(it)}):
            idx = [k for k, it in enumerate(ev) if keyfn(it) == g]
            if len(idx) >= min_n:
                res[g] = {"n": len(idx), "acc": accuracy(P[idx], G[idx])}
        return res
    out["by_family"] = _group(lambda it: it.family, 20)
    out["by_tier"] = _group(lambda it: str(it.meta.get("tier", "")), 1)
    sc = [k for k, it in enumerate(ev) if it.qtype == "score"]
    if sc:
        out["qwk"] = qwk(P[sc].argmax(1), G[sc], max(len(ev[k].options) for k in sc))
    if return_probs:
        return out, {"P": P, "G": G, "items": ev, "mass": np.array([hits[it.id].mass for it in ev])}
    return out
