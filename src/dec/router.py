"""Router de 3 ejes (GRAPH R7 aplicado a decisiones): huecos / oráculo / consecuencia, regla lexicográfica."""
from __future__ import annotations
from dataclasses import dataclass, field
import re
import warnings
import numpy as np
from dec.schema import DecisionItem, CostMatrix, state_text

_STOP = set("the a an and or of to in on for with is are was were be been this that these those which what when where who whom why how does do did will would should could can may might must shall into from by as at it its their there here than then".split())
_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-']*")

def _terms(text: str) -> list[str]:
    out = []
    for t in _TOK.findall(text):
        tl = t.lower().strip(".-'")
        if (len(tl) >= 4 and tl not in _STOP) or any(ch.isdigit() for ch in tl): out.append(tl)
    return out

@dataclass
class GapSignals:
    entropy_norm: float; mass: float; overlap: float; n_options: int; missing_terms: list[str] = field(default_factory=list)
    sem_gap: float = 0.0   # 1 - max coseno MiniLM(pregunta, frase del estado); 0.0 si no se calculó

def _sem_gap_from(q: np.ndarray, S: np.ndarray) -> float:
    """1 - max coseno entre el vector de pregunta y las frases (embeddings normalizados); sin frases, 1.0."""
    return 1.0 if len(S) == 0 else float(min(1.0, max(0.0, 1.0 - float(np.max(S @ q)))))

def _sem_inputs(item: DecisionItem) -> tuple[str, list[str]]:
    from dec.eval.unanswerable import _sentences   # import perezoso: unanswerable importa sklearn
    return item.instructions, _sentences(state_text(item.state))   # solo el enunciado (sin descripciones de opciones)

def gap_signals(item: DecisionItem, probs: np.ndarray, mass: float, sem: bool = False) -> GapSignals:
    """Calcula entropía normalizada, masa, solapamiento léxico, términos ausentes y (si sem=True) sem_gap semántico."""
    p = np.clip(np.asarray(probs, float), 1e-12, None)
    p = p / p.sum()
    K = len(p)
    H = 0.0 if K < 2 else float(-(p * np.log(p)).sum() / np.log(K))
    q_terms = _terms(item.instructions); s_terms = set(_terms(state_text(item.state)))
    missing = [t for t in q_terms if t not in s_terms]
    overlap = 1.0 if not q_terms else 1 - len(missing) / len(q_terms)
    sg = 0.0
    if sem:
        from dec.eval import minilm   # acceso por módulo para poder sustituir embed_sentences en tests
        q, sents = _sem_inputs(item)
        E = minilm.embed_sentences([q] + sents); sg = _sem_gap_from(E[0], E[1:])
    return GapSignals(H, float(mass), float(overlap), K, missing, sg)

def gap_signals_batch(items: list[DecisionItem], P, masses) -> list[GapSignals]:
    """gap_signals(sem=True) para muchos ítems con una sola llamada de embedding (P: filas de probs, posiblemente rellenas a Kmax)."""
    from dec.eval import minilm
    ins = [_sem_inputs(it) for it in items]
    texts = list(dict.fromkeys([q for q, _ in ins] + [s for _, ss in ins for s in ss]))
    E = minilm.embed_sentences(texts); pos = {t: i for i, t in enumerate(texts)}
    out = []
    for it, (q, ss), p, m in zip(items, ins, P, masses):
        base = gap_signals(it, np.asarray(p, float)[:len(it.options)], float(m))
        base.sem_gap = _sem_gap_from(E[pos[q]], E[[pos[x] for x in ss]] if ss else E[:0])
        out.append(base)
    return out

@dataclass
class RouterThresholds:
    entropy: float = 0.8; mass: float = 0.0; overlap: float = 0.3; c_high: float = 5.0
    sem: float = 1.1   # umbral de sem_gap; 1.1 = desactivado (sem_gap <= 1)

@dataclass
class RouterVerdict:
    huecos: str; oraculo: str; consecuencia: str; evidencia: dict; any_red: bool

def route(s: GapSignals, costs: CostMatrix | None, thr: RouterThresholds, has_constraints: bool) -> RouterVerdict:
    """Clasifica huecos, oráculo y consecuencia; thr.mass <= 0 desactiva el eje de masa (sin memoria)."""
    mass_low = (s.mass < thr.mass) if thr.mass > 0 else True
    lexical = s.entropy_norm > thr.entropy and mass_low and s.overlap < thr.overlap
    huecos = "rojo" if (lexical or s.sem_gap > thr.sem) else "verde"
    oraculo = "verde" if ((thr.mass > 0 and s.mass >= thr.mass) or has_constraints) else "rojo"
    high = costs is not None and bool(costs.irreversible or (costs.loss and max(costs.loss.values()) >= thr.c_high))
    cons = "alta" if high else "baja"
    ev = {"entropy_norm": s.entropy_norm, "mass": s.mass, "overlap": s.overlap, "sem_gap": s.sem_gap, "missing_terms": s.missing_terms[:8],
          "has_constraints": has_constraints, "irreversible": sorted(costs.irreversible) if costs is not None else []}
    return RouterVerdict(huecos, oraculo, cons, ev, any_red=(huecos == "rojo" or oraculo == "rojo" or cons == "alta"))

def fit_thresholds(answerable: list[GapSignals], unanswerable: list[GapSignals], use_sem: bool = False) -> RouterThresholds:
    """Ajusta umbrales por balanced accuracy; con todas las masas en 0 ajusta solo entropía y solapamiento (mass=0)."""
    if not answerable or not unanswerable:
        raise ValueError("fit_thresholds requires non-empty answerable and unanswerable lists")
    masses = np.array([s.mass for s in answerable + unanswerable])
    mgrid = np.quantile(masses, [0.1, 0.25, 0.5, 0.75]) if masses.max() > 0 else [0.0]
    sg = np.array([s.sem_gap for s in answerable + unanswerable])
    sgrid = [1.1] + [float(x) for x in np.unique(np.quantile(sg, np.linspace(0.05, 0.95, 19)))] if use_sem else [1.1]   # 1.1 = sem apagado
    best, bt = -1.0, RouterThresholds(mass=0.0)
    for e in np.linspace(0.5, 0.95, 10):
        for m in mgrid:
            for o in np.linspace(0.1, 0.7, 13):
                for g in sgrid:
                    thr = RouterThresholds(float(e), float(m), float(o), 5.0, g)
                    tpr = np.mean([route(s, None, thr, False).huecos == "rojo" for s in unanswerable])
                    tnr = np.mean([route(s, None, thr, False).huecos == "verde" for s in answerable])
                    if (tpr + tnr) / 2 > best: best, bt = (tpr + tnr) / 2, thr
    if best <= 0.5 + 1e-9:
        warnings.warn(f"fit_thresholds: best balanced accuracy is {best:.2f}; signals do not separate answerable from unanswerable", UserWarning)
    return bt
