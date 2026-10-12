"""Variantes sin respuesta: quitar del estado la oración con mayor similitud TF-IDF a (pregunta + opción correcta)."""
from __future__ import annotations
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dec.schema import DecisionItem, state_text

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_TOK = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-']*")


class NoLexicalEvidence(ValueError):
    """La pregunta y la opción correcta no comparten términos con ninguna oración del estado."""


def _sentences(text: str) -> list[str]:
    """Divide el estado en oraciones no vacías."""
    return [s for s in _SENT.split(text) if s and s.strip()]


def _norm(t: str) -> str:
    """Minúsculas y sin punto final (conserva puntos internos como en 44.39)."""
    return t.rstrip(".").lower()


def _terms(text: str) -> set[str]:
    """Conjunto de términos normalizados de un texto."""
    return {_norm(t) for t in _TOK.findall(text)}


def _gold(item: DecisionItem):
    """Devuelve la opción correcta del ítem o KeyError(item.id)."""
    for o in item.options:
        if o.key == item.gold:
            return o
    raise KeyError(item.id)


def evidence_sentence(item: DecisionItem) -> tuple[int, str]:
    """Índice y texto de la oración más similar a (pregunta + opción correcta); ValueError si no hay solapamiento léxico."""
    sents = _sentences(state_text(item.state))
    gold = _gold(item)
    query = f"{item.instructions} {gold.key.replace('_', ' ')} {gold.desc}"
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True).fit(sents + [query])
    sims = cosine_similarity(vec.transform([query]), vec.transform(sents))[0]
    if sims.max() == 0:
        raise NoLexicalEvidence("sin evidencia léxica")
    i = int(sims.argmax())
    return i, sents[i]


def make_unanswerable(item: DecisionItem) -> DecisionItem:
    """Copia del ítem sin la oración de evidencia; ValueError si el estado tiene menos de 2 oraciones."""
    sents = _sentences(state_text(item.state))
    if len(sents) < 2:
        raise ValueError("el estado necesita al menos 2 oraciones")
    i, removed = evidence_sentence(item)
    rest = " ".join(s for k, s in enumerate(sents) if k != i)
    gold = _gold(item)
    gterms = {t for t in _terms(gold.key.replace("_", " ") + " " + gold.desc)
              if len(t) >= 4 or any(c.isdigit() for c in t)}
    rem_terms, rest_terms = _terms(removed), _terms(rest)
    # strict: algún término clave de la opción correcta aparece solo en la oración quitada
    strict = bool(gterms) and bool(gterms & (rem_terms - rest_terms))
    return DecisionItem(**{**item.__dict__, "id": item.id + ":noev", "state": rest,
                           "meta": {**item.meta, "unanswerable": True, "removed": removed,
                                    "strict": strict, "source_id": item.id}})


def build_g8_items(items: list[DecisionItem], families=("trap", "ambiguous", "adversarial")) -> list[DecisionItem]:
    """Variantes sin respuesta de los ítems de las familias dadas; omite los que no tienen evidencia léxica."""
    out = []
    for it in items:
        if it.family not in families or len(_sentences(state_text(it.state))) < 3:
            continue
        try:
            out.append(make_unanswerable(it))
        except (NoLexicalEvidence, ValueError):   # ValueError: vocabulario vacío del vectorizador
            continue
    return out
