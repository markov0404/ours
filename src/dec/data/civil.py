"""google/civil_comments: 7 atributos continuos -> 7 preguntas noul por comentario.

Umbrales de gold: 0.5 para todos los atributos salvo severe_toxicity, que usa 0.3
porque en civil_comments ese atributo nunca supera 0.4 (con 0.5 no habría positivos).
Hasta RARE_MAX filas con severe_toxicity >= 0.3 se fuerzan en la muestra; son las
primeras en orden de dataset, por lo que son idénticas entre semillas.
Consecuencia: el gold de severe_toxicity difiere del ">= 0.5" del brief y la
implicación severe_toxicity => toxicity tiene soporte de solo ~21 filas en test.
"""
from __future__ import annotations
import random
from datasets import load_dataset
from dec.schema import DecisionItem, Option
from . import register

ATTRS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
QUESTION = {
    "toxicity": "Is this comment toxic (rude, disrespectful or unreasonable)?",
    "severe_toxicity": "Is this comment severely toxic (hateful, aggressive, likely to make someone leave)?",
    "obscene": "Does this comment contain obscene language?",
    "threat": "Does this comment contain a threat?",
    "insult": "Does this comment contain an insult?",
    "identity_attack": "Does this comment attack someone based on identity?",
    "sexual_explicit": "Is this comment sexually explicit?",
}

# En civil_comments severe_toxicity nunca supera 0.4, así que con 0.5 no habría positivos.
THRESH = {a: 0.5 for a in ATTRS}
THRESH["severe_toxicity"] = 0.3
RARE_MAX = 200  # filas con severe_toxicity>=umbral que se fuerzan en la muestra (rarísimas)

def _yes(r: dict, a: str) -> bool:
    return r[a] >= THRESH[a]

def _implications(rows: list[dict]) -> list[tuple[str, str]]:
    out = []
    for a in ATTRS:
        if a == "toxicity": continue
        pos = [r for r in rows if _yes(r, a)]
        if len(pos) >= 20 and sum(r["toxicity"] >= THRESH["toxicity"] for r in pos) / len(pos) >= 0.995:
            out.append((f"civil:{a}", "civil:toxicity"))
    return out

@register("civil")
def load_civil(split: str, n: int = 12000, seed: int = 0) -> list[DecisionItem]:
    ds = load_dataset("google/civil_comments", split=split)
    rng = random.Random(seed)
    idx = list(range(len(ds))); rng.shuffle(idx)
    sev = list(ds["severe_toxicity"])  # Column perezosa: indexarla 1.8M veces tarda ~85 min
    forced = [i for i in range(len(sev)) if sev[i] >= THRESH["severe_toxicity"]][:RARE_MAX]
    fset = set(forced)
    pos, neg = [], []
    for i in forced:
        r = ds[i]
        (pos if r["toxicity"] >= THRESH["toxicity"] else neg).append(r)
    for i in idx:
        if i in fset: continue
        r = ds[i]
        (pos if r["toxicity"] >= THRESH["toxicity"] else neg).append(r)
        if len(pos) >= n // 2 and len(neg) >= n - n // 2: break
    rows = pos[: n // 2] + neg[: n - len(pos[: n // 2])]
    rng.shuffle(rows)
    imps = _implications(rows)
    opts = [Option("yes", "Yes"), Option("no", "No")]
    items = []
    for j, r in enumerate(rows):
        sid = f"civil:{split}:{seed}:{j}"
        for a in ATTRS:
            items.append(DecisionItem(id=f"{sid}:{a}", dataset="civil", state_id=sid, state=r["text"], qtype="noul",
                instructions=QUESTION[a], options=opts, gold="yes" if _yes(r, a) else "no",
                question_key=f"civil:{a}", domain="moderation", family="noul",
                meta={"score": float(r[a]), "implications": imps}))
    return items
