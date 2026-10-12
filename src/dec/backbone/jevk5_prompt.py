# Adaptado de jevk5 0.3.3 (Apache-2.0), https://github.com/allebee/jevk5  (jevk5/prompt.py: SYSTEM, CHAT_TEMPLATE,
# messages, prompt_text, decision_options copiados verbatim; render() es nuestro adaptador).
from __future__ import annotations
import json

from dec.schema import DecisionItem

# jevk5_config.json de alibiserikbay/JevK5 (temperature 1.22, knockout_temperature 0.93).
CONFIG = {"temperature": 1.22, "knockout_temperature": 0.93, "letters": "ABCDEFGHIJKLMNOP", "uses_chat_template": True}
LETTERS = CONFIG["letters"]
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
# Qwen3.5's chat template with thinking off, rendered once and pinned here.
CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)


def messages(state, criterion: str, options: list[str]) -> list[dict]:
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [{"letter": LETTERS[i], "description": d} for i, d in enumerate(options)],
    }
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def prompt_text(state, criterion: str, options: list[str]) -> str:
    """The full prompt string, chat template included, for front ends without a template engine."""
    system, user = messages(state, criterion, options)
    return CHAT_TEMPLATE.format(system=system["content"], user=user["content"])


def decision_options(question: dict) -> list[tuple[str, str]]:
    """(option id, option text) for a typed question: noul -> true/false, choice -> its
    criteria, score -> level indices. Texts are "id: description", as SemIf's JevBench mapping."""
    crit = question.get("criteria")
    if question["type"] == "noul":
        pairs = [(k, (crit or {}).get(k) or f"The proposition is {k}.") for k in ("true", "false")]
    elif question["type"] == "choice":
        if isinstance(crit, list):
            crit = dict.fromkeys(crit)
        pairs = [(k, v or k) for k, v in crit.items()]
    else:
        pairs = [(str(i), level) for i, level in enumerate(crit)]
    return [(k, f"{k}: {d}") for k, d in pairs]


def _question(item: DecisionItem) -> dict:
    if item.qtype == "noul":
        d = {o.key: o.desc for o in item.options}
        return {"type": "noul", "instructions": item.instructions,
                "criteria": {"true": d.get("yes") or None, "false": d.get("no") or None}}
    if item.qtype == "score":
        return {"type": "score", "instructions": item.instructions, "criteria": [o.desc or o.key for o in item.options]}
    return {"type": "choice", "instructions": item.instructions, "criteria": {o.key: o.desc for o in item.options}}


def render(item: DecisionItem) -> str:
    if len(item.options) > len(LETTERS):
        raise ValueError(f"jevk5 admite como máximo {len(LETTERS)} opciones por pasada (hay {len(item.options)})")
    q = _question(item)
    return prompt_text(item.state, q["instructions"], [t for _, t in decision_options(q)])
