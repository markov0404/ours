# src/dec/backbone/prompt.py
"""Construcción de prompts: estilo 'numeric' (propio, lectura por sufijos o letras) y 'jevk5' (plantilla copiada)."""
from __future__ import annotations
import json
import string
from dataclasses import dataclass
from dec.schema import DecisionItem, state_text
from . import jevk5_prompt

LETTERS = "ABCDEFGHIJKLMNOPQRST"   # 20 códigos de una letra

@dataclass
class PromptParts:
    prefix_text: str
    option_codes: list[str]
    suffixes: list[str] | None
    state_char_span: tuple[int, int]
    truncated: bool
    style: str
    readout: str

def auto_readout(n_options: int, n_codes: int = 0) -> str:
    """letters (A..T) hasta 20 opciones; codes (dos letras mayúsculas, un token) hasta `n_codes`; sufijos numéricos después."""
    if n_options <= len(LETTERS):
        return "letters"
    return "codes" if n_options <= n_codes else "suffix"


_ALPHABETS: dict[str, list[str]] = {}


def code_alphabet(tokenizer) -> list[str]:
    """Códigos de dos letras mayúsculas que el tokenizador representa con UN token (ids distintos), en orden alfabético.

    Permiten leer >20 opciones (intents: 60/77/151) en una sola pasada, como las letras; el camino de sufijos costaba
    dos prefills + ~10 pasadas por ítem en Qwen3.5. Las letras van en su propio chunk BPE, así que la tokenización
    de "AB" no depende del contexto "[AB]".
    """
    key = getattr(tokenizer, "name_or_path", None) or str(id(tokenizer))
    if key not in _ALPHABETS:
        seen: set[int] = set(); out: list[str] = []
        for a in string.ascii_uppercase:
            for b in string.ascii_uppercase:
                t = tokenizer(a + b, add_special_tokens=False).input_ids
                if len(t) == 1 and t[0] not in seen:
                    seen.add(t[0]); out.append(a + b)
        _ALPHABETS[key] = out
    return _ALPHABETS[key]

def _truncate_state(state: str, tokenizer, budget: int) -> tuple[str, bool]:
    ids = tokenizer(state, add_special_tokens=False).input_ids
    if len(ids) <= budget:
        return state, False
    head = int(budget * 0.6); tail = budget - head
    return tokenizer.decode(ids[:head]) + "\n[...]\n" + tokenizer.decode(ids[-tail:]), True

HEADER = "You are a decision model. Read the state and answer the question by choosing exactly one option. Reply with the option tag only.\n\nState:\n"

def build_prompt(item: DecisionItem, style: str, tokenizer, max_tokens: int = 8192, readout: str = "auto") -> PromptParts:
    n = len(item.options)
    alphabet = code_alphabet(tokenizer) if (readout in ("auto", "codes") and n > len(LETTERS)) else []
    ro = auto_readout(n, len(alphabet)) if readout == "auto" else readout
    if style == "jevk5" and n > 16:
        # jevk5 lee como máximo 16 opciones en una pasada (knockout no implementado): los datasets de intents (18-151
        # etiquetas) caen al prompt numeric con la lectura automática (letras <= 20, sufijos si no). Queda registrado en
        # Encoded.readout; el backbone_id no cambia.
        style = "numeric"
    if style == "jevk5":
        # jevk5 admite como máximo 16 opciones por pasada (knockout/tree no implementados): render() lanza ValueError.
        text = jevk5_prompt.render(item)
        # El estado va JSON-escapado dentro del turno de usuario: el span está en el texto ESCAPADO (su longitud puede
        # diferir de len(state)); la Task 10 debe mapear tokens sobre ese fragmento, no sobre item.state.
        esc = json.dumps(item.state, ensure_ascii=False)[1:-1]
        u = text.find("<|im_start|>user")
        a = text.find(esc, u) if u >= 0 else -1
        if a < 0:
            raise ValueError("no se encuentra el estado (escapado) en el prompt jevk5")
        return PromptParts(text, list(jevk5_prompt.CONFIG["letters"][:n]), None, (a, a + len(esc)), False, style, "letters")
    if style != "numeric":
        raise ValueError(style)
    # Estilo numeric: letras A..T si la lectura es por letras (token único), dígitos "1".."n" si es por sufijos.
    if ro == "letters" and n > len(LETTERS):
        raise ValueError(f"lectura por letras admite como máximo {len(LETTERS)} opciones (hay {n})")
    if ro == "codes" and n > len(alphabet):
        raise ValueError(f"lectura por códigos admite como máximo {len(alphabet)} opciones con este tokenizador (hay {n})")
    codes = list(LETTERS[:n]) if ro == "letters" else alphabet[:n] if ro == "codes" else [str(i + 1) for i in range(n)]
    lines = [f"[{c}] {o.key}: {o.desc}".rstrip(": ") for c, o in zip(codes, item.options)]
    tail = f"\n\nQuestion: {item.instructions}\nOptions:\n" + "\n".join(lines) + "\nAnswer: ["
    fixed = len(tokenizer(HEADER + tail, add_special_tokens=False).input_ids)
    budget = max(256, max_tokens - fixed - 8)
    state, truncated = _truncate_state(state_text(item.state), tokenizer, budget)
    prefix = HEADER + state + tail
    a = len(HEADER)
    suffixes = None if ro in ("letters", "codes") else [f"{c}]" for c in codes]
    return PromptParts(prefix, codes, suffixes, (a, a + len(state)), truncated, style, ro)
