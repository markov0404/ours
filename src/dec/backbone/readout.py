# src/dec/backbone/readout.py
"""Lecturas sin generación: logits de letras en la última posición, o suma de log-probs de sufijos con KV compartido."""
from __future__ import annotations
import copy
import numpy as np
import torch
import torch.nn.functional as F

def letters_probs(last_logits: torch.Tensor, code_token_ids: list[int], T: float) -> np.ndarray:
    sel = last_logits[code_token_ids].float() / T
    return F.softmax(sel, dim=-1).cpu().numpy()

def letter_token_ids(tokenizer, codes: list[str]) -> list[int]:
    ids = []
    for c in codes:
        t = tokenizer(c, add_special_tokens=False).input_ids
        if len(t) != 1:
            raise ValueError(f"el código {c!r} no es un token único: {t}")
        ids.append(t[0])
    return ids

def _expand_cache(cache, B: int) -> None:
    """Replica en sitio el batch (1 -> B) de una caché. transformers 5.19: DynamicCache.batch_repeat_interleave existe,
    pero las capas LinearAttentionLayer de Qwen3.5 (híbrido) no lo implementan y to/from_legacy_cache ya no existen;
    entonces se repite en dim 0 todo tensor (o lista de tensores) de cada capa."""
    try:
        cache.batch_repeat_interleave(B)
        return
    except AttributeError:
        pass
    for layer in cache.layers:
        for name, v in vars(layer).items():
            if isinstance(v, torch.Tensor) and v.dim() > 0 and v.shape[0] == 1:
                setattr(layer, name, v.repeat_interleave(B, dim=0))
            elif isinstance(v, dict) and v and all(isinstance(t, torch.Tensor) for t in v.values()):
                setattr(layer, name, {k: t.repeat_interleave(B, dim=0) for k, t in v.items()})
            elif isinstance(v, list) and v and all(isinstance(t, torch.Tensor) or t is None for t in v):
                setattr(layer, name, [None if t is None else t.repeat_interleave(B, dim=0) for t in v])

@torch.inference_mode()
def suffix_logprobs(model, tokenizer, prefix_ids: torch.Tensor, suffix_ids: list[list[int]], chunk: int = 16) -> np.ndarray:
    """prefix_ids: (1, L). Devuelve (K,) con sum_t log p(suffix_t | prefix, suffix_<t)."""
    dev = prefix_ids.device
    out = model(input_ids=prefix_ids, use_cache=True)
    past = out.past_key_values
    first_logp = F.log_softmax(out.logits[0, -1].float(), -1)          # predice el 1er token de cada sufijo
    scores = np.zeros(len(suffix_ids), dtype=np.float64)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    for start in range(0, len(suffix_ids), chunk):
        group = suffix_ids[start:start + chunk]; B = len(group); Lmax = max(len(s) for s in group)
        ids = torch.full((B, Lmax), pad, dtype=torch.long, device=dev)
        mask = torch.zeros((B, Lmax), dtype=torch.long, device=dev)
        for i, s in enumerate(group):
            ids[i, :len(s)] = torch.tensor(s, device=dev); mask[i, :len(s)] = 1
        for i, s in enumerate(group):
            scores[start + i] += float(first_logp[s[0]])
        if Lmax > 1:
            cache = copy.deepcopy(past)
            _expand_cache(cache, B)
            full_mask = torch.cat([torch.ones((B, prefix_ids.shape[1]), dtype=torch.long, device=dev), mask], 1)
            o = model(input_ids=ids, attention_mask=full_mask, past_key_values=cache, use_cache=True)
            logp = F.log_softmax(o.logits.float(), -1)                 # (B, Lmax, V); posición t predice token t+1
            for i, s in enumerate(group):
                for t in range(1, len(s)):
                    scores[start + i] += float(logp[i, t - 1, s[t]])
    return scores


@torch.inference_mode()
def packed_suffix_logprobs(model, prefix_ids: torch.Tensor, suffix_ids: list[list[int]], use_ltk: bool = True) -> np.ndarray:
    """Un solo forward sobre prefix + suf_1 + ... + suf_K. position_ids reinician en len(prefix) en cada sufijo y la
    máscara 4D aditiva impide que un sufijo atienda a otro. Requiere atención estándar (sdpa/eager) en todas las capas:
    las capas de atención lineal (Qwen3.5 híbrido) ignoran la máscara -> usar suffix_logprobs allí.
    prefix_ids: (1, P). Devuelve (K,) con sum_t log p(suffix_t | prefix, suffix_<t)."""
    dev = prefix_ids.device; P = prefix_ids.shape[1]
    lens = [len(s) for s in suffix_ids]; S = sum(lens); N = P + S
    flat = torch.tensor([t for s in suffix_ids for t in s], dtype=torch.long, device=dev)
    ids = torch.cat([prefix_ids[0], flat])[None]
    pos = torch.empty(N, dtype=torch.long, device=dev); pos[:P] = torch.arange(P, device=dev)
    allowed = torch.zeros((N, N), dtype=torch.bool, device=dev)
    allowed[:P, :P] = torch.tril(torch.ones((P, P), dtype=torch.bool, device=dev))
    allowed[P:, :P] = True
    offs = []; o = 0
    for L in lens:
        offs.append(o)
        pos[P + o:P + o + L] = torch.arange(P, P + L, device=dev)
        allowed[P + o:P + o + L, P + o:P + o + L] = torch.tril(torch.ones((L, L), dtype=torch.bool, device=dev))
        o += L
    dtype = next(model.parameters()).dtype
    mask = torch.zeros((N, N), dtype=dtype, device=dev).masked_fill(~allowed, torch.finfo(dtype).min)[None, None]
    kw = {"logits_to_keep": S + 1} if use_ltk else {}
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos[None], use_cache=False, **kw)
    lg = out.logits[0, -(S + 1):].float()                      # índice j <-> posición P-1+j
    logp = lg - torch.logsumexp(lg, -1, keepdim=True)
    rows, cols = [], []
    for s, o in zip(suffix_ids, offs):
        for t, tok in enumerate(s):
            rows.append(0 if t == 0 else o + t); cols.append(tok)
    vals = logp[torch.tensor(rows, device=dev), torch.tensor(cols, device=dev)].double().cpu().numpy()
    return np.add.reduceat(vals, np.cumsum([0] + lens[:-1]))
