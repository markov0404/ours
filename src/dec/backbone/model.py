"""Backbone congelado: una pasada por (estado, pregunta) -> p0 y h (pool del estado).

Encode por lotes con presupuesto de tokens y padding a la izquierda; el hidden final se captura con un forward hook
(sin output_hidden_states); la lectura por letras usa logits_to_keep=1; la lectura por sufijos va empaquetada en un
solo forward con máscara 4D (readout.packed_suffix_logprobs) cuando la arquitectura solo tiene atención estándar.
"""
from __future__ import annotations
import time
import inspect
import numpy as np
import torch
from loguru import logger
from dec._compat import patch_torch_accelerator
patch_torch_accelerator()   # antes de importar transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from dec.schema import DecisionItem, Encoded
from .prompt import build_prompt
from .readout import letter_token_ids, suffix_logprobs, packed_suffix_logprobs


def resolve_device(device: str = "auto") -> str:
    """`auto` = cuda si hay, si no mps (Apple), si no cpu; el resto se valida y se devuelve tal cual."""
    if device == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        return "mps" if mps is not None and mps.is_available() else "cpu"
    if device not in ("cuda", "cpu", "mps") and not device.startswith("cuda:"):
        raise ValueError(f"device desconocido: {device} (auto|cuda|cpu|mps)")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' pedido pero CUDA no está disponible; usar --device cpu o auto")
    return device


def resolve_dtype(dtype: str, device: str) -> str:
    """`auto` = bf16 en cuda, fp32 en cpu/mps (bf16 en CPU es lento o no soporta algunas operaciones)."""
    if dtype == "auto":
        return "bf16" if device.startswith("cuda") else "fp32"
    if dtype == "bf16" and not device.startswith("cuda"):
        logger.warning(f"dtype=bf16 en {device}: puede ser lento o fallar en algunas operaciones; se recomienda fp32 (o dtype='auto')")
    return dtype


def _load_model(model_id: str, dtype: str, revision: str | None = None, device: str = "cuda"):
    if dtype == "int8":
        raise NotImplementedError("dtype='int8' no implementado (requiere bitsandbytes); usar 'bf16' en GPU o 'fp32' en CPU")
    if dtype not in ("bf16", "fp32"):
        raise ValueError(f"dtype desconocido: {dtype}")
    kw = {"dtype": torch.bfloat16 if dtype == "bf16" else torch.float32, "revision": revision}
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, **kw).to(device)
    except Exception as e:  # arquitecturas multimodales (qwen3_5 ForConditionalGeneration)
        logger.warning(f"AutoModelForCausalLM falló ({type(e).__name__}: {e}); probando AutoModelForImageTextToText")
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(model_id, **kw).to(device)


def find_hook_target(model) -> tuple[str, torch.nn.Module]:
    """Norma final del decoder de texto (su salida == hidden_states[-1] de HF); si no hay, la última capa decoder."""
    best = None
    for name, m in model.named_modules():
        if isinstance(m, torch.nn.ModuleList) and name.split(".")[-1] == "layers" \
                and not any(s in name for s in ("visual", "vision", "audio")):
            if best is None or len(m) > len(best[1]):
                best = (name, m)
    if best is None:
        raise RuntimeError("no se encuentra la lista de capas decoder del modelo")
    parent_name = best[0].rsplit(".", 1)[0]
    parent = model.get_submodule(parent_name)
    norm = getattr(parent, "norm", None)
    if isinstance(norm, torch.nn.Module):
        return f"{parent_name}.norm", norm
    return f"{best[0]}[-1]", best[1][-1]


class Backbone:
    def __init__(self, model_id: str, dtype: str = "auto", style: str = "numeric", temperature: float = 1.0,
                 max_tokens: int = 8192, readout: str = "auto", revision: str | None = None, suffix_chunk: int = 16,
                 device: str = "auto"):
        """`device`: auto|cuda|cpu|mps (auto = cuda > mps > cpu). `dtype`: auto|bf16|fp32 (auto = bf16 en cuda, fp32 en el resto)."""
        self.device = resolve_device(device)
        dtype = resolve_dtype(dtype, self.device)
        self.model_id, self.dtype, self.style, self.T, self.max_tokens, self.readout = model_id, dtype, style, temperature, max_tokens, readout
        self.suffix_chunk = suffix_chunk   # sufijos por forward en el camino por trozos (atención híbrida); cada trozo copia la cache del prefijo
        self.tok = AutoTokenizer.from_pretrained(model_id, revision=revision, padding_side="left")
        self.model = _load_model(model_id, dtype, revision, self.device).eval()
        cfg = AutoConfig.from_pretrained(model_id, revision=revision)
        tc = cfg.get_text_config() if hasattr(cfg, "get_text_config") else getattr(cfg, "text_config", cfg)
        self.hidden_size = int(tc.hidden_size)
        self.backbone_id = f"{model_id}@{revision or 'main'}:{style}:{dtype}"
        self._letter_ids: dict[tuple[str, ...], list[int]] = {}
        self.hook_path, target = find_hook_target(self.model)
        self._hidden: torch.Tensor | None = None
        self._capture = True
        target.register_forward_hook(self._hook)
        self.use_ltk = "logits_to_keep" in inspect.signature(self.model.forward).parameters
        # el empaquetado necesita que TODAS las capas respeten una máscara 4D: no vale con atención lineal/híbrida
        layer_types = set(getattr(tc, "layer_types", None) or [])
        # el camino empaquetado solo se valida en CUDA; en cpu/mps se usa el camino por trozos (mismo resultado, más lento)
        self.hybrid = not (layer_types <= {"full_attention"}) or getattr(tc, "model_type", "") in ("mamba", "rwkv")
        self.can_pack = not self.hybrid and self.device.startswith("cuda")
        logger.info(f"Backbone {self.backbone_id} en {self.device}: hook={self.hook_path} logits_to_keep={self.use_ltk} packing={self.can_pack}")

    def _hook(self, module, args, output):
        if self._capture:
            self._hidden = output[0] if isinstance(output, tuple) else output

    def _forward(self, **kw):
        if self.use_ltk is False:
            kw.pop("logits_to_keep", None)
        return self.model(use_cache=False, **kw)

    @staticmethod
    def _span_positions(offsets: list[tuple[int, int]], char_span: tuple[int, int]) -> list[int]:
        a, b = char_span
        return [i for i, (s, e) in enumerate(offsets) if e > a and s < b and e > s]

    def _prepare(self, item: DecisionItem):
        pp = build_prompt(item, self.style, self.tok, self.max_tokens, self.readout)
        enc = self.tok(pp.prefix_text, return_offsets_mapping=True, add_special_tokens=True)
        pos = self._span_positions(enc["offset_mapping"], pp.state_char_span)   # offsets sobre el texto renderizado (jevk5: escapado)
        return pp, enc["input_ids"], pos

    def _pool(self, hs: torch.Tensor, pos: list[int], pad: int):
        """hs: (L_total, d) de una fila; pos: posiciones del estado sin padding; pad: padding a la izquierda."""
        if pos:
            idx = torch.tensor(pos, device=hs.device) + pad
            h = hs[idx].float().mean(0)
        else:
            h = hs[pad:].float().mean(0)
        return h.cpu().numpy(), hs[-1].float().cpu().numpy()

    @torch.inference_mode()
    def _encode_letters_batch(self, batch: list[tuple[int, object, list[int], list[int]]]) -> dict[int, Encoded]:
        Lmax = max(len(ids) for _, _, ids, _ in batch); B = len(batch)
        padid = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
        ids = torch.full((B, Lmax), padid, dtype=torch.long); mask = torch.zeros((B, Lmax), dtype=torch.long)
        for r, (_, _, x, _) in enumerate(batch):
            ids[r, Lmax - len(x):] = torch.tensor(x); mask[r, Lmax - len(x):] = 1
        position_ids = (mask.cumsum(1) - 1).clamp(min=0)
        dev = self.device
        out = self._forward(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=position_ids.to(dev), logits_to_keep=1)
        hs = self._hidden; res = {}
        for r, (i, pp, x, pos) in enumerate(batch):
            h, h_last = self._pool(hs[r], pos, Lmax - len(x))
            key = tuple(pp.option_codes)
            if key not in self._letter_ids:
                self._letter_ids[key] = letter_token_ids(self.tok, pp.option_codes)
            logits = out.logits[r, -1][self._letter_ids[key]].float()
            p0 = torch.softmax(logits / self.T, -1).cpu().numpy()
            res[i] = (pp, p0, logits.cpu().numpy(), h, h_last)
        return res

    @torch.inference_mode()
    def _encode_suffix_one(self, pp, x: list[int], pos: list[int]):
        suf = [self.tok(s, add_special_tokens=False).input_ids for s in pp.suffixes]
        ids = torch.tensor([x], device=self.device)
        if self.can_pack:
            logits = packed_suffix_logprobs(self.model, ids, suf, use_ltk=self.use_ltk)
            hs = self._hidden[0, :len(x)]
        else:                                   # fallback por chunks (atención híbrida): prefill extra para el hidden
            self._forward(input_ids=ids, logits_to_keep=1)
            hs = self._hidden[0]
            self._capture = False
            try:
                logits = suffix_logprobs(self.model, self.tok, ids, suf, chunk=self.suffix_chunk)
            finally:
                self._capture = True
        h, h_last = self._pool(hs, pos, 0)
        z = logits / self.T; z = z - z.max(); p0 = np.exp(z) / np.exp(z).sum()
        return pp, p0, np.asarray(logits), h, h_last

    def _sync(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def default_token_budget(self) -> int:
        """Presupuesto de tokens por lote: batch-1 en arquitecturas híbridas (Qwen3.5/JevK5: la atención lineal es más lenta por lotes)."""
        return 1 if self.hybrid else 16384

    def encode_one(self, item: DecisionItem) -> Encoded:
        return self.encode([item], log_every=10**9)[0]

    def encode(self, items: list[DecisionItem], log_every: int = 50, token_budget: int = 16384) -> list[Encoded]:
        t0 = time.time(); prep = [self._prepare(it) for it in items]
        raw: dict[int, tuple] = {}; done = 0; ntok = 0
        order = sorted(range(len(items)), key=lambda i: len(prep[i][1]))
        letters = [i for i in order if prep[i][0].readout in ("letters", "codes")]   # ambos leen logits de un token
        batches, cur = [], []
        for i in letters:                        # ascendente: la longitud máxima del lote es la del último añadido
            if cur and (len(cur) + 1) * len(prep[i][1]) > token_budget:
                batches.append(cur); cur = []
            cur.append(i)
        if cur:
            batches.append(cur)
        batches += [[i] for i in order if prep[i][0].readout not in ("letters", "codes")]
        next_log = log_every
        for b in batches:
            tb = time.time()
            if prep[b[0]][0].readout in ("letters", "codes"):
                raw.update(self._encode_letters_batch([(i, prep[i][0], prep[i][1], prep[i][2]) for i in b]))
                n = len(b) * max(len(prep[i][1]) for i in b)
            else:
                i = b[0]; raw[i] = self._encode_suffix_one(prep[i][0], prep[i][1], prep[i][2]); n = len(prep[i][1])
            self._sync(); dt = time.time() - tb
            ntok += n; done += len(b)
            logger.debug(f"lote {len(b)} ítems · {n} tokens · {n/dt:.0f} tok/s")
            if done >= next_log:
                el = time.time() - t0; next_log += log_every
                logger.info(f"{done}/{len(items)} · {el/60:.1f} min · {ntok/el:.0f} tok/s · ETA {(len(items)-done)*el/done/60:.1f} min")
        out = []
        for i, it in enumerate(items):
            pp, p0, logits, h, h_last = raw[i]
            out.append(Encoded(it.id, p0.astype(np.float32), np.asarray(logits, np.float32), h.astype(np.float32),
                               h_last.astype(np.float32), pp.readout, pp.truncated))
        logger.info(f"encode: {len(items)} ítems en {(time.time()-t0)/60:.2f} min")
        return out
