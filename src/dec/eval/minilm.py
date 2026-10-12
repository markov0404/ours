"""Embeddings MiniLM de estados (baseline B2), cacheados por dataset."""
from __future__ import annotations
from pathlib import Path
import hashlib, os
import numpy as np
from dec.schema import DecisionItem, state_text


MINILM_ID = "sentence-transformers/all-MiniLM-L6-v2"
_MODEL = None
_DEVICE: str | None = None   # None = elección de sentence-transformers (cuda si hay)


def set_device(device: str | None) -> None:
    """Fija el device del MiniLM (cpu/cuda/mps) antes de cargarlo; si ya estaba cargado en otro device, se recarga."""
    global _MODEL, _DEVICE
    if device != _DEVICE:
        _DEVICE, _MODEL = device, None


def _load_model():
    """Carga (una sola vez por proceso) el MiniLM compartido por embed_states, embed_sentences y embed_texts."""
    global _MODEL
    if _MODEL is None:
        from dec._compat import patch_torch_accelerator
        patch_torch_accelerator()
        from sentence_transformers import SentenceTransformer
        _MODEL = SentenceTransformer(MINILM_ID, device=_DEVICE)
    return _MODEL


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embeddings MiniLM normalizados (float32) de textos, SIN cache en disco.

    Es el camino de producción (pool del cliente y peticiones de dec-serve): `embed_states` cachea por dataset/state_id y en el
    servidor todas las peticiones comparten dataset `serve` e ids repetidos, así que devolvería el embedding de otra petición.
    """
    if not texts:
        return np.zeros((0, 384), np.float32)
    return _load_model().encode(list(texts), convert_to_numpy=True, normalize_embeddings=True, batch_size=batch_size).astype(np.float32)


def embed_states(items: list[DecisionItem]) -> np.ndarray:
    """Embeddings MiniLM normalizados de `item.state`, cacheados en cache/minilm/<dataset>.npz por state_id."""
    cache = Path("cache/minilm"); cache.mkdir(parents=True, exist_ok=True)
    ds = items[0].dataset; p = cache / f"{ds}.npz"; store = dict(np.load(p)) if p.exists() else {}
    todo = sorted({it.state_id: state_text(it.state) for it in items if it.state_id not in store}.items())
    if todo:
        m = _load_model()
        E = m.encode([s for _, s in todo], convert_to_numpy=True, normalize_embeddings=True, batch_size=64)
        store.update({sid: e.astype(np.float32) for (sid, _), e in zip(todo, E)}); np.savez(p, **store)
    return np.stack([store[it.state_id] for it in items])


def embed_sentences(texts: list[str], cache: Path | str = Path("cache/minilm/sentences")) -> np.ndarray:
    """Embeddings MiniLM normalizados (float32) de frases sueltas, cacheados por sha1 del texto.

    Formato: directorio `cache` con shards `<sha1 del lote>.npz` (claves `keys`: sha1 de cada texto, `embs`: matriz). Solo se
    escribe un shard nuevo con los textos que faltan (escritura atómica tmp+replace), así que es append-safe y no reescribe lo viejo.
    """
    cache = Path(cache); cache.mkdir(parents=True, exist_ok=True)
    keys = [hashlib.sha1(t.encode("utf-8")).hexdigest() for t in texts]
    store: dict[str, np.ndarray] = {}
    for f in sorted(cache.glob("*.npz")):
        try:
            z = np.load(f, allow_pickle=False); store.update(zip(z["keys"].tolist(), z["embs"]))
        except Exception:
            continue   # shard corrupto o a medio escribir: se recalcula
    miss = {k: t for k, t in zip(keys, texts) if k not in store}
    if miss:
        E = _load_model().encode(list(miss.values()), convert_to_numpy=True, normalize_embeddings=True, batch_size=128).astype(np.float32)
        store.update(zip(miss, E))
        name = hashlib.sha1("".join(sorted(miss)).encode()).hexdigest()
        tmp = cache / f".{name}.tmp.npz"
        np.savez(tmp, keys=np.array(list(miss), dtype=str), embs=E); os.replace(tmp, cache / f"{name}.npz")
    if not texts:
        return np.zeros((0, 384), np.float32)
    return np.stack([store[k] for k in keys]).astype(np.float32)
