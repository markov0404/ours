"""Métricas aprendidas sobre features cacheadas: diagonal global (Ours-1) y φ condicionada por pregunta (Ours-2)."""
from __future__ import annotations
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F

class GlobalDiagonalMetric(nn.Module):
    def __init__(self, d: int):
        super().__init__(); self.log_w = nn.Parameter(torch.zeros(d))
    def weights(self, q: torch.Tensor | None = None) -> torch.Tensor:
        w = F.softplus(self.log_w); return w / w.mean()
    def mask(self, q_emb: np.ndarray | None = None) -> np.ndarray:
        with torch.no_grad(): return self.weights().cpu().numpy().astype(np.float32)

class PhiMask(nn.Module):
    """Prototipos por dimensión; la pregunta elige top-J dims (QCD φ). Soft en entrenamiento, dura en inferencia (STE)."""
    def __init__(self, d: int, d_q: int, d_p: int = 128, J: int = 256):
        super().__init__()
        if J > d: raise ValueError(f"J={J} > d={d}")
        self.J = J
        self.P = nn.Parameter(torch.randn(d, d_p) * 0.05); self.Wq = nn.Linear(d_q, d_p, bias=True)
    def scores(self, q: torch.Tensor) -> torch.Tensor:        # (B, d)
        return self.Wq(q) @ self.P.T
    def weights(self, q: torch.Tensor) -> torch.Tensor:        # (B, d) soft mask + STE hacia top-J
        s = self.scores(q); soft = torch.sigmoid(s)
        hard = torch.zeros_like(soft).scatter_(1, s.topk(self.J, dim=1).indices, 1.0)
        return hard + soft - soft.detach()
    def mask(self, q_emb: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            s = self.scores(torch.as_tensor(q_emb, dtype=torch.float32)[None])[0]
            m = torch.zeros_like(s).scatter_(0, s.topk(self.J).indices, 1.0)
        return m.cpu().numpy().astype(np.float32)

def _pairwise(h: torch.Tensor, C: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    # h (B,d) anclas, C (B,M,d) candidatos, w (B,d) pesos -> (B,M) distancias
    return torch.sqrt((((C - h[:, None]) ** 2) * w[:, None]).sum(-1) + 1e-9)

def fit_nca(metric: nn.Module, H: np.ndarray, y: list[str], qkeys: list[str], Q: np.ndarray | None,
            epochs: int = 20, lr: float = 1e-2, batch: int = 256, cands: int = 1024, seed: int = 0) -> dict:
    """NCA por pregunta. Estandariza H internamente (devuelve `mu`, `sd` por dim). PrecedentMemory estandariza por
    partición y aplica las máscaras en espacio estandarizado, coherente con los pesos aprendidos; quien aplique
    máscaras en otro sitio debe estandarizar con mu/sd."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    mu = H.mean(0); sd = H.std(0) + 1e-6
    Ht = torch.as_tensor((H - mu) / sd, dtype=torch.float32)
    yi = torch.as_tensor(np.unique(y, return_inverse=True)[1]); qarr = np.asarray(qkeys)
    Qt = None if Q is None else torch.as_tensor(Q, dtype=torch.float32)
    log_tau = nn.Parameter(torch.tensor(0.0)); opt = torch.optim.Adam(list(metric.parameters()) + [log_tau], lr=lr)
    groups = {q: np.where(qarr == q)[0] for q in np.unique(qarr)}; hist = []
    valid = np.array([len(groups[k]) >= 2 for k in qarr])  # singletons nunca tienen positivos
    for _ in range(epochs):
        perm = rng.permutation(np.where(valid)[0]); tot = 0.0; nb = 0
        for s in range(0, len(perm), batch):
            idx = perm[s:s + batch]
            M = min(cands, min(len(groups[k]) for k in set(qarr[idx])))  # por batch (apilable)
            cand = np.stack([rng.choice(groups[qarr[i]], size=M, replace=False) for i in idx])
            h = Ht[idx]; C = Ht[cand]
            w = metric.weights(Qt[idx]) if Qt is not None else metric.weights()[None].expand(len(idx), -1)
            d = _pairwise(h, C, w) / torch.exp(log_tau)
            same_item = torch.as_tensor(cand) == torch.as_tensor(idx)[:, None]
            d = d.masked_fill(same_item, 1e4)  # finito: ninguna fila queda toda -inf
            logp = F.log_softmax(-d, dim=1)
            pos = (yi[torch.as_tensor(cand)] == yi[idx][:, None]) & ~same_item
            ok = pos.any(1)
            if not ok.any(): continue
            lp = torch.logsumexp(logp.masked_fill(~pos, -1e4), dim=1)
            loss = -lp[ok].clamp(min=-30).mean()
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss.detach()); nb += 1
        hist.append(tot / max(1, nb))
    return {"loss": hist, "tau": float(torch.exp(log_tau.detach())), "mu": mu, "sd": sd}

def embed_questions(texts: list[str], cache: Path = Path("cache/qemb.npz")) -> np.ndarray:
    """MiniLM normalizado; cachea por texto en `cache` (npz con claves texts/embs)."""
    cache = Path(cache); store: dict[str, np.ndarray] = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False); store = dict(zip(z["texts"].tolist(), z["embs"]))
    miss = [t for t in dict.fromkeys(texts) if t not in store]
    if miss:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        for t, e in zip(miss, model.encode(miss, convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)):
            store[t] = e
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, texts=np.array(list(store), dtype=str), embs=np.stack(list(store.values())))
    return np.stack([store[t] for t in texts]).astype(np.float32)
