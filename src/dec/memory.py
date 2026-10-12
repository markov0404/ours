"""Memoria de precedentes: FAISS plano por question_key + rerank por distancia en subespacio (QCD P4 con vecinos reales)."""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import faiss

@dataclass
class Standardizer:
    mean: np.ndarray | None = None
    std: np.ndarray | None = None
    def fit(self, H: np.ndarray) -> "Standardizer":
        self.mean = H.mean(0); self.std = H.std(0) + 1e-6; return self
    def transform(self, H: np.ndarray) -> np.ndarray:
        return ((H - self.mean) / self.std).astype(np.float32)

@dataclass
class MemoryHit:
    p_mem: dict[str, float]
    mass: float
    precedents: list[tuple[str, str, float]]
    dists: np.ndarray = field(default_factory=lambda: np.zeros(0))

class _Partition:
    def __init__(self, d: int):
        self.d = d; self.H: list[np.ndarray] = []; self.labels: list[str] = []; self.ids: list[str] = []
        self.index = None; self.Hn = None; self.Hz = None; self.std = Standardizer()
    def build(self, device: str = "cpu") -> None:
        H = np.stack(self.H).astype(np.float32); self.Hz = self.std.fit(H).transform(H)
        self.Hn = self.Hz / (np.linalg.norm(self.Hz, axis=1, keepdims=True) + 1e-9)
        self.index = faiss.IndexFlatIP(self.d); self.index.add(self.Hn)
        self.id_index = {i: k for k, i in enumerate(self.ids)}
        if device != "cpu":   # copia en GPU: candidatos por producto interno y rerank con máscara en dos kernels (< 1 ms)
            import torch
            self.Hz_t = torch.as_tensor(self.Hz, device=device); self.Hn_t = torch.as_tensor(self.Hn, device=device)

class PrecedentMemory:
    def __init__(self, d: int, k: int = 32, candidates: int = 200, tau: float | None = None, device: str = "cpu"):
        """`device="cuda"` replica la búsqueda (candidatos IP + rerank con máscara) en GPU; mismos resultados, latencia sub-ms."""
        self.d, self.k, self.candidates, self.tau, self.device = d, k, candidates, tau, device
        self.parts: dict[str, _Partition] = {}

    def add(self, h: np.ndarray, label: str, question_key: str, item_id: str) -> None:
        p = self.parts.setdefault(question_key, _Partition(self.d)); p.H.append(np.asarray(h, np.float32)); p.labels.append(label); p.ids.append(item_id)

    def add_many(self, H, labels, qkeys, ids) -> None:
        for h, l, q, i in zip(H, labels, qkeys, ids): self.add(h, l, q, i)

    def build(self) -> None:
        for p in self.parts.values(): p.build(self.device)

    def size(self, question_key: str) -> int:
        return len(self.parts[question_key].ids) if question_key in self.parts else 0

    def class_freq(self, question_key: str) -> dict[str, float]:
        """Frecuencia relativa de cada etiqueta en la partición (prior empírico de la memoria; hip-13)."""
        if question_key not in self.parts: return {}
        labs = self.parts[question_key].labels; n = len(labs)
        return {l: labs.count(l) / n for l in set(labs)} if n else {}

    def labels(self, question_key: str) -> list[str]:
        return sorted(set(self.parts[question_key].labels)) if question_key in self.parts else []

    def calibrate_tau(self, H: np.ndarray, qkeys: list[str], ids: list[str], masks: list | None = None,
                      sample: int = 2000, seed: int = 0) -> float | None:
        """Fija τ para toda la memoria: mediana de las distancias LOO a los k vecinos de una muestra de precedentes.

        Sin τ fija, `search` usa la mediana de las distancias de la PROPIA consulta: invariante a la escala, la masa queda
        ~constante (≈ k/e) y deja de distinguir consultas con precedentes cercanos de consultas lejanas (OOS).
        """
        rng = np.random.default_rng(seed); idx = rng.permutation(len(ids))[:sample]; ds = []
        for i in idx:
            hit = self.search(H[i], qkeys[i], mask=None if masks is None else masks[i], exclude_id=ids[i], tau=1.0)
            if len(hit.dists):
                ds.append(np.asarray(hit.dists, np.float64))
        self.tau = float(np.median(np.concatenate(ds))) + 1e-6 if ds else None
        return self.tau

    def _search_gpu(self, p, z: np.ndarray, zn: np.ndarray, mask, k: int, exclude_id, tau, C: int) -> MemoryHit:
        """Misma lógica que la rama CPU (candidatos por IP sobre Hn, rerank euclídeo con máscara sobre Hz) en torch/GPU."""
        import torch
        dev = p.Hz_t.device
        zt = torch.as_tensor(z, device=dev); znt = torch.as_tensor(zn, device=dev)
        ip = p.Hn_t @ znt
        if exclude_id is not None and exclude_id in p.id_index:
            ip[p.id_index[exclude_id]] = -float("inf")
        cand = torch.topk(ip, min(C, ip.shape[0])).indices
        if exclude_id is not None and exclude_id in p.id_index:
            cand = cand[cand != p.id_index[exclude_id]]
        if cand.numel() == 0:
            return MemoryHit({}, 0.0, [])
        diff = p.Hz_t[cand] - zt[None]
        w = torch.ones(self.d, device=dev) if mask is None else torch.as_tensor(np.asarray(mask, np.float32), device=dev)
        d = torch.sqrt(((diff ** 2) * w[None]).sum(1) + 1e-12)
        kk = min(k, d.shape[0]); dk, order = torch.topk(d, kk, largest=False)
        d_np = dk.cpu().numpy(); cand_np = cand[order].cpu().numpy().tolist()
        t = tau or self.tau or (float(np.median(d_np)) + 1e-6)
        wts = np.exp(-d_np / t); mass = float(wts.sum()); wn = wts / (mass + 1e-12)
        p_mem: dict[str, float] = {}
        for c, wi in zip(cand_np, wn): p_mem[p.labels[c]] = p_mem.get(p.labels[c], 0.0) + float(wi)
        return MemoryHit(p_mem, mass, [(p.ids[c], p.labels[c], float(wi)) for c, wi in zip(cand_np, wts)], d_np)

    def search(self, h: np.ndarray, question_key: str, mask: np.ndarray | None = None, k: int | None = None,
               exclude_id: str | None = None, tau: float | None = None) -> MemoryHit:
        p = self.parts.get(question_key)
        if p is None or p.index is None or not p.ids:
            return MemoryHit({}, 0.0, [])
        k = k or self.k
        z = p.std.transform(np.asarray(h, np.float32)[None])[0]; zn = z / (np.linalg.norm(z) + 1e-9)
        C = min(self.candidates + 1, len(p.ids))
        if self.device != "cpu":
            return self._search_gpu(p, z, zn, mask, k, exclude_id, tau, C)
        _, cand = p.index.search(zn[None].astype(np.float32), C)
        cand = [c for c in cand[0] if c >= 0 and p.ids[c] != exclude_id]
        if not cand:
            return MemoryHit({}, 0.0, [])
        diff = p.Hz[cand] - z[None]
        w = np.ones(self.d, np.float32) if mask is None else np.asarray(mask, np.float32)
        d = np.sqrt(((diff ** 2) * w[None]).sum(1) + 1e-12)
        order = np.argsort(d)[:k]; d = d[order]; cand = [cand[i] for i in order]
        t = tau or self.tau or (np.median(d) + 1e-6)
        wts = np.exp(-d / t); mass = float(wts.sum()); wn = wts / (mass + 1e-12)
        p_mem: dict[str, float] = {}
        for c, wi in zip(cand, wn): p_mem[p.labels[c]] = p_mem.get(p.labels[c], 0.0) + float(wi)
        return MemoryHit(p_mem, mass, [(p.ids[c], p.labels[c], float(wi)) for c, wi in zip(cand, wts)], d)
