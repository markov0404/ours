"""Compatibilidad de entorno para correr en máquinas sin GPU."""
from __future__ import annotations


def patch_torch_accelerator() -> None:
    """torch 2.6 con build CUDA y sin GPU visible (CUDA_VISIBLE_DEVICES vacío, o sin driver): `torch.accelerator.current_accelerator()`
    lanza RuntimeError en vez de devolver None, y transformers lo llama al importarse (`get_device_type`). Se envuelve para devolver
    None, que es lo que hacen las versiones posteriores; en el resto de los casos no cambia nada."""
    import torch
    acc = getattr(torch, "accelerator", None)
    fn = getattr(acc, "current_accelerator", None)
    if fn is None or getattr(fn, "_dec_safe", False):
        return
    try:
        fn()
        return
    except RuntimeError:
        pass

    def safe(*a, **k):
        try:
            return fn(*a, **k)
        except RuntimeError:
            return None
    safe._dec_safe = True
    acc.current_accelerator = safe


def cpu_quota() -> int | None:
    """vCPUs efectivas por cuota de cgroup (Docker --cpus, Kubernetes, pods), o None si no hay cuota."""
    import math
    from pathlib import Path
    try:
        p2 = Path("/sys/fs/cgroup/cpu.max")
        if p2.exists():
            q, per = p2.read_text().split()[:2]
            return None if q == "max" else max(1, math.floor(int(q) / int(per)))
        q1, p1 = Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"), Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
        if q1.exists() and p1.exists():
            q, per = int(q1.read_text()), int(p1.read_text())
            return None if q <= 0 else max(1, math.floor(q / per))
    except (OSError, ValueError):
        return None
    return None


def set_cpu_threads(threads: int | None) -> int:
    """Fija los hilos de torch: `threads` si se pide; si no, la cuota de cgroup cuando es menor que el default de torch
    (con cuota, más hilos que vCPUs reales multiplica la latencia: medido 4.5 s vs 1.7 s por decisión con JevK5)."""
    import torch
    n = threads or cpu_quota()
    if n and (threads or n < torch.get_num_threads()):
        torch.set_num_threads(n)
    return torch.get_num_threads()
