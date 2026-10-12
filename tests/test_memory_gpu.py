"""La búsqueda en GPU replica la de CPU (candidatos IP + rerank con máscara), con y sin exclude_id."""
import numpy as np, pytest, torch
from dec.memory import PrecedentMemory


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
def test_gpu_search_matches_cpu():
    rng = np.random.default_rng(0); d, n = 64, 2000
    H = rng.normal(size=(n, d)).astype(np.float32); labels = [f"c{int(x)}" for x in rng.integers(0, 4, n)]; ids = [f"i{i}" for i in range(n)]
    cpu, gpu = PrecedentMemory(d, k=16), PrecedentMemory(d, k=16, device="cuda")
    for m in (cpu, gpu):
        m.add_many(H, labels, ["q"] * n, ids); m.build(); m.tau = 5.0
    mask = rng.uniform(0.2, 1.0, d).astype(np.float32)
    for q in range(20):
        h = H[q] + 0.3 * rng.normal(size=d).astype(np.float32)
        for ex in (None, ids[q]):
            a, b = cpu.search(h, "q", mask=mask, exclude_id=ex), gpu.search(h, "q", mask=mask, exclude_id=ex)
            assert [x[0] for x in a.precedents] == [x[0] for x in b.precedents], (q, ex)
            assert abs(a.mass - b.mass) < 1e-3 * max(1.0, a.mass)
            assert all(abs(a.p_mem[k] - b.p_mem.get(k, 0.0)) < 1e-3 for k in a.p_mem)
