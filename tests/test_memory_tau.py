"""τ fija por memoria: la masa debe separar consultas con precedentes cercanos de consultas lejanas (OOS)."""
import numpy as np
from dec.memory import PrecedentMemory


def test_calibrated_tau_makes_mass_informative():
    rng = np.random.default_rng(0); d, n = 16, 400
    H = rng.normal(size=(n, d)).astype(np.float32); labels = [f"c{int(x)}" for x in rng.integers(0, 3, n)]; ids = [f"i{i}" for i in range(n)]
    mem = PrecedentMemory(d=d, k=16); mem.add_many(H, labels, ["q"] * n, ids); mem.build()
    near, far = H[0] + 0.05 * rng.normal(size=d).astype(np.float32), (H[0] + 40.0).astype(np.float32)
    m_near0, m_far0 = mem.search(near, "q").mass, mem.search(far, "q").mass
    tau = mem.calibrate_tau(H, ["q"] * n, ids)
    assert tau is not None and tau > 0
    m_near, m_far = mem.search(near, "q").mass, mem.search(far, "q").mass
    assert m_near > 2 * m_far, (m_near, m_far)
    assert abs(m_near0 - m_far0) < 0.3 * max(m_near0, 1e-9)   # sin τ fija la masa era ~constante (documenta el defecto)
