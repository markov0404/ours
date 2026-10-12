import numpy as np, pytest
from dec.memory import PrecedentMemory, Standardizer

def _data(seed=0, n=300, d=32):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 3, n); H = rng.normal(size=(n, d)).astype(np.float32)
    H[:, :4] += y[:, None] * 3.0          # las 4 primeras dims separan clases
    return H, [f"c{v}" for v in y]

def test_search_returns_probs_mass_and_precedents():
    H, y = _data(); mem = PrecedentMemory(d=32)
    mem.add_many(H, y, ["q"] * len(y), [f"i{i}" for i in range(len(y))]); mem.build()
    hit = mem.search(H[0], "q", exclude_id="i0")
    assert abs(sum(hit.p_mem.values()) - 1) < 1e-6 and hit.mass > 0 and len(hit.precedents) == 32
    assert max(hit.p_mem, key=hit.p_mem.get) == y[0]
    assert all(pid != "i0" for pid, _, _ in hit.precedents)

def test_mask_restricts_distance_to_subspace():
    H, y = _data(); mem = PrecedentMemory(d=32)
    mem.add_many(H, y, ["q"] * len(y), [f"i{i}" for i in range(len(y))]); mem.build()
    good = np.zeros(32, np.float32); good[:4] = 1.0
    bad = np.zeros(32, np.float32); bad[-4:] = 1.0
    def acc(m):
        hits = [mem.search(H[i], "q", mask=m, exclude_id=f"i{i}").p_mem for i in range(60)]
        return np.mean([max(p, key=p.get) == y[i] for i, p in enumerate(hits)])
    assert acc(good) > acc(bad) + 0.2

def test_weights_monotonic_in_distance_and_empty_partition():
    H, y = _data(); mem = PrecedentMemory(d=32)
    mem.add_many(H, y, ["q"] * len(y), [f"i{i}" for i in range(len(y))]); mem.build()
    hit = mem.search(H[0], "q", exclude_id="i0")
    w = [w for _, _, w in hit.precedents]
    assert all(w[i] >= w[i + 1] - 1e-9 for i in range(len(w) - 1))
    empty = mem.search(H[0], "otra")
    assert empty.mass == 0.0 and empty.p_mem == {} and empty.precedents == []

def test_standardizer_zero_mean_unit_var():
    H, _ = _data(); Z = Standardizer().fit(H).transform(H)
    assert np.allclose(Z.mean(0), 0, atol=1e-5) and np.allclose(Z.std(0), 1, atol=1e-4)
