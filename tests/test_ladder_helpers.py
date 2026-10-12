import numpy as np
from test_ladder_synthetic import _synthetic
from dec.eval import ladder
from dec.eval.ladder import run_config


def test_return_probs_is_backwards_compatible_and_consistent():
    prep = _synthetic(n_pool=100, n_eval=50)
    plain = run_config(prep, "ours1")
    out, extra = run_config(prep, "ours1", return_probs=True)
    assert out["acc"] == plain["acc"]
    assert extra["P"].shape == (50, 3) and extra["G"].shape == (50,) and len(extra["items"]) == 50 and extra["mass"].shape == (50,)
    assert abs(float((extra["P"].argmax(1) == extra["G"]).mean()) - out["acc"]) < 1e-9


def test_prepare_pool_from_uses_external_train_caches(monkeypatch):
    syn = _synthetic(n_pool=30, n_eval=10)
    ext = [type(it)(**{**it.__dict__, "id": "ext" + it.id}) for it in syn.items_pool]
    monkeypatch.setattr(ladder, "pool_and_eval", lambda d, s, l: ([], syn.items_eval, []))
    monkeypatch.setattr(ladder, "load", lambda name, split, **kw: ext if (name, split) == ("extds", "train") else [])
    import dec.backbone.cache as cache
    class _P:
        def __init__(self, ok): self.ok = ok
        def exists(self): return self.ok
    monkeypatch.setattr(cache, "cache_path", lambda d, split, bb: _P(d == "extds" or (d == "syn" and split == "test")))
    allenc = {**syn.enc, **{"ext" + k: v for k, v in syn.enc.items()}}
    monkeypatch.setattr(cache, "read_cache", lambda p: allenc)
    monkeypatch.setattr(ladder, "embed_questions", lambda qs: np.ones((len(qs), 4), np.float32))
    prep = ladder.prepare("syn", "bb", 10 ** 9, 0, pool_from=["extds"])
    assert len(prep.items_pool) == 30 and all(it.id.startswith("ext") for it in prep.items_pool)
    assert len(prep.labeled) == 30 and prep.items_eval == syn.items_eval
