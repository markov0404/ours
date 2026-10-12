"""K heterogéneo por question_key (plumb/massive/jevbench): la escalera no debe romperse ni en B0/B1 ni con memoria."""
import numpy as np
from dec.schema import DecisionItem, Option, Encoded
from dec.eval.ladder import Prepared, run_config, fit_fusion_grouped, FusionBank
from dec.fusion import FusionRecord


def _mixed(seed=0, n_pool=300, n_eval=120, d=24):
    rng = np.random.default_rng(seed); items, enc = [], {}
    for i in range(n_pool + n_eval):
        K = 2 if i % 2 == 0 else 4; qk = f"syn:q{K}"; opts = [Option(f"c{j}") for j in range(K)]
        y = int(rng.integers(0, K)); h = rng.normal(size=d).astype(np.float32); h[:3] += y * 2.0
        p0 = rng.dirichlet(np.ones(K)).astype(np.float32); p0[y] += 0.3; p0 /= p0.sum()
        it = DecisionItem(id=f"m{i}", dataset="syn", state_id=f"s{i}", state=f"estado {i}", qtype="choice", instructions=f"q{K}?",
                          options=opts, gold=f"c{y}", question_key=qk)
        items.append(it); enc[it.id] = Encoded(it.id, p0, np.log(p0), h, h, "letters", False)
    return Prepared(items_pool=items[:n_pool], items_eval=items[n_pool:], enc=enc, labeled=items[:n_pool], constraints=[],
                    qemb={"q2?": np.ones(4, np.float32), "q4?": np.zeros(4, np.float32)}, dataset="syn", seed=seed, backbone_id="syn")


def test_mixed_k_runs_all_cpu_configs():
    prep = _mixed()
    for c in ("B0", "B1", "B3", "ours1"):
        out, ex = run_config(prep, c, return_probs=True)
        assert ex["P"].shape == (len(prep.items_eval), 4) and 0 <= out["acc"] <= 1, c
        assert set(out["fusion"]) == {"default", "by_k"}
    assert out["fusion"]["by_k"].keys() == {"2", "4"}


def test_fit_fusion_grouped_falls_back_to_largest_group():
    rng = np.random.default_rng(0); recs = []
    for i in range(60):
        K = 3 if i < 55 else 5; p = rng.dirichlet(np.ones(K)); recs.append(FusionRecord(p, np.zeros(K), 0.0, int(p.argmax())))
    bank = fit_fusion_grouped(recs, min_n=20)
    assert isinstance(bank, FusionBank) and set(bank.by_k) == {3} and bank.for_k(5) is bank.default
