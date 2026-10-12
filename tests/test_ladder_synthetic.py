import numpy as np
from dec.schema import DecisionItem, Option, Encoded
from dec.eval.ladder import Prepared, run_config, CONFIGS

def _synthetic(seed=0, n_pool=400, n_eval=200, d=32, K=3):
    rng = np.random.default_rng(seed); items, enc = [], {}
    opts = [Option(f"c{i}") for i in range(K)]
    for i in range(n_pool + n_eval):
        y = int(rng.integers(0, K)); h = rng.normal(size=d).astype(np.float32); h[:4] += y * 2.0
        p0 = rng.dirichlet(np.ones(K)).astype(np.float32); p0[y] += 0.2; p0 /= p0.sum()
        it = DecisionItem(id=f"i{i}", dataset="syn", state_id=f"s{i}", state=f"estado {i} con {'rojo' if y==0 else 'azul' if y==1 else 'verde'}",
                          qtype="choice", instructions="color?", options=opts, gold=f"c{y}", question_key="syn:q")
        items.append(it); enc[it.id] = Encoded(it.id, p0, np.log(p0), h, h, "letters", False)
    return Prepared(items_pool=items[:n_pool], items_eval=items[n_pool:], enc=enc, labeled=items[:n_pool],
                    constraints=[], qemb={"color?": np.ones(4, np.float32)}, dataset="syn", seed=seed, backbone_id="syn")

def test_all_cpu_configs_run_and_memory_beats_prior():
    prep = _synthetic(); out = {c: run_config(prep, c) for c in CONFIGS if c != "B2"}   # B2 necesita MiniLM: se prueba en integración
    for c, r in out.items():
        assert set(["acc", "ece", "brier", "n_eval", "n_labeled", "risk"]) <= set(r), c
    assert out["B3"]["acc"] > out["B0"]["acc"] + 0.1 and out["ours1"]["acc"] >= out["B3"]["acc"] - 0.02
    assert 0 <= out["ours1"]["risk"]["automation"] <= 1

def test_empty_labeled_never_crashes():
    prep = _synthetic(); prep.labeled = []
    for c in CONFIGS:
        if c == "B2": continue
        r = run_config(prep, c)
        assert r["n_labeled"] == 0 and 0 <= r["acc"] <= 1, c

def test_coherence_path_reports_keys():
    from dec.coherence import implies
    prep = _synthetic(n_pool=60, n_eval=40)
    ev = []
    for it in prep.items_eval:
        ev.append(it)
        q2 = DecisionItem(id=it.id + "b", dataset="syn", state_id=it.state_id, state=it.state, qtype="choice", instructions="color?",
                          options=it.options, gold=it.gold, question_key="syn:q2")
        ev.append(q2); prep.enc[q2.id] = prep.enc[it.id]
    prep.items_eval = ev; prep.constraints = [implies("syn:q", "c0", "syn:q2", {"c0"})]
    r = run_config(prep, "B3")
    assert {"incoherence_before", "incoherence_after", "acc_after", "map_fallbacks"} <= set(r["coherence"])
