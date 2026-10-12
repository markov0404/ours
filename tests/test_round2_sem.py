import numpy as np
import pytest
from dec.schema import DecisionItem, Option
from dec.router import gap_signals, gap_signals_batch, route, RouterThresholds, fit_thresholds, GapSignals
from scripts.gates.rules import g8_rule_strict

Q = "When will the credit be applied to the account?"
P3 = np.array([0.34, 0.33, 0.33])


def _item(state, instr=Q):
    return DecisionItem(id="i", dataset="t", state_id="s", state=state, qtype="choice", instructions=instr,
                        options=[Option("a"), Option("b"), Option("c")], gold="a", question_key="q")


def _fake_embed(texts):
    """Embedder falso: bolsa de letras normalizada (textos idénticos -> coseno 1; vocabularios distintos -> bajo)."""
    out = np.zeros((len(texts), 26), np.float32)
    for i, t in enumerate(texts):
        for ch in t.lower():
            if "a" <= ch <= "z": out[i, ord(ch) - 97] += 1
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


@pytest.fixture
def fake(monkeypatch):
    from dec.eval import minilm
    monkeypatch.setattr(minilm, "embed_sentences", _fake_embed)


def test_sem_gap_zero_when_question_in_state(fake):
    s = gap_signals(_item(f"Hello there. {Q} Bye."), P3, 0.0, sem=True)
    assert s.sem_gap == pytest.approx(0.0, abs=1e-5)


def test_sem_gap_large_for_unrelated_state(fake):
    s = gap_signals(_item("Zzz xyz. Quux jqx zqx."), P3, 0.0, sem=True)
    assert s.sem_gap > 0.3


def test_sem_off_by_default():
    assert gap_signals(_item("anything"), P3, 0.0).sem_gap == 0.0


def test_batch_matches_single(fake):
    items = [_item(f"{Q} Other."), _item("Zzz xyz. Quux jqx.")]
    b = gap_signals_batch(items, np.tile(P3, (2, 1)), [0.0, 0.0])
    for it, s in zip(items, b):
        assert s.sem_gap == pytest.approx(gap_signals(it, P3, 0.0, sem=True).sem_gap, abs=1e-5)


def test_route_red_on_sem_gap():
    s = GapSignals(0.1, 0.0, 0.9, 3, [], sem_gap=0.7)
    assert route(s, None, RouterThresholds(sem=0.5), False).huecos == "rojo"
    assert route(s, None, RouterThresholds(sem=0.8), False).huecos == "verde"
    assert route(s, None, RouterThresholds(), False).huecos == "verde"   # desactivado por defecto


def test_fit_thresholds_use_sem_separable():
    ans = [GapSignals(0.5, 0.0, 0.9, 3, [], sem_gap=0.1 + 0.01 * i) for i in range(10)]
    unans = [GapSignals(0.5, 0.0, 0.9, 3, [], sem_gap=0.6 + 0.01 * i) for i in range(10)]
    thr = fit_thresholds(ans, unans, use_sem=True)
    assert thr.sem <= 1.0 and 0.2 <= thr.sem < 0.7
    assert fit_thresholds(ans, unans, use_sem=False).sem == 1.1


def test_g8_rule_strict():
    f = {"acc_answerable": 0.8, "abstain_unanswerable": 0.0}
    assert g8_rule_strict(f, {"acc_answerable": 0.79, "abstain_unanswerable": 0.5, "tpr": 0.5, "fpr": 0.1})["zone"] == "A"
    assert g8_rule_strict(f, {"acc_answerable": 0.77, "abstain_unanswerable": 0.9})["zone"] == "C"
    assert g8_rule_strict(f, {"acc_answerable": 0.8, "abstain_unanswerable": 0.49})["zone"] == "C"
    d = g8_rule_strict(f, {"acc_answerable": 0.8, "abstain_unanswerable": 0.6, "tpr": 0.6, "fpr": 0.2})["detail"]
    assert d["tpr"] == 0.6 and d["fpr"] == 0.2


def test_embed_sentences_cache_roundtrip(tmp_path, monkeypatch):
    from dec.eval import minilm
    calls = []
    class M:
        def encode(self, texts, **kw): calls.append(list(texts)); return _fake_embed(texts)
    monkeypatch.setattr(minilm, "_MODEL", M())
    a = minilm.embed_sentences(["one two", "three"], cache=tmp_path)
    b = minilm.embed_sentences(["three", "four", "one two"], cache=tmp_path)
    assert calls == [["one two", "three"], ["four"]] and np.allclose(b[0], a[1]) and np.allclose(b[2], a[0])


@pytest.mark.slow
def test_real_minilm_integration():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available(): pytest.skip("sin GPU")
    s = gap_signals(_item(f"Intro. {Q} Bye."), P3, 0.0, sem=True)
    t = gap_signals(_item("The weather is nice and sunny today."), P3, 0.0, sem=True)
    assert s.sem_gap < 0.05 and t.sem_gap > 0.3
