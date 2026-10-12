import numpy as np, pytest, torch
from dec.schema import DecisionItem, Option
from dec.backbone.model import Backbone
from dec.backbone.cache import write_cache, read_cache
from dec.backbone.readout import suffix_logprobs, packed_suffix_logprobs

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU")
MID = "Qwen/Qwen3-0.6B"

@pytest.fixture(scope="module")
def bb32():
    return Backbone(MID, dtype="fp32", style="numeric")

@pytest.fixture(scope="module")
def bb():
    return Backbone(MID, dtype="bf16", style="numeric")

def _items():
    base = dict(dataset="t", qtype="choice", instructions="Which team should handle this?", question_key="q", gold="billing",
                options=[Option("billing", "Payments and refunds"), Option("tech", "Bugs"), Option("sales", "New purchases")])
    return [DecisionItem(id="a", state_id="a", state="I was billed twice for order #4411. Please refund the duplicate.", **base),
            DecisionItem(id="b", state_id="b", state="The app crashes when I open settings.", **base)]

def _many(i, n=40, extra=0):
    st = f"transfer money to my savings account number {i}. " + "Please hurry, this is quite urgent. " * extra
    return DecisionItem(id=f"m{i}", dataset="t", state_id=f"m{i}", state=st, qtype="choice",
                        instructions="Which intent?", options=[Option(f"intent_{k}") for k in range(n)], gold="intent_0", question_key="q")

@pytest.mark.timeout(900)
def test_encode_shapes_and_h_independent_of_question(bb32, tmp_path):
    # fp32: en bf16 el ruido numérico entre formas de lote (~7e-3 en h) supera atol 1e-3
    bb = bb32
    items = _items()
    enc = bb.encode(items)
    assert len(enc) == 2 and enc[0].p0.shape == (3,) and abs(enc[0].p0.sum() - 1) < 1e-4
    assert enc[0].h.shape[0] == bb.hidden_size and enc[0].h.dtype == np.float32
    other_q = DecisionItem(**{**items[0].__dict__, "id": "a2", "instructions": "Is this urgent?"})
    enc2 = bb.encode([other_q])[0]
    assert np.allclose(enc2.h, enc[0].h, atol=1e-3)
    assert not np.allclose(enc[1].h, enc[0].h, atol=1e-3)
    p = tmp_path / "c.parquet"
    write_cache(p, enc, bb.backbone_id)
    back = read_cache(p)
    assert set(back) == {"a", "b"} and np.allclose(back["a"].h, enc[0].h, atol=1e-2)

@pytest.mark.timeout(900)
def test_codes_readout_for_many_options(bb):
    e = bb.encode([_many(0)])[0]            # 40 opciones: códigos de dos letras en una pasada
    assert e.readout == "codes" and e.p0.shape == (40,) and abs(e.p0.sum() - 1) < 1e-4

def test_suffix_readout_on_request(bb):
    old = bb.readout; bb.readout = "suffix"
    try:
        e = bb.encode([_many(0)])[0]
    finally:
        bb.readout = old
    assert e.readout == "suffix" and e.p0.shape == (40,) and abs(e.p0.sum() - 1) < 1e-4

def test_int8_not_implemented():
    with pytest.raises(NotImplementedError):
        Backbone(MID, dtype="int8")

def test_hook_matches_output_hidden_states(bb32):
    pp, ids, _ = bb32._prepare(_items()[0])
    x = torch.tensor([ids], device="cuda")
    with torch.inference_mode():
        out = bb32.model(input_ids=x, output_hidden_states=True, use_cache=False)
    assert torch.allclose(bb32._hidden[0].float(), out.hidden_states[-1][0].float(), atol=1e-4)

@pytest.mark.timeout(900)
def test_packed_suffix_equals_chunked(bb32):
    assert bb32.can_pack
    old = bb32.readout; bb32.readout = "suffix"   # forzar sufijos: por defecto 40 opciones leen por códigos
    try:
        _check_packed(bb32)
    finally:
        bb32.readout = old


def _check_packed(bb32):
    for i in range(5):
        pp, ids, _ = bb32._prepare(_many(i, extra=i))
        x = torch.tensor([ids], device="cuda")
        suf = [bb32.tok(s, add_special_tokens=False).input_ids for s in pp.suffixes]
        ref = suffix_logprobs(bb32.model, bb32.tok, x, suf)
        got = packed_suffix_logprobs(bb32.model, x, suf, use_ltk=bb32.use_ltk)
        assert np.allclose(ref, got, atol=1e-2), np.abs(ref - got).max()

@pytest.mark.timeout(900)
def test_batched_encode_equals_batch1(bb32):
    items = []
    for i in range(8):
        base = _items()[i % 2]
        items.append(DecisionItem(**{**base.__dict__, "id": f"x{i}", "state": base.state + " Additional context." * (i * 5)}))
    items.append(_many(0)); items.append(_many(1, extra=3))   # mezcla letras y sufijos
    one = [bb32.encode([it], token_budget=1)[0] for it in items]
    bat = bb32.encode(items, token_budget=4096)
    assert any(len(b) > 1 for b in [bat])
    for a, b in zip(one, bat):
        assert a.item_id == b.item_id and a.readout == b.readout
        assert np.allclose(a.p0, b.p0, atol=1e-3)
        assert np.linalg.norm(a.h - b.h) / np.linalg.norm(a.h) < 1e-2

@pytest.mark.timeout(900)
def test_batched_encode_bf16_close_to_batch1(bb):
    items = [DecisionItem(**{**_items()[i % 2].__dict__, "id": f"y{i}", "state": _items()[i % 2].state + " More." * (i * 4)}) for i in range(6)]
    one = [bb.encode([it], token_budget=1)[0] for it in items]
    bat = bb.encode(items, token_budget=4096)
    for a, b in zip(one, bat):
        assert np.allclose(a.p0, b.p0, atol=0.15)  # bf16: logits ~25 se cuantizan a pasos de 0.125-0.25
        assert np.linalg.norm(a.h - b.h) / np.linalg.norm(a.h) < 5e-2
