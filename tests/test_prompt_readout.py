# tests/test_prompt_readout.py
import json, numpy as np, torch, pytest
from transformers import AutoTokenizer
from dec.schema import DecisionItem, Option
from dec.backbone.prompt import build_prompt, auto_readout, LETTERS
from dec.backbone.readout import letters_probs, letter_token_ids
from dec.backbone import jevk5_prompt

@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B")

def _item(n_opts, state="short state"):
    return DecisionItem(id="i", dataset="t", state_id="s", state=state, qtype="choice", instructions="Pick one.",
                        options=[Option(f"opt{i}", f"desc {i}") for i in range(n_opts)], gold="opt0", question_key="q")

def test_auto_readout_switches_to_suffix_above_20():
    assert auto_readout(2) == "letters" and auto_readout(20) == "letters" and auto_readout(21) == "suffix"
    assert auto_readout(21, 500) == "codes" and auto_readout(501, 500) == "suffix"

def test_numeric_prompt_contains_all_options_and_state_span(tok):
    it = _item(77)
    pp = build_prompt(it, "numeric", tok)
    assert pp.readout == "codes" and pp.suffixes is None and len(pp.option_codes) == 77   # banking77: una pasada
    a, b = pp.state_char_span
    assert pp.prefix_text[a:b] == it.state
    assert f"[{pp.option_codes[-1]}] opt76: desc 76" in pp.prefix_text and not pp.truncated

def test_truncation_keeps_head_tail_and_all_options(tok):
    it = _item(5, state=" ".join(f"w{i}" for i in range(20000)))
    pp = build_prompt(it, "numeric", tok, max_tokens=2048)
    assert pp.truncated and "[...]" in pp.prefix_text
    assert all(f"[{c}]" in pp.prefix_text for c in "ABCDE")
    assert len(tok(pp.prefix_text).input_ids) <= 2048

def test_jevk5_prompt_matches_fixture():
    with open("tests/fixtures/jevk5_prompt_example.json") as f:
        fx = json.load(f)
    q = fx["question"]
    it = DecisionItem(id="i", dataset="t", state_id="s", state=fx["state"], qtype="choice", instructions=q["instructions"],
                      options=[Option(k, v) for k, v in q["criteria"].items()], gold="billing", question_key="q")
    assert jevk5_prompt.render(it) == fx["prompt"]

def test_letters_probs_softmax_with_temperature():
    logits = torch.zeros(100); logits[5] = 2.0; logits[7] = 0.0
    p = letters_probs(logits, [5, 7], T=1.0)
    assert abs(p.sum() - 1) < 1e-6 and p[0] > p[1]
    p2 = letters_probs(logits, [5, 7], T=100.0)
    assert abs(p2[0] - 0.5) < 0.01

def test_20_options_use_letters(tok):
    pp = build_prompt(_item(20), "numeric", tok)
    assert pp.readout == "letters" and pp.option_codes == list("ABCDEFGHIJKLMNOPQRST")
    assert pp.suffixes is None and "[T]" in pp.prefix_text

def test_21_options_use_codes_by_default_and_suffix_digits_on_request(tok):
    pp = build_prompt(_item(21), "numeric", tok)
    assert pp.readout == "codes" and pp.suffixes is None and len(pp.option_codes) == 21
    pp = build_prompt(_item(21), "numeric", tok, readout="suffix")
    assert pp.readout == "suffix" and pp.option_codes == [str(i) for i in range(1, 22)]
    assert pp.suffixes[0] == "1]"

def test_explicit_letters_over_20_raises(tok):
    with pytest.raises(ValueError):
        build_prompt(_item(25), "numeric", tok, readout="letters")

def test_letter_token_ids_distinct(tok):
    ids = letter_token_ids(tok, list(LETTERS))
    assert len(ids) == 20 and len(set(ids)) == 20

def test_jevk5_span_in_escaped_text(tok):
    st = 'He said "hi"\nnewline'
    it = _item(3, state=st)
    pp = build_prompt(it, "jevk5", tok)
    a, b = pp.state_char_span
    assert pp.prefix_text[a:b] == json.dumps(st, ensure_ascii=False)[1:-1] and b - a != len(st)

def test_jevk5_over_16_options_falls_back_to_numeric(tok):
    pp = build_prompt(_item(17), "jevk5", tok)          # 17 <= 20: letras
    assert pp.style == "numeric" and pp.readout == "letters" and len(pp.option_codes) == 17
    pp = build_prompt(_item(40), "jevk5", tok)          # > 20: códigos de dos letras, un token cada uno, sin sufijos
    assert pp.style == "numeric" and pp.readout == "codes" and pp.suffixes is None and len(pp.option_codes) == 40
    assert all(len(tok(c, add_special_tokens=False).input_ids) == 1 for c in pp.option_codes) and len(set(pp.option_codes)) == 40


def test_code_alphabet_covers_clinc_and_suffix_beyond(tok):
    from dec.backbone.prompt import code_alphabet
    alpha = code_alphabet(tok)
    assert len(alpha) >= 151, len(alpha)                 # clinc: 151 intents en una pasada
    pp = build_prompt(_item(len(alpha) + 1), "numeric", tok)
    assert pp.readout == "suffix" and pp.suffixes is not None
