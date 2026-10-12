import subprocess, pytest
from dec.data import load

@pytest.mark.timeout(900)
def test_banking77():
    # mteb/banking77 test tiene 3076 filas (el original PolyAI tiene 3080)
    items = load("banking77", "test")
    assert len(items) == 3076 and len(items[0].options) == 77

@pytest.mark.timeout(600)
def test_jevbench_public_231():
    subprocess.run(["bash", "scripts/fetch_jevbench.sh"], check=True)
    items = load("jevbench", "public")
    assert len(items) == 231
    tiers = {it.meta["tier"] for it in items}
    assert tiers == {"original", "easy", "hard"}
    assert all(it.gold in it.option_keys() for it in items)


def _choice(crit, labels=("a", "b", "c")):
    return {"id": "x", "group": "g", "family": "f", "split": "public", "state": "s", "labels": list(labels),
            "expected": "b", "question": {"type": "choice", "instructions": "i", "criteria": crit}}


def test_choice_criteria_list_same_length():
    from dec.data.jevbench import _to_item
    it = _to_item(_choice(["da", "db", "dc"]), "easy")
    assert [(o.key, o.desc) for o in it.options] == [("a", "da"), ("b", "db"), ("c", "dc")]
    assert it.gold in it.option_keys()


def test_choice_criteria_list_length_mismatch():
    from dec.data.jevbench import _to_item
    it = _to_item(_choice(["da"]), "easy")
    assert [(o.key, o.desc) for o in it.options] == [("a", "a"), ("b", "b"), ("c", "c")]


def test_choice_options_follow_criteria_order():
    from dec.data.jevbench import _to_item
    r = {"id": "x", "labels": ["b", "a", "c"], "expected": "a", "state": "s",
         "question": {"type": "choice", "instructions": "?", "criteria": {"a": "A", "b": "B"}}}
    it = _to_item(r, "hard")
    assert [o.key for o in it.options] == ["a", "b", "c"] and it.options[0].desc == "A" and it.gold == "a"
