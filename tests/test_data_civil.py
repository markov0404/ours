import pytest
from dec.data import load

ATTRS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]

@pytest.mark.timeout(1200)
def test_civil_seven_questions_and_implications():
    items = load("civil", "test", n=3000, seed=0)
    assert len(items) == 3000 * 7
    keys = {it.question_key for it in items}
    assert keys == {f"civil:{a}" for a in ATTRS}
    imps = [tuple(p) for p in items[0].meta["implications"]]  # JSON de la caché devuelve listas
    assert ("civil:severe_toxicity", "civil:toxicity") in imps
    # balance mínimo: al menos 5% de 'yes' en toxicity
    tox = [it for it in items if it.question_key == "civil:toxicity"]
    assert sum(it.gold == "yes" for it in tox) / len(tox) >= 0.05
