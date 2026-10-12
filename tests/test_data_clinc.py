import pytest
from dec.data import load

@pytest.mark.timeout(900)
def test_clinc_plus_has_oos():
    items = load("clinc", "test")
    assert len(items) == 5500
    assert len(items[0].options) == 151 and "oos" in items[0].option_keys()
    assert sum(it.meta["is_oos"] for it in items) == 1000
