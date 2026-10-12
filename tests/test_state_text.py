"""Estados estructurados (dict/list): texto estable para numeric/router/memoria; jevk5 los embebe nativos."""
import json
import numpy as np
from dec.schema import DecisionItem, Option, state_text
from dec.router import gap_signals


def _item(state) -> DecisionItem:
    return DecisionItem("x", "t", "s", state, "choice", "Which team handles refunds?", [Option("billing"), Option("tech")], "billing", "t:q")


def test_state_text_identity_and_json():
    assert state_text("hola") == "hola"
    d = {"order": 4411, "note": "año"}
    t = state_text(d)
    assert json.loads(t) == d and "año" in t and t.startswith("{\n")


def test_gap_signals_accepts_dict_state():
    g = gap_signals(_item({"team": "refunds handled by billing"}), np.array([0.7, 0.3]), 0.0)
    assert isinstance(g, dict) or g is not None
