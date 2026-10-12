"""AmazonScience/massive via rama parquet. Dos preguntas por utterance: scenario e intent."""
from __future__ import annotations
from datasets import load_dataset
from dec.schema import DecisionItem, Option
from . import register

_BASE = "hf://datasets/AmazonScience/massive@refs/convert/parquet"

def _names(ds, col: str) -> list[str]:
    feat = ds.features[col]
    if hasattr(feat, "names"):
        return list(feat.names)
    raise RuntimeError(f"la columna {col} no trae nombres de clase; revisar la rama parquet")

@register("massive")
def load_massive(split: str, locale: str = "en-US") -> list[DecisionItem]:
    ds = load_dataset("parquet", data_files={split: f"{_BASE}/{locale}/{split}/0000.parquet"}, split=split)
    scen_names, int_names = _names(ds, "scenario"), _names(ds, "intent")
    # intent name = "<scenario>_<verb>" en MASSIVE: el prefijo da el escenario
    scen_of_intent = {n: n.split("_")[0] for n in int_names}
    if not set(scen_of_intent.values()) <= set(scen_names):
        raise RuntimeError("prefijo de intent no coincide con scenario")
    scen_opts = [Option(n, n.replace("_", " ")) for n in scen_names]
    int_opts = [Option(n, n.replace("_", " ")) for n in int_names]
    items = []
    for r in ds:
        sid = f"massive:{locale}:{split}:{r['id']}"
        common = dict(dataset="massive", state_id=sid, state=r["utt"], domain=locale, family="intent")
        items.append(DecisionItem(id=sid + ":scenario", qtype="choice",
            instructions="Which scenario does this user request belong to?",
            options=scen_opts, gold=scen_names[r["scenario"]], question_key="massive:scenario", **common))
        items.append(DecisionItem(id=sid + ":intent", qtype="choice",
            instructions="What is the user's intent?",
            options=int_opts, gold=int_names[r["intent"]], question_key="massive:intent",
            meta={"scenario_of_intent": scen_of_intent}, **common))
    return items
