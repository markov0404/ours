"""clinc_oos config plus via rama parquet: 150 intenciones + oos."""
from __future__ import annotations
from datasets import load_dataset
from dec.schema import DecisionItem, Option
from . import register

_BASE = "hf://datasets/clinc/clinc_oos@refs/convert/parquet/plus"

def _intent_names(feat) -> list[str]:
    if hasattr(feat, "names"):
        return list(feat.names)
    if hasattr(feat, "int2str"):
        return [feat.int2str(i) for i in range(feat.num_classes)]
    raise RuntimeError("features['intent'] no expone names ni int2str; no se inventan nombres")

@register("clinc")
def load_clinc(split: str) -> list[DecisionItem]:
    ds = load_dataset("parquet", data_files={split: f"{_BASE}/{split}/0000.parquet"}, split=split)
    names = _intent_names(ds.features["intent"])
    opts = [Option(n, n.replace("_", " ")) for n in names]
    items = []
    for i, r in enumerate(ds):
        gold = names[r["intent"]]
        sid = f"clinc:{split}:{i}"
        items.append(DecisionItem(id=sid, dataset="clinc", state_id=sid, state=r["text"], qtype="choice",
            instructions="Which intent does the user express? Choose 'oos' if none applies.",
            options=opts, gold=gold, question_key="clinc:intent", domain="assistant", family="intent",
            meta={"is_oos": gold == "oos"}))
    return items
