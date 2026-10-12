"""mteb/banking77 (parquet). 77 intenciones bancarias."""
from __future__ import annotations
from datasets import load_dataset
from dec.schema import DecisionItem, Option
from . import register

@register("banking77")
def load_banking77(split: str) -> list[DecisionItem]:
    ds = load_dataset("mteb/banking77", split=split)
    if "label_text" in ds.column_names:
        names = sorted(set(ds["label_text"]))
        def label_of(r): return r["label_text"]
    else:
        names = list(ds.features["label"].names)
        def label_of(r): return names[r["label"]]
    opts = [Option(n, n.replace("_", " ")) for n in names]
    items = []
    for i, r in enumerate(ds):
        sid = f"banking77:{split}:{i}"
        items.append(DecisionItem(id=sid, dataset="banking77", state_id=sid, state=r["text"], qtype="choice",
            instructions="Which banking intent does the customer express?", options=opts, gold=label_of(r),
            question_key="banking77:intent", domain="banking", family="intent"))
    return items
