"""Rendimiento de Backbone.encode: batch 1 vs lotes por presupuesto de tokens."""
import argparse, time
import torch
from dec.schema import DecisionItem, Option
from dec.backbone.model import Backbone

def make_items(n: int, n_opts_suffix: int = 0) -> list[DecisionItem]:
    out = []
    for i in range(n):
        reps = 5 + (i * 37) % 120                       # longitudes variadas (~100-2000 tokens)
        st = f"Customer message {i}: I was billed twice for order #{i}. " + "Please check my account history carefully. " * reps
        k = n_opts_suffix if (n_opts_suffix and i % 4 == 0) else 5
        out.append(DecisionItem(id=f"b{i}", dataset="bench", state_id=f"b{i}", state=st, qtype="choice", instructions="Which team?",
                                options=[Option(f"opt_{j}", f"description of option {j}") for j in range(k)], gold="opt_0", question_key="q"))
    return out

def run(bb, items, budget):
    torch.cuda.synchronize(); t = time.time()
    enc = bb.encode(items, log_every=10**9, token_budget=budget)
    torch.cuda.synchronize(); dt = time.time() - t
    ntok = sum(len(bb._prepare(it)[1]) for it in items)
    return dt, len(items) / dt, ntok / dt

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B"); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--dtype", default="bf16"); ap.add_argument("--budget", type=int, default=16384)
    ap.add_argument("--suffix-options", type=int, default=0, help="si >0, 1 de cada 4 ítems tiene tantas opciones (lectura por sufijos)")
    a = ap.parse_args()
    bb = Backbone(a.model, dtype=a.dtype)
    items = make_items(a.n, a.suffix_options)
    run(bb, items[:8], a.budget)                         # warm-up
    for name, b in (("batch1", 1), (f"batched(budget={a.budget})", a.budget)):
        dt, ips, tps = run(bb, items, b)
        print(f"BENCH {a.model} {name}: {dt:.1f}s  {ips:.1f} items/s  {tps:.0f} tok/s")
