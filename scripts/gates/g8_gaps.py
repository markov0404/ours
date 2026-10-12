"""G8: huecos. Plumb (trap/ambiguous/adversarial) con variantes `:noev` (evidencia ablacionada); modo escalation, c_escalate = 1.

Umbrales del router ajustados con señales de los ítems del POOL no etiquetados (y sus variantes `:noev`); evaluación sobre
los ítems de evaluación respondibles y sus variantes `:noev`. Las codificaciones `:noev` deben estar en el cache del backbone
(`plumb train --g8` y `plumb test --g8`: pool, held y test); si faltan, el script falla indicando cuántas.
"""
from __future__ import annotations
import argparse, sys
from dataclasses import replace
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.gates.rules import g8_rule, g8_rule_strict
from scripts.gates._util import finish, default_backbone

FAMILIES = ("trap", "ambiguous", "adversarial")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--results-dir", default="results")
    ap.add_argument("--backbone", default=None); ap.add_argument("--n", type=int, default=1000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sem", action="store_true", help="señal sem_gap (MiniLM) y ajuste con use_sem=True")
    ap.add_argument("--gate", default="G8", choices=["G8", "R2-D"], help="R2-D: regla estricta (hip-14) en el veredicto")
    a = ap.parse_args(argv)
    from dec.eval.ladder import prepare, run_config  # import perezoso: --help no necesita caches
    from dec.eval.unanswerable import build_g8_items
    from dec.router import gap_signals, gap_signals_batch, route, fit_thresholds
    from dec.policy import decide, RiskState
    from dec.schema import CostMatrix
    from dec.memory import MemoryHit
    bb = a.backbone or default_backbone(a.results_dir)
    prep = prepare("plumb", bb, a.n, a.seed)
    lab = {it.id for it in prep.labeled}
    cal_ans = [it for it in prep.items_pool if it.family in FAMILIES and it.id not in lab]   # pool no etiquetado: sin fuga con la memoria
    cal_noev = [x for x in build_g8_items(prep.items_pool, FAMILIES) if x.meta["source_id"] not in lab]
    ev_ans = [it for it in prep.items_eval if it.family in FAMILIES and it.meta.get("unanswerable") is not True]
    ev_noev = build_g8_items(ev_ans, FAMILIES)
    missing = [it.id for it in cal_noev + ev_noev if it.id not in prep.enc]
    if missing:
        raise FileNotFoundError(f"{len(missing)} variantes :noev sin cache para {bb} (ej. {missing[0]}); correr el cache de plumb con --g8")
    allit = cal_ans + cal_noev + ev_ans + ev_noev
    out, ex = run_config(replace(prep, items_eval=allit), "ours1", return_probs=True)
    P, mass = ex["P"], ex["mass"]; idx = {it.id: i for i, it in enumerate(allit)}
    if a.sem:
        sg = gap_signals_batch(allit, P, mass); sig = {it.id: s for it, s in zip(allit, sg)}   # P viene rellenada a Kmax
    else:
        sig = {it.id: gap_signals(it, P[idx[it.id], :len(it.options)], mass[idx[it.id]]) for it in allit}
    thr = replace(fit_thresholds([sig[i.id] for i in cal_ans], [sig[i.id] for i in cal_noev], use_sem=a.sem), mass=0.0)   # eje oráculo apagado: G8 mide huecos
    costs = CostMatrix(c_escalate=1.0)   # con 0/1 el Bayes nunca escala (1 - pmax < 1): ESCALATE solo sale del router
    risk = RiskState(theta=0.0, alpha=0.05, bound=float("nan"))

    def act(it) -> str:
        i = idx[it.id]; m = float(mass[i]); cold = m == 0
        v = route(sig[it.id], costs, thr, False)
        # G8 mide SOLO el eje de huecos: el oráculo (masa de precedentes) se fuerza a verde; con thr.mass=0 `route` lo pondría
        # siempre en rojo ("sin memoria") y escalaría el 100 % (intento 1 de G8, vacuo).
        v = replace(v, oraculo="verde", any_red=(v.huecos == "rojo"))
        hit = MemoryHit({}, m, [])
        return decide(it, prep.enc[it.id].p0, hit, P[i], v, risk, costs, "escalation", reason="sin_precedentes" if cold else "prior").action

    acts_a = {it.id: act(it) for it in ev_ans}; acts_n = {it.id: act(it) for it in ev_noev}
    kept = [it for it in ev_ans if acts_a[it.id] != "ESCALATE"]
    # acc_answerable sobre TODOS los respondibles: escalar uno respondible cuenta como error (si no, el gate se infla escalando)
    ours = {"acc_answerable": float(np.mean([acts_a[it.id] == it.gold for it in ev_ans])) if ev_ans else 0.0,
            "abstain_unanswerable": float(np.mean([acts_n[it.id] == "ESCALATE" for it in ev_noev])) if ev_noev else 0.0}
    forced = {"acc_answerable": float(np.mean([int(np.argmax(prep.enc[it.id].p0)) == it.gold_index() for it in ev_ans])), "abstain_unanswerable": 0.0}
    strict = [it for it in ev_noev if it.meta.get("strict")]
    ours_s = {**ours, "abstain_unanswerable": float(np.mean([acts_n[it.id] == "ESCALATE" for it in strict])) if strict else 0.0}
    forced_s = {**forced, "abstain_unanswerable": 0.0}
    v = g8_rule(forced, ours)
    vs = g8_rule(forced_s, ours_s)
    if a.gate == "R2-D":
        tpr = ours["abstain_unanswerable"]; fpr = float(np.mean([acts_a[it.id] == "ESCALATE" for it in ev_ans])) if ev_ans else 0.0
        vr = g8_rule_strict(forced, {**ours, "tpr": tpr, "fpr": fpr})
        vr["detail"]["rule_round1"] = v
        axes = {"entropy": sum(sig[i.id].entropy_norm > thr.entropy for i in ev_ans + ev_noev),
                "overlap": sum(sig[i.id].overlap < thr.overlap for i in ev_ans + ev_noev),
                "sem": sum(sig[i.id].sem_gap > thr.sem for i in ev_ans + ev_noev),
                "huecos_rojo_ans": sum(route(sig[i.id], costs, thr, False).huecos == "rojo" for i in ev_ans),
                "huecos_rojo_noev": sum(route(sig[i.id], costs, thr, False).huecos == "rojo" for i in ev_noev)}
        return finish("R2-D", vr, {"forced": forced, "ours": ours, "tpr": tpr, "fpr": fpr, "sem": a.sem, "coverage_answerable": len(kept) / max(len(ev_ans), 1),
                                   "n": {"answerable": len(ev_ans), "noev": len(ev_noev), "cal_answerable": len(cal_ans), "cal_noev": len(cal_noev)},
                                   "thresholds": thr.__dict__, "axis_counts": axes},
                      {"backbone_id": bb}, numbers=f"TPR {tpr:.3f} FPR {fpr:.3f} acc {ours['acc_answerable']:.3f} vs forzado {forced['acc_answerable']:.3f}")
    return finish("G8", v, {"forced": forced, "ours": ours, "coverage_answerable": len(kept) / max(len(ev_ans), 1),
                            "n": {"answerable": len(ev_ans), "noev": len(ev_noev), "strict": len(strict), "cal_answerable": len(cal_ans), "cal_noev": len(cal_noev)},
                            "thresholds": thr.__dict__},
                  {"backbone_id": bb, "strict": {"zone": vs["zone"], "detail": vs["detail"], "ours": ours_s, "forced": forced_s}},
                  numbers=f"forzado {v['detail']['forced']:.3f} vs ours {v['detail']['ours']:.3f}; strict {vs['detail']['forced']:.3f} vs {vs['detail']['ours']:.3f}")


if __name__ == "__main__":
    raise SystemExit(main())
