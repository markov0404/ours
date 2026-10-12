"""Reglas binding de la spec §7 como funciones puras sobre los JSON de resultados."""
from __future__ import annotations


def _acc(c: dict, k: str) -> float:
    """Media de accuracy de la clave `config@n` en el agregado de una curva."""
    return c["agg"][k]["acc"][0]


def _ece(c: dict, k: str) -> float:
    """Media de ECE de la clave `config@n` en el agregado de una curva."""
    return c["agg"][k]["ece"][0]


def g1_rule(curves: dict[str, dict], n: int = 1000) -> dict:
    """G1: Ours-1 supera a B1 en >= 3 pp de accuracy con ECE menor en >= 2 dominios."""
    wins = {d: (_acc(c, f"ours1@{n}") - _acc(c, f"B1@{n}") >= 0.03) and (_ece(c, f"ours1@{n}") < _ece(c, f"B1@{n}")) for d, c in curves.items()}
    return {"zone": "A" if sum(wins.values()) >= 2 else "C", "detail": wins}


def g2_rule(curves: dict[str, dict], n: int = 1000, multi: tuple[str, ...] = ("plumb", "civil")) -> dict:
    """G2: Ours-2 gana >= 1.5 pp sobre Ours-1 en dominios multipregunta y no retrocede > 1 pp en el resto."""
    gain = {d: _acc(curves[d], f"ours2@{n}") - _acc(curves[d], f"ours1@{n}") for d in curves}
    ok_multi = all(gain[d] >= 0.015 for d in multi if d in gain) and all(d in gain for d in multi)
    no_reg = all(gain[d] >= -0.01 for d in gain if d not in multi)
    return {"zone": "A" if ok_multi and no_reg else "C", "detail": gain}


def g3_rule(results: dict[str, dict]) -> dict:
    """G3: la decodificación conjunta reduce la incoherencia >= 50% relativo sin perder accuracy, en todos los dominios (>= 2)."""
    ok = {}
    for d, r in results.items():
        c = r.get("coherence") or {}
        if not c:   # sin restricciones minadas (p. ej. civil sin implicaciones con soporte): no evaluable → C
            ok[d] = False; continue
        rel = 1 - c["incoherence_after"] / max(c["incoherence_before"], 1e-9)
        ok[d] = rel >= 0.5 and c["acc_after"] >= r["acc"] - 1e-9
    return {"zone": "A" if all(ok.values()) and len(ok) >= 2 else "C", "detail": ok}


def g4_rule(by_alpha: dict[float, dict], oos: dict) -> dict:
    """G4: riesgo realizado <= alpha con automatización >= 50% para alpha en {0.02, 0.05}, y la masa mejora el AUROC OOS."""
    ok = all(by_alpha[a]["risk"]["realized_risk"] <= a and by_alpha[a]["risk"]["automation"] >= 0.5 for a in (0.02, 0.05))
    ok_oos = oos["auroc_conf_mass"] > oos["auroc_conf"]
    return {"zone": "A" if ok and ok_oos else "C", "detail": {"risk_ok": ok, "oos_ok": ok_oos}}


def g5_rule(costs_by_scenario: dict[str, dict]) -> dict:
    """G5: la acción Bayes cuesta estrictamente menos que argmax y que el umbral fijo en cada escenario."""
    if not costs_by_scenario:
        return {"zone": "C", "detail": {"error": "sin escenarios evaluables"}}
    ok = {k: v["bayes"] < v["argmax"] and v["bayes"] < v["threshold"] for k, v in costs_by_scenario.items()}
    return {"zone": "A" if all(ok.values()) else "C", "detail": ok}


def g6_rule(ours: dict, backbone: dict) -> dict:
    """G6: Ours-1 no pierde accuracy ni empeora ECE frente al backbone en el benchmark público."""
    ok = ours["acc"] >= backbone["acc"] - 1e-9 and ours["ece"] <= backbone["ece"] + 1e-9
    return {"zone": "A" if ok else "C", "detail": {"ours": ours, "backbone": backbone}}


def g8_rule(forced: dict, ours: dict) -> dict:
    """G8: precisión en respondibles + abstención en no respondibles supera al p0 forzado en >= 10 pp."""
    s_f = forced["acc_answerable"] + forced["abstain_unanswerable"]; s_o = ours["acc_answerable"] + ours["abstain_unanswerable"]
    return {"zone": "A" if s_o - s_f >= 0.10 else "C", "detail": {"forced": s_f, "ours": s_o}}


def g8_rule_strict(forced: dict, ours: dict) -> dict:
    """R2-D (hip-14): zona A si no se pierde precisión en respondibles (>= forzado - 0.02) y se abstiene en >= 50 % de los :noev."""
    ok = ours["acc_answerable"] >= forced["acc_answerable"] - 0.02 and ours["abstain_unanswerable"] >= 0.5
    return {"zone": "A" if ok else "C", "detail": {"forced_acc": forced["acc_answerable"], "ours_acc": ours["acc_answerable"],
            "abstain_unanswerable": ours["abstain_unanswerable"], "tpr": ours.get("tpr"), "fpr": ours.get("fpr")}}


def g9_rule(curves_weak: dict[str, dict], curves_strong: dict[str, dict], n: int = 1000, config: str = "B2") -> dict:
    """G9 (borde): la memoria ayuda más al backbone débil. Δ = acc(config@n) − acc(B0@0); A si Δ_débil > Δ_fuerte en ≥ 2 dominios."""
    detail = {}
    for d in curves_weak:
        if d not in curves_strong:
            continue
        dw = _acc(curves_weak[d], f"{config}@{n}") - _acc(curves_weak[d], "B0@0")
        ds = _acc(curves_strong[d], f"{config}@{n}") - _acc(curves_strong[d], "B0@0")
        detail[d] = {"delta_weak": dw, "delta_strong": ds, "weak_gains_more": dw > ds}
    wins = sum(v["weak_gains_more"] for v in detail.values())
    return {"zone": "A" if wins >= 2 and len(detail) >= 3 else "C", "detail": detail}
