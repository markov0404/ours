"""Gráficas SVG del README a partir de docs/gates/results (solo biblioteca estándar: sin numpy ni matplotlib).

Uso: python3 scripts/make_readme_charts.py  -> docs/img/*.svg
"""
from __future__ import annotations
import json
from pathlib import Path

R = Path("docs/gates/results"); OUT = Path("docs/img"); OUT.mkdir(parents=True, exist_ok=True)
PAL = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
FONT = "font-family='Inter, Helvetica, Arial, sans-serif'"


def _svg(w: int, h: int, body: str, title: str) -> str:
    return (f"<svg xmlns='http://www.w3.org/2000/svg' width='{w}' height='{h}' viewBox='0 0 {w} {h}' {FONT}>"
            f"<rect width='{w}' height='{h}' fill='white'/><text x='{w/2}' y='22' text-anchor='middle' font-size='15' font-weight='600' fill='#222'>{title}</text>{body}</svg>")


def _axes(x0, y0, x1, y1, ymin, ymax, ticks, fmt="{:.2f}") -> str:
    s = f"<line x1='{x0}' y1='{y1}' x2='{x1}' y2='{y1}' stroke='#444'/><line x1='{x0}' y1='{y0}' x2='{x0}' y2='{y1}' stroke='#444'/>"
    for t in ticks:
        y = y1 - (t - ymin) / (ymax - ymin) * (y1 - y0)
        s += f"<line x1='{x0}' y1='{y:.1f}' x2='{x1}' y2='{y:.1f}' stroke='#eee'/><text x='{x0-6}' y='{y+4:.1f}' text-anchor='end' font-size='11' fill='#444'>{fmt.format(t)}</text>"
    return s


def grouped_bars(path: Path, title: str, groups: list[str], series: dict[str, list[float]], ymin: float, ymax: float, ticks: list[float],
                 ylabel: str = "", fmt="{:.2f}", value_fmt="{:.3f}", w: int = 760, h: int = 380) -> None:
    x0, y0, x1, y1 = 70, 40, w - 20, h - 70
    body = _axes(x0, y0, x1, y1, ymin, ymax, ticks, fmt)
    gw = (x1 - x0) / len(groups); bw = gw * 0.8 / len(series)
    for gi, g in enumerate(groups):
        gx = x0 + gi * gw + gw * 0.1
        body += f"<text x='{gx + gw*0.4:.1f}' y='{y1+18}' text-anchor='middle' font-size='12' fill='#222'>{g}</text>"
        for si, (name, vals) in enumerate(series.items()):
            v = vals[gi]
            if v is None: continue
            yv = y1 - (max(min(v, ymax), ymin) - ymin) / (ymax - ymin) * (y1 - y0); x = gx + si * bw
            body += f"<rect x='{x:.1f}' y='{yv:.1f}' width='{bw*0.92:.1f}' height='{y1-yv:.1f}' fill='{PAL[si % len(PAL)]}'/>"
            body += f"<text x='{x + bw*0.46:.1f}' y='{yv-3:.1f}' text-anchor='middle' font-size='9.5' fill='#333'>{value_fmt.format(v)}</text>"
    lx = x0
    for si, name in enumerate(series):
        body += f"<rect x='{lx}' y='{h-28}' width='12' height='12' fill='{PAL[si % len(PAL)]}'/><text x='{lx+16}' y='{h-18}' font-size='11' fill='#222'>{name}</text>"
        lx += 16 + 7 * len(name) + 24
    if ylabel:
        body += f"<text x='14' y='{(y0+y1)/2}' transform='rotate(-90 14 {(y0+y1)/2})' text-anchor='middle' font-size='11' fill='#444'>{ylabel}</text>"
    path.write_text(_svg(w, h, body, title))


def lines(path: Path, title: str, xs: list[str], series: dict[str, list[float]], ymin: float, ymax: float, ticks: list[float],
          ylabel: str = "", w: int = 760, h: int = 380) -> None:
    x0, y0, x1, y1 = 70, 40, w - 160, h - 60
    body = _axes(x0, y0, x1, y1, ymin, ymax, ticks)
    step = (x1 - x0) / (len(xs) - 1)
    for i, xl in enumerate(xs):
        body += f"<text x='{x0 + i*step:.1f}' y='{y1+18}' text-anchor='middle' font-size='12' fill='#222'>{xl}</text>"
    for si, (name, vals) in enumerate(series.items()):
        pts = [(x0 + i * step, y1 - (v - ymin) / (ymax - ymin) * (y1 - y0)) for i, v in enumerate(vals) if v is not None]
        body += "<polyline fill='none' stroke='" + PAL[si % len(PAL)] + "' stroke-width='2.2' points='" + " ".join(f"{x:.1f},{y:.1f}" for x, y in pts) + "'/>"
        for x, y in pts:
            body += f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.2' fill='{PAL[si % len(PAL)]}'/>"
        body += f"<rect x='{x1+14}' y='{y0 + si*20}' width='12' height='12' fill='{PAL[si % len(PAL)]}'/><text x='{x1+30}' y='{y0 + si*20 + 10}' font-size='11' fill='#222'>{name}</text>"
    if ylabel:
        body += f"<text x='14' y='{(y0+y1)/2}' transform='rotate(-90 14 {(y0+y1)/2})' text-anchor='middle' font-size='11' fill='#444'>{ylabel}</text>"
    path.write_text(_svg(w, h, body, title))


def acc(agg: dict, key: str):
    return agg[key]["acc"][0] if key in agg else None


def main() -> None:
    curves = {d: json.load(open(R / "curves" / f"{d}.json"))["agg"] for d in ("plumb", "massive", "clinc", "civil", "banking77")}
    # 1. curvas de aprendizaje (Ours-1)
    lines(OUT / "learning_curves.svg", "Precisión de Ours-1 según precedentes etiquetados (JevK5, media de 3 semillas)",
          ["0 (B0)", "50", "200", "1000"],
          {d: [acc(a, "B0@0")] + [acc(a, f"ours1@{n}") for n in (50, 200, 1000)] for d, a in curves.items()},
          0.60, 0.95, [0.6, 0.7, 0.8, 0.9], "accuracy")
    # 2. escalera a n=1000
    grouped_bars(OUT / "ladder_n1000.svg", "Escalera a n = 1000: backbone solo vs baselines vs capa (accuracy)",
                 list(curves), {"B0 backbone": [acc(a, "B0@0") for a in curves.values()], "B1 +T": [acc(a, "B1@1000") for a in curves.values()],
                                "B2 mem. MiniLM": [acc(a, "B2@1000") for a in curves.values()], "Ours-1": [acc(a, "ours1@1000") for a in curves.values()],
                                "Ours-2 (φ)": [acc(a, "ours2@1000") for a in curves.values()]},
                 0.6, 0.95, [0.6, 0.7, 0.8, 0.9], "accuracy")
    # 3. ECE a n=1000
    grouped_bars(OUT / "ece_n1000.svg", "Calibración a n = 1000 (ECE, menor es mejor)", list(curves),
                 {"B0": [curves[d]["B0@0"]["ece"][0] for d in curves], "B1 +T": [curves[d]["B1@1000"]["ece"][0] for d in curves],
                  "B2": [curves[d]["B2@1000"]["ece"][0] for d in curves], "Ours-1": [curves[d]["ours1@1000"]["ece"][0] for d in curves],
                  "Ours-2": [curves[d]["ours2@1000"]["ece"][0] for d in curves]}, 0.0, 0.2, [0.0, 0.05, 0.1, 0.15, 0.2], "ECE")
    # 4. G9: backbone débil vs fuerte
    weak = {d: json.load(open(R / "curves_qwen06" / f"{d}.json"))["agg"] for d in ("plumb", "massive", "clinc")}
    grouped_bars(OUT / "g9_weak_backbone.svg", "G9: ganancia de la memoria (B2 @1000 − B0) según backbone", ["plumb", "massive", "clinc"],
                 {"Qwen3-0.6B (débil)": [acc(weak[d], "B2@1000") - acc(weak[d], "B0@0") for d in weak],
                  "JevK5 (fuerte)": [acc(curves[d], "B2@1000") - acc(curves[d], "B0@0") for d in weak]},
                 0.0, 0.32, [0.0, 0.1, 0.2, 0.3], "Δ accuracy", value_fmt="{:+.3f}")
    # 5. civil por pregunta
    cq = json.load(open(R / "civil_per_question.json"))["questions"]
    qs = ["threat", "identity_attack", "severe_toxicity", "obscene", "sexual_explicit", "insult", "toxicity"]
    grouped_bars(OUT / "civil_recall.svg", "civil: recall de positivos por pregunta (3000 comentarios), JevK5 solo vs con memoria", qs,
                 {"JevK5 solo": [cq[q]["B0"]["detected"] / cq[q]["positives"] for q in qs], "Ours-1": [cq[q]["ours1"]["detected"] / cq[q]["positives"] for q in qs]},
                 0.0, 1.0, [0, 0.25, 0.5, 0.75, 1.0], "recall", value_fmt="{:.2f}", w=860)
    grouped_bars(OUT / "civil_false_pos.svg", "civil: falsas alarmas por pregunta (de 3000)", qs,
                 {"JevK5 solo": [cq[q]["B0"]["false_pos"] for q in qs], "Ours-1": [cq[q]["ours1"]["false_pos"] for q in qs]},
                 0, 1300, [0, 300, 600, 900, 1200], "falsas alarmas", fmt="{:.0f}", value_fmt="{:.0f}", w=860)
    # 6. coherencia massive
    mq = json.load(open(R / "civil_per_question.json"))["massive_per_question"]
    grouped_bars(OUT / "massive_coherence.svg", "massive: decodificación conjunta (jerarquía intent → escenario)", ["accuracy B0", "accuracy Ours-1", "incoherencia"],
                 {"antes": [mq["joint_decode"]["B0"]["before"], mq["joint_decode"]["ours1"]["before"], mq["joint_decode"]["incoherence"]["before"]],
                  "después": [mq["joint_decode"]["B0"]["after"], mq["joint_decode"]["ours1"]["after"], mq["joint_decode"]["incoherence"]["after"]]},
                 0.0, 0.9, [0, 0.2, 0.4, 0.6, 0.8], "", w=640)
    # 7. latencia
    g7 = json.load(open(R / "G7.json"))["inputs"]["runs"]
    grouped_bars(OUT / "latency.svg", "Latencia de la capa por decisión de 3 preguntas (H100, p50 ms) y referencia del backbone",
                 ["capa GPU 21k", "capa GPU 3k", "capa CPU 21k", "capa CPU 3k", "backbone A40", "backbone H100"],
                 {"ms": [g7["cuda@21000"]["p50_ms"], g7["cuda@3000"]["p50_ms"], g7["cpu@21000"]["p50_ms"], g7["cpu@3000"]["p50_ms"], 62.0, 93.0]},
                 0, 100, [0, 25, 50, 75, 100], "ms", fmt="{:.0f}", value_fmt="{:.1f}", w=820)
    # 8. ronda 2: Ours-3 (memoria MiniLM + métrica) y recalibración post-fusión
    r2 = {d: json.load(open(R / "r2" / "curves" / f"{d}.json"))["agg"] for d in curves}
    grouped_bars(OUT / "r2_ours3_acc.svg", "Ronda 2: precisión a n = 1000 — backbone, Ours-1 y Ours-3 (memoria MiniLM + métrica)", list(curves),
                 {"B0 backbone": [acc(curves[d], "B0@0") for d in curves], "B2 mem. MiniLM": [acc(r2[d], "B2@1000") for d in curves],
                  "Ours-1": [acc(r2[d], "ours1@1000") for d in curves], "Ours-3": [acc(r2[d], "ours3@1000") for d in curves],
                  "Ours-3-cal": [acc(r2[d], "ours3cal@1000") for d in curves]}, 0.6, 0.95, [0.6, 0.7, 0.8, 0.9], "accuracy", w=820)
    grouped_bars(OUT / "r2_ours3_ece.svg", "Ronda 2: calibración a n = 1000 (ECE, menor es mejor)", list(curves),
                 {"B0": [curves[d]["B0@0"]["ece"][0] for d in curves], "B1 +T": [r2[d]["B1@1000"]["ece"][0] for d in curves],
                  "Ours-1": [r2[d]["ours1@1000"]["ece"][0] for d in curves], "Ours-1-cal": [r2[d]["ours1cal@1000"]["ece"][0] for d in curves],
                  "Ours-3": [r2[d]["ours3@1000"]["ece"][0] for d in curves], "Ours-3-cal": [r2[d]["ours3cal@1000"]["ece"][0] for d in curves]},
                 0.0, 0.2, [0.0, 0.05, 0.1, 0.15, 0.2], "ECE", w=900)
    grouped_bars(OUT / "r2_ours3_gain.svg", "Ronda 2: ganancia sobre el backbone solo (acc Ours-3-cal − B0, puntos)", list(curves),
                 {"Ours-1": [100 * (acc(r2[d], "ours1@1000") - acc(curves[d], "B0@0")) for d in curves],
                  "Ours-3-cal": [100 * (acc(r2[d], "ours3cal@1000") - acc(curves[d], "B0@0")) for d in curves]},
                 0.0, 12.0, [0, 3, 6, 9, 12], "pp", fmt="{:.0f}", value_fmt="{:+.1f}")
    print("ok:", sorted(p.name for p in OUT.glob("*.svg")))


if __name__ == "__main__":
    main()
