"""Reconstruye en el libro LOCAL (docs/gates/exploracion.md) los nodos `intento` de los gates a partir de docs/gates/results/G*.json.

Por qué: los gates corren en el pod y escriben allí su nodo, pero `pod.sh sync` pisaba el libro remoto; el libro de registro es el
local y este script lo pone al día de forma idempotente (no repite un nodo cuyo texto ya está). Usa el texto guardado en
`payload["ledger"]["text"]` si existe; si no (JSON antiguos), lo reconstruye desde zona y detalle y lo marca como reconstruido.
Uso (desde la raíz del repo): python scripts/gates/replay_ledger.py [--results docs/gates/results] [--dry-run]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.gates.common import record_attempt, HIP, LEDGER

ORDER = ("G0", "G1", "G4", "G8", "G3", "G2", "G5", "G6", "G7", "G9", "R2-A", "R2-B", "R2-C", "R2-D", "R2-E", "R2-F")   # spec + ronda 2


def _summary(gate: str, d: dict) -> str:
    det = d.get("detail") or {}
    if gate == "G0":
        return f"elegido {d.get('chosen')}"
    if isinstance(det, dict):
        flat = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in det.items() if not isinstance(v, (dict, list))}
        return json.dumps(flat, ensure_ascii=False)[:300] if flat else json.dumps(det, ensure_ascii=False, default=str)[:300]
    return json.dumps(det, ensure_ascii=False, default=str)[:300]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument("--results", default="docs/gates/results"); ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    ledger = Path(LEDGER).read_text() if Path(LEDGER).exists() else ""
    added = 0
    for gate in ORDER:
        p = Path(a.results) / f"{gate}.json"
        if not p.exists():
            print(f"{gate}: sin JSON, se omite"); continue
        d = json.loads(p.read_text())
        if gate == "G0":
            text = f"G0 medido: elegido {d.get('chosen')}"; sustiene = True
            evidence = json.dumps({r["backbone_id"]: (r["acc"], r["ece"]) for r in d.get("results", []) if "acc" in r})
        else:
            led = d.get("ledger") or {}
            text = led.get("text") or f"{gate} corrido: zona {d.get('zone')} [reconstruido desde {p.name}: {_summary(gate, d)}]"
            sustiene = bool(led.get("sustiene", d.get("zone") == "A")); evidence = json.dumps(d.get("detail"), default=str)
        if f"- texto: {text}" in ledger:
            print(f"{gate}: ya registrado"); continue
        print(f"{gate}: {'(dry) ' if a.dry_run else ''}+ {text[:110]}")
        if not a.dry_run:
            record_attempt(HIP[gate], text, evidence, sustiene); added += 1
    print(f"nodos añadidos: {added}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
