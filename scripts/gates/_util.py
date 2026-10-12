"""Piezas compartidas por los scripts de gates G1-G8: backbone por defecto y cierre (escribir, copiar, registrar, imprimir)."""
from __future__ import annotations
import json, shutil
from pathlib import Path
from scripts.gates.common import write_gate, record_attempt, HIP


def default_backbone(results_dir: str = "results") -> str:
    """Backbone elegido por G0 (clave `chosen` de results/gates/G0.json)."""
    return json.loads((Path(results_dir) / "gates" / "G0.json").read_text())["chosen"]


def finish(gate: str, verdict: dict, inputs: dict, extra: dict | None = None, text: str | None = None, numbers: str = "") -> int:
    """Escribe results/gates/<gate>.json, lo copia a docs/gates/results/, registra el intento en el libro e imprime el JSON."""
    ledger_text = (text or f"{gate} corrido: zona {verdict['zone']}") + (f" [{numbers}]" if numbers else "")
    payload = {"gate": gate, **verdict, "inputs": inputs, **(extra or {}), "ledger": {"text": ledger_text, "sustiene": verdict["zone"] == "A"}}
    p = write_gate(gate, payload)
    Path("docs/gates/results").mkdir(parents=True, exist_ok=True); shutil.copy(p, f"docs/gates/results/{gate}.json")
    record_attempt(HIP[gate], ledger_text, json.dumps(verdict["detail"], default=str), sustiene=(verdict["zone"] == "A"))   # el libro de registro es el local: ver replay_ledger.py
    print(json.dumps(payload, indent=2, default=str))
    return 0
