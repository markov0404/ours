"""Utilidades compartidas por los scripts de gates: escritura de veredictos y registro en el libro de gates."""
from __future__ import annotations
import json, subprocess
from pathlib import Path

GATES_DIR = Path("results/gates")
LEDGER = Path("docs/gates/exploracion.md")
TOOL = Path("vendor/graph/exploracion.sh")
HIP = {"G0": "hip-1", "G1": "hip-2", "G4": "hip-3", "G8": "hip-4", "G3": "hip-5", "G2": "hip-6", "G5": "hip-7", "G6": "hip-8", "G7": "hip-9", "G9": "hip-10",
       # ronda 2 (2026-10-12): hipótesis derivadas, pre-registradas en el libro antes de correr
       "R2-A": "hip-11", "R2-B": "hip-12", "R2-C": "hip-13", "R2-D": "hip-14", "R2-E": "hip-15", "R2-F": "hip-16"}

def write_gate(name: str, payload: dict) -> Path:
    GATES_DIR.mkdir(parents=True, exist_ok=True)
    p = GATES_DIR / f"{name}.json"; p.write_text(json.dumps(payload, indent=2, ensure_ascii=False)); return p

def record_attempt(hip_id: str, text: str, evidence: str, sustiene: bool) -> None:
    flag = "--sostiene" if sustiene else "--refuta"
    subprocess.run(["bash", str(TOOL), "agregar", str(LEDGER), "intento", text, flag, hip_id,
                    "--preregistro", f"regla binding de {hip_id} en el libro de gates (spec §7), fijada antes de correr",
                    "--evidencia", evidence, "--estado", "sostenido" if sustiene else "refutado",
                    "--origen", "scripts/gates"], check=True, capture_output=True)
