import json, shutil
from scripts.gates import common

def test_write_gate_and_record_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "GATES_DIR", tmp_path)
    led = tmp_path / "exploracion.md"
    shutil.copy("docs/gates/exploracion.md", led)
    monkeypatch.setattr(common, "LEDGER", led)
    common.write_gate("GX", {"zone": "A"})
    assert json.load(open(tmp_path / "GX.json"))["zone"] == "A"
    common.record_attempt("hip-1", "prueba", "evidencia", sustiene=True)
    assert "prueba" in led.read_text()
