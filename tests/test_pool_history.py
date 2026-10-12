"""Historial del cliente (dec-pool): validación con número de línea, resumen y avisos; resolución de device/dtype (CPU, sin modelos)."""
import json, subprocess, sys
from pathlib import Path
import pytest
from dec.pool import HistoryError, load_history, parse_history_row, summarize, pool_warnings, main as pool_main

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "examples" / "history_example.jsonl"
Q_NOUL = {"type": "noul", "instructions": "Escalate?", "criteria": {"true": "yes", "false": "no"}, "question_key": "s:esc"}
Q_CHOICE = {"type": "choice", "instructions": "Priority?", "criteria": {"low": "l", "high": "h"}, "question_key": "s:pri"}
Q_SCORE = {"type": "score", "instructions": "Impact?", "criteria": ["none", "minor", "major"], "question_key": "s:imp"}


def test_example_history_is_valid():
    items = load_history(EXAMPLE)
    assert 10 <= len(items) <= 20
    s = summarize(items)
    assert set(s) == {"support:priority", "support:escalate"}
    assert s["support:priority"]["classes"]["urgent"] >= 1 and s["support:escalate"]["classes"] == {"yes": 4, "no": 4}
    assert all(it.gold in it.option_keys() and it.dataset == "client" for it in items)


def test_row_normalisation():
    assert parse_history_row({"state": "x", "question": Q_NOUL, "decision": True}, 1).gold == "yes"
    assert parse_history_row({"state": "x", "question": Q_NOUL, "decision": "false"}, 1).gold == "no"
    it = parse_history_row({"state": {"a": 1}, "question": Q_SCORE, "decision": 2}, 1)
    assert it.gold == "2" and it.option_keys() == ["0", "1", "2"] and it.question_key == "s:imp"
    a = parse_history_row({"state": "x", "question": Q_CHOICE, "decision": "low"}, 1)
    b = parse_history_row({"state": "x", "question": Q_CHOICE, "decision": "low"}, 7)
    assert a.id == b.id and a.id.startswith("h-")   # id derivado del contenido: estable al reanudar


def test_bad_rows_report_line_numbers(tmp_path):
    rows = [json.dumps({"id": "ok", "state": "x", "question": Q_CHOICE, "decision": "low"}),
            "{not json",
            json.dumps({"state": "x", "question": {**Q_CHOICE, "question_key": ""}, "decision": "low"}),
            json.dumps({"state": "x", "question": Q_CHOICE, "decision": "medium"}),
            json.dumps({"id": "ok", "state": "y", "question": Q_CHOICE, "decision": "high"}),
            "",
            json.dumps({"state": "x", "question": Q_SCORE, "decision": 5}),
            json.dumps({"state": "x", "question": Q_NOUL}),
            json.dumps({"state": "", "question": Q_NOUL, "decision": "yes"}),
            json.dumps({"state": "x", "question": {**Q_CHOICE, "type": "multi"}, "decision": "low"})]
    p = tmp_path / "h.jsonl"; p.write_text("\n".join(rows) + "\n")
    with pytest.raises(HistoryError) as ei:
        load_history(p)
    msg = str(ei.value)
    for ln, frag in [(2, "JSON inválido"), (3, "question_key"), (4, "'medium'"), (5, "id duplicado"), (7, "fuera de rango"),
                     (8, "'decision'"), (9, "'state'"), (10, "question.type")]:
        assert any(l.startswith(f"línea {ln}:") and frag in l for l in msg.splitlines()), (ln, frag, msg)
    assert "línea 1:" not in msg and "línea 6:" not in msg and "8 filas inválidas" in msg


def test_cli_rejects_bad_history_without_loading_model(tmp_path, capsys):
    p = tmp_path / "h.jsonl"; p.write_text(json.dumps({"state": "x", "question": Q_CHOICE, "decision": "nope"}) + "\n")
    rc = pool_main(["build", "--history", str(p), "--backbone", "no/existe", "--out", str(tmp_path / "pool")])
    assert rc == 2 and "línea 1:" in capsys.readouterr().err


def test_pool_warnings():
    meta = {"question_keys": summarize(load_history(EXAMPLE))}
    w = pool_warnings(meta)
    assert any(x.startswith("support:priority:") and "< 200" in x for x in w)
    assert any(x.startswith("support:escalate:") and "< 50" in x for x in w)
    big = {"question_keys": {"q": {"n": 400, "qtype": "noul", "classes": {"yes": 200, "no": 200}}}}
    assert pool_warnings(big) == []


def test_device_and_dtype_resolution(monkeypatch):
    import torch
    from dec.backbone.model import resolve_device, resolve_dtype
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
    assert resolve_device("auto") == "cpu" and resolve_device("cpu") == "cpu"
    with pytest.raises(RuntimeError):
        resolve_device("cuda")
    with pytest.raises(ValueError):
        resolve_device("tpu")
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert resolve_device("auto") == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == "cuda"
    assert resolve_dtype("auto", "cuda") == "bf16" and resolve_dtype("auto", "cpu") == "fp32" and resolve_dtype("auto", "mps") == "fp32"
    assert resolve_dtype("fp32", "cuda") == "fp32" and resolve_dtype("bf16", "cpu") == "bf16"   # explícito: se respeta (con aviso)


def test_dec_pool_info_cli_on_missing_dir(tmp_path):
    r = subprocess.run([sys.executable, "-m", "dec.pool", "info", str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2 and "pool.json" in r.stderr
