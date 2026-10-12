"""Formato de cable de POST /v1/systemone, derivado del adaptador `typesafe` de JevBench.

Fuente: vendor/jevbench/jevbench/adapters/typesafe.py (+ base.py: build_question, http_post_json).

Request (typesafe.py:36-40, base.py:72-81):
  {"state": str, "model": str,
   "questions": {"decision": {"type": "noul"|"choice"|"score", "instructions": str, "criteria"?: ..., "question_key"?: str}}}   # question_key: extensión dec (clave del pool del cliente)
  - noul:   criteria = {"true": str, "false": str}
  - choice: criteria = {opcion: descripcion, ...} (las claves, en orden, son las opciones)
  - score:  criteria = [descripcion_nivel_0, descripcion_nivel_1, ...] (nivel = indice)
Cabeceras (typesafe.py:50-57): Authorization: Bearer <key>, Content-Type: application/json (no se valida).

Response 200 (typesafe.py:74-102):
  {"model": str, "usage": usage or {}, "answers": {"decision": {"type": <mismo qtype>, ...}}}
  - noul:   "noul": float en [0,1] = P(yes) numerico (no bool)         (l.86-91)
  - choice: "choice": una de las opciones y "probabilities": {opcion: p} (l.92-97)
  - score:  "probabilities": {"0": p, "1": p, ...} (claves = indices)    (l.98-102)
  - usage: {} en modo benchmark; en modo memory {"items": N, "memory_hits": ítems con precedentes (masa > 0),
    "unmapped": ítems sin question_key del pool (camino solo-backbone)}.
Errores: el adaptador marca error si status != 200, si falta answers.decision o si
type no coincide (l.71-84); aqui se responde 422 con {"error": ...} ante un request invalido.
"""
from __future__ import annotations
from dec.schema import Decision, DecisionItem, Option


def parse_request(body: dict) -> tuple[str, list[DecisionItem]]:
    """Convierte el body en (modelo, items); el id de cada item es la clave de la pregunta."""
    if not isinstance(body, dict) or not isinstance(body.get("questions"), dict) or not body["questions"]:
        raise ValueError("falta 'questions'")
    state = body.get("state", "")
    if not isinstance(state, (str, dict, list)):
        state = str(state)   # dict/list se conservan: jevk5 los embebe como JSON nativo, el resto usa state_text()
    model = str(body.get("model") or "dec")
    items: list[DecisionItem] = []
    for qk, q in body["questions"].items():
        qtype = q["type"]; crit = q.get("criteria")
        if qtype == "noul":
            crit = crit or {}
            opts = [Option("yes", str(crit.get("true", "yes"))), Option("no", str(crit.get("false", "no")))]
        elif qtype == "score":
            if not isinstance(crit, list):
                raise ValueError("score requiere criteria como lista")
            opts = [Option(str(i), str(d)) for i, d in enumerate(crit)]
        elif qtype == "choice":
            if isinstance(crit, dict):
                opts = [Option(str(k), str(v)) for k, v in crit.items()]
            elif isinstance(crit, list):
                opts = [Option(str(k), str(k)) for k in crit]
            else:
                raise ValueError("choice requiere criteria")
        else:
            raise ValueError(f"tipo desconocido: {qtype}")
        items.append(DecisionItem(id=str(qk), dataset="serve", state_id=f"serve:{qk}", state=state, qtype=qtype,
                                  instructions=str(q.get("instructions", "")), options=opts, gold=None,
                                  question_key=str(q.get("question_key") or f"serve:{qtype}")))   # extensión dec: el cliente puede declarar su question_key
    return model, items


def format_response(model: str, decisions: list[Decision], items: list[DecisionItem], usage: dict | None = None) -> dict:
    """Construye el body de respuesta que lee el adaptador; `usage` (opcional) lleva los contadores de memoria."""
    answers: dict[str, dict] = {}
    for d, it in zip(decisions, items):
        probs = {k: float(v) for k, v in d.probs.items()}
        if it.qtype == "noul":
            answers[it.id] = {"type": "noul", "noul": min(1.0, max(0.0, probs["yes"]))}
        elif it.qtype == "choice":
            answers[it.id] = {"type": "choice", "choice": d.action, "probabilities": probs}
        else:
            answers[it.id] = {"type": "score", "probabilities": probs}
    return {"model": model, "usage": usage or {}, "answers": answers}
