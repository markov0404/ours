"""Vuelca el prompt renderizado por jevk5 (0.3.3) para un ejemplo fijo, como fixture de regresión.
Correr con el venv de jevk5 (en el pod: /workspace/venv-jevk5/bin/python)."""
import json
import jevk5
from jevk5.prompt import decision_options, prompt_text

STATE = "I was billed twice for order #4411. Please refund the duplicate charge today."
Q = {"type": "choice", "instructions": "Which team should handle this?",
     "criteria": {"billing": "Payments and refunds", "tech": "Bugs", "sales": "New purchases"}}
texts = [t for _, t in decision_options(Q)]
text = prompt_text(STATE, Q["instructions"], texts)
with open("tests/fixtures/jevk5_prompt_example.json", "w") as f:
    json.dump({"jevk5_version": jevk5.__version__, "state": STATE, "question": Q, "prompt": text}, f, indent=2)
print(text)
