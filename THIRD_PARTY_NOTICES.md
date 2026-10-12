# Avisos de terceros

`dec` (licencia PolyForm Noncommercial 1.0.0) usa, descarga o evalúa sobre los siguientes componentes de terceros. Ninguno se redistribuye en este repositorio (se obtienen
de su origen en tiempo de ejecución); cada uno conserva su licencia.

## Modelos

| Componente | Origen | Licencia | Uso en `dec` |
|---|---|---|---|
| JevK5 (`alibiserikbay/JevK5`) | Hugging Face | Apache‑2.0 | backbone elegido en G0; estilo de prompt portado en `src/dec/backbone/jevk5_prompt.py` |
| Qwen3.5‑4B, Qwen3‑0.6B (`Qwen/*`) | Hugging Face | Apache‑2.0 | candidato en G0; backbone débil en G9 |
| all‑MiniLM‑L6‑v2 (`sentence-transformers`) | Hugging Face | Apache‑2.0 | embeddings de estados (B2) y de preguntas (φ) |
| Clef‑flash (`cloudflare/clef-flash`) | Hugging Face | ver ficha | solo sondeo de carga en G0 (no evaluado) |

## Datos y evaluación

| Componente | Origen | Licencia | Uso |
|---|---|---|---|
| JevBench (harness y 231 ítems públicos) | github.com/fstandhartinger/jevbench | MIT | comparación con cifras publicadas vía `dec-serve`; `vendor/jevbench` se clona, no se versiona |
| MASSIVE (en‑US) | Amazon / Hugging Face | CC BY 4.0 | dataset nivel 1 |
| CLINC150 (`clinc/clinc_oos`, config `plus`) | Hugging Face | CC BY 3.0 | dataset nivel 1 |
| Civil Comments (Jigsaw) | Hugging Face | CC0 1.0 | dataset nivel 2 |
| Banking77 (`mteb/banking77`) | Hugging Face | CC BY 4.0 | dataset nivel 2 |
| plumb | Hugging Face | ver la ficha del dataset | dataset nivel 1 (decisiones tipo JevBench) |

Las caches de codificaciones derivadas de estos datos se guardan en un bus privado de Hugging Face y no se publican.

## Bibliotecas principales

PyTorch (BSD‑3), Transformers (Apache‑2.0), flash‑linear‑attention (MIT), FAISS (MIT), sentence‑transformers (Apache‑2.0),
scikit‑learn (BSD‑3), SciPy (BSD‑3), NumPy (BSD‑3), pyarrow (Apache‑2.0), datasets y huggingface_hub (Apache‑2.0), FastAPI y
uvicorn (MIT y BSD‑3), loguru (MIT), tqdm (MIT/MPL‑2.0), safetensors (Apache‑2.0), pytest (MIT).

## Método

Clasificación selectiva con garantía: Geifman & El‑Yaniv, 2017. Conformal dividido: Vovk; Angelopoulos & Bates, 2021. Herramienta de
libro de exploración `vendor/graph/exploracion.sh`: del propio autor (proyecto GRAPH).
