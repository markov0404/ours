# Resultados de la exploración — capa de decisión `dec` (2026-10-12)

Backbone elegido en G0: **JevK5** (`alibiserikbay/JevK5@main:jevk5:bf16`, T = 1.22). Todas las cifras de esta página salen de los
JSON versionados en este directorio y de los nodos `int-*` del libro `docs/gates/exploracion.md`. Hardware: GPU A40 48 GB
(campaña, curvas y gates) y una H100 80GB solo para G7.

## 1. Cuadro de gates (reglas pre-registradas en la spec §7)

| Gate | Pregunta | Regla | Zona | Medido |
|---|---|---|---|---|
| G0 | ¿Qué backbone? | max acc en los 231 públicos; ECE desempata | JevK5 | JevK5 0.875 / ECE 0.019 vs Qwen3.5-4B 0.693 / 0.059. Por tier = model card v0.3 (0.944 / 1.000 / 0.775 vs 0.784) |
| G1 | ¿El precedente ayuda? | +3 pp y menor ECE que B1 en ≥ 2 de plumb/massive/clinc a n = 1000 | **C** | plumb +0.2 pp, massive +1.3 pp, clinc +0.2 pp |
| G4 | ¿Riesgo acotado y automatización? | riesgo ≤ α y automatización ≥ 0.5 en α ∈ {0.02, 0.05} (clinc) + AUROC OOS mejora | **C** | sin umbral factible (cota mínima 0.074 > 0.05); OOS sí mejora 0.772 → 0.790 |
| G8 | ¿Hueco → escalar? | (acc respondibles + abstención sin evidencia) − forzado ≥ 0.10 | **A (débil)** | forzado 0.788 vs 0.684 + 0.295 = 0.979; TPR 0.30 / FPR 0.21 del detector léxico |
| G3 | ¿La coherencia reduce incoherencia sin perder acc? | ≥ 50 % relativo en ≥ 2 dominios | **A** | massive 0.324 → 0.009 (acc 0.713 → 0.794); civil 0.004 → 0.000 |
| G2 | ¿φ por pregunta supera a la métrica global? | +1.5 pp en plumb y civil, sin regresión | **C** | plumb −0.0, civil −0.6, massive +1.0, clinc +0.1 pp |
| G5 | ¿La decisión por coste es más barata? | bayes < argmax y < umbral 0.5 en TODOS los escenarios | **C** | gana 4/6, empata 1 (B/threat: nadie marca amenazas), pierde 1 (C/toxicity: escala de más) |
| G6 | ¿No empeora en público con memoria externa? | acc ≥ y ECE ≤ que B0 en JevBench | **C** | acc igual (0.8745) pero ECE 0.040 vs 0.019: la temperatura ajustada en el pool externo (12 k ítems sin `question_key` compatible, cobertura de memoria 0 %) empeora la calibración |
| G7 | ¿La capa cabe en el presupuesto? | sobrecoste p50 < 3 ms en H100 | **A** | 2.44 ms con 21 k precedentes (búsqueda en GPU); 2.33 ms con 3 k; CPU 35.9 / 9.2 ms |
| G9 | ¿La memoria ayuda más al backbone débil? | Δ(B2@1000) del 0.6B > Δ del 4B en ≥ 2 de 3 dominios | **A** | Qwen3-0.6B: massive +27.6 pp (0.345 → 0.621), clinc +25.1 pp (0.224 → 0.474), plumb +0.1; JevK5: +4.6 / +0.3 / +0.3 |

Orden de ejecución: G0, G1, G4, G8, G3, G2, G5, G6, G7, G9 (el de la spec).

## 2. Curvas (JevK5, n = 1000 precedentes etiquetados, media ± sd de 3 semillas; acc / ECE)

| Dataset (n eval) | B0 solo backbone | B1 +temperatura | B2 memoria MiniLM | B3 memoria h | Ours‑1 (+métrica global) | Ours‑2 (+φ por pregunta) |
|---|---|---|---|---|---|---|
| plumb (1133) | 0.794 / 0.030 | 0.794 / 0.026 | 0.796 / 0.025 | 0.796 / 0.025 | 0.796 / 0.025 | 0.796 / 0.026 |
| massive (5948) | 0.700 / 0.044 | 0.700 / 0.034 | 0.745 / 0.141 | 0.712 / 0.091 | 0.713 / 0.093 | 0.723 / 0.124 |
| clinc (5500) | 0.765 / 0.063 | 0.765 / 0.045 | 0.768 / 0.064 | 0.767 / 0.050 | 0.767 / 0.051 | 0.768 / 0.056 |
| civil (21000) | 0.801 / 0.043 | 0.801 / 0.041 | 0.912 / 0.083 | 0.909 / 0.073 | 0.909 / 0.072 | 0.902 / 0.035 |
| banking77 (3076) | 0.700 / 0.063 | 0.700 / 0.039 | 0.725 / 0.174 | 0.704 / 0.110 | 0.705 / 0.116 | 0.708 / 0.129 |

Los agregados completos (todas las n y semillas) están en `curves/<dataset>.json` (JevK5) y `curves_qwen06/<dataset>.json` (Qwen3-0.6B).

### Curvas de aprendizaje: ¿mejora a medida que conoce al cliente?

Precisión de Ours‑1 según el número de precedentes etiquetados (B0 = sin precedentes):

| Dataset | B0 | n = 50 | n = 200 | n = 1000 |
|---|---|---|---|---|
| plumb | 0.794 | **0.763** | 0.797 | 0.796 |
| massive | 0.700 | 0.700 | 0.700 | 0.713 |
| clinc | 0.765 | 0.765 | 0.766 | 0.767 |
| civil | 0.801 | 0.885 | 0.891 | 0.909 |
| banking77 | 0.700 | 0.700 | 0.700 | 0.705 |

Lectura: con pocos precedentes (50) la memoria puede empeorar (plumb −3 pp); a partir de ~200 no hace daño y las ganancias crecen
despacio con n, salvo donde la convención de etiquetado difiere del prior del modelo (civil), donde la mayor parte del salto llega
ya con 50 ejemplos. Con un backbone débil (G9) la memoria vale mucho más: Qwen3-0.6B gana +25–28 pp en intents con 1000 precedentes
sobre embeddings MiniLM, frente a +0.3–4.6 pp de JevK5.

## 3. Lo que aprendimos (lecturas honestas)

**La coherencia estructural es la palanca más fuerte y es gratis.** En massive (intent + escenario, 2974 pedidos), declarar la
jerarquía y decodificar en conjunto baja las contradicciones de 32 % a 1 % y sube la precisión de 0.700 a 0.783 con el backbone
solo, y de 0.713 a 0.794 con memoria. Coste: 0.1 ms de CPU por decisión.

**El precedente ayuda poco en precisión donde el backbone ya sabe la tarea** (plumb, intents) **y mucho donde la convención de
etiquetado no está en su prior** (civil, +10.8 pp), pero esa ganancia hay que leerla por pregunta:

| civil, 3000 comentarios por pregunta | Positivos | JevK5 solo | Con memoria (Ours‑1) |
|---|---|---|---|
| amenaza | 53 | detecta 17, 64 falsas alarmas | detecta 0, 0 falsas alarmas |
| ataque a identidad | 119 | detecta 105, 1233 falsas alarmas | detecta 1, 5 falsas alarmas |
| toxicidad | 1500 | detecta 1038 (69 %), 347 falsas alarmas | detecta 1410 (94 %), 789 falsas alarmas |

Con ~143 etiquetas por pregunta la memoria casi no ve positivos raros y aprende "casi nunca es sí": la precisión sube porque deja de
alarmar. Para moderación eso es peor. Donde la clase es frecuente (toxicidad) la memoria sí mejora el recall. Conclusión: la memoria
debe ir acoplada a la decisión por coste (G5) y reportarse con recall por pregunta, nunca solo con precisión.

**La memoria empeora la calibración** (ECE massive 0.044 → 0.093, civil 0.043 → 0.072) salvo Ours‑2 en civil. La fusión se ajusta
por NLL sobre registros LOO y queda sobreconfiada en evaluación: una recalibración post-fusión es el siguiente intento natural.

**Features del precedente: MiniLM le gana a los estados ocultos del backbone en intents** (massive 0.745 vs 0.713). La elección de
representación del precedente queda como línea abierta (G9 mide la misma idea con un backbone débil).

**Riesgo selectivo.** Con δ = 0.05 (cota al 95 %) y rejilla fija de coberturas: plumb automatiza el 20 % con 0.4 % de error realizado
a α = 0.05; a α = 0.02 no hay umbral factible. En clinc (acc 0.77) no hay umbral factible ni a 0.05: G4 falla por el backbone, no por
la cota. La detección de fuera de alcance mejora con la masa de precedentes (AUROC 0.772 → 0.790).

**Decisión por coste (G5).** Con matrices asimétricas la acción Bayes es más barata que el argmax en 4 de 6 escenarios (p. ej.
perder una amenaza ×10: 358 vs 530). Empata cuando la memoria nunca marca amenazas (B/threat: 53 = 53) y pierde cuando el coste de
escalar es bajo y escala de más (C/toxicity: 1499 vs 1239). La regla exige ganar en todos → C.

**Huecos (G8).** El primer intento fue vacuo (el eje "oráculo" del router sin memoria declarada escalaba el 100 %). Aislado el eje de
huecos, pasa la regla pre-registrada pero el detector léxico es débil: escala el 29.5 % de los ítems sin evidencia y el 20.7 % de los
respondibles; en el 65 % de los pares la ablación no cambia el solapamiento léxico. La regla "suma" premia la abstención sin
penalizar bastante la pérdida de respondibles; para la próxima ronda se pre-registra una regla más estricta (acc respondibles ≥
forzado − 0.02 y abstención ≥ 0.5). No se cambia la de esta ronda.

## 4. Coste y latencia

- **Tokens por decisión: iguales al backbone solo.** Los precedentes entran por el estado oculto; no hay few-shot en el prompt ni
  tokens generados.
- **Capa (3 preguntas, 21 k precedentes):** 2.44 ms p50 en H100 con búsqueda en GPU (memoria 1.38, φ 0.47, router + política 0.34);
  35.9 ms en CPU exacta. Con 3 k precedentes: 2.33 / 9.2 ms.
- **Extremo a extremo con el harness real de JevBench (HTTP + backbone + capa, batch‑1):** A40 con flash-linear-attention: 62 ms
  (original) / 106 ms (hard) por decisión; H100 sin flash-linear-attention: 93 / 148 ms.
- **Preparación por cliente:** codificar 1000 precedentes ≈ 2–3 min de A40 (< 0.05 USD); 5 KB por precedente en fp16; ajuste de
  métrica, fusión y umbral en segundos de CPU. Lo caro son las etiquetas, que en el caso real ya existen (decisiones pasadas).

## 5. Dónde estamos frente a lo que existe

- **JevBench público (lo que miden Jev, Decision‑1 y JevK5):** empate exacto con JevK5 (es el backbone; `dec-serve` reproduce su
  model card a un ítem en hard: 86 vs 87 de 111). La capa no puede sumar ahí porque no hay precedentes de cliente, y G6 muestra que
  ajustar la fusión sobre un pool externo incompatible empeora la calibración: en modo benchmark sin pool propio hay que servir B0.
- **Lo que ellos no hacen:** decisiones coherentes entre preguntas (+8–9 pp donde hay estructura), automatización con error
  garantizado sobre los datos del cliente, adaptación a su convención de etiquetado donde hay suficientes ejemplos por clase, y
  detección de fuera de alcance por masa de precedentes.
- **Lo que no vendemos:** precisión bruta superior en benchmarks públicos, ni detección de clases raras a partir de pocas etiquetas.

## 6. Próximos intentos (pre-registro sugerido, no retroactivo)

1. Recalibración post-fusión (temperatura sobre la salida fusionada, ajustada en LOO) → objetivo: ECE de Ours ≤ B1 en todos los dominios.
2. Memoria sobre MiniLM + métrica (B2 + φ) y combinación de representaciones → objetivo: massive ≥ 0.745 con ECE ≤ 0.06.
3. Decisión por coste con prior corregido por frecuencia de clase (clases raras) → objetivo: recall de amenaza ≥ JevK5 solo con menos falsas alarmas.
4. Detector de huecos con similitud semántica (MiniLM) además de léxica → objetivo: TPR ≥ 0.6 a FPR ≤ 0.1 en plumb `:noev`.
5. Regla G8 estricta (ver §3) y G4 con α = 0.10 añadido como nivel informativo.

## 7. Reproducibilidad

```
# codificar cada dataset con el backbone (una pasada por ítem, reanudable; JevK5 en batch-1)
python scripts/cache_dataset.py --dataset plumb --split train --backbone alibiserikbay/JevK5 --style jevk5 --T 1.22 --token-budget 1
# auditoría, curvas y gates
dec-eval audit --dataset plumb
dec-eval curves --dataset plumb --backbone alibiserikbay/JevK5@main:jevk5:bf16
python scripts/gates/g1_precedent.py   # … g2 … g9, r2*
python scripts/gates/g7_latency.py     # requiere GPU
```
