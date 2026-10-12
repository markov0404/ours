# Grafo de exploración · dec
> Esquema-exploracion: 1 · Sembrado: spec 2026-10-11 · Última tarea: ninguna

## Resumen
Libro de gates de la capa de decisión `dec`. Cada gate es una hipótesis con su regla binding (preregistro) escrita antes de correr, su prior subjetivo y su evidencia de refutación o sostén. Reglas fijadas antes de correr (diseño del 2026-10-11). Orden de ejecución: G0, G1, G4, G8, G3, G2, G5, G6, G7, G9.

## Nodos
<!-- exploracion:nodos:start -->
### NODO hip-1
- tipo: hipótesis
- estado: vivo
- texto: G0 backbone: entre JevK5 v0.3, Qwen3.5-4B sin entrenar (lectura por sufijos) y Clef-flash 9B, uno domina en precision con ECE como desempate sobre los 231 items publicos de JevBench
- preregistro: Medir precision, ECE, latencia y tasa de flips al reordenar opciones en el mismo harness; elegir por precision; 9B solo si el 4B queda mas de 3 pp por debajo
- evidencia-refutacion: 
- origen: spec 2026-10-11 §7 · prior —
- fecha: 2026-10-11
### NODO hip-2
- tipo: hipótesis
- estado: vivo
- texto: G1 precedente: la memoria de precedentes con metrica diagonal aprendida global (Ours-1) mejora sobre temperatura ajustada (B1)
- preregistro: Ours-1 supera a B1 en al menos 3 pp de precision a n=1000 en al menos 2 de 3 dominios de nivel 1 (plumb, MASSIVE, CLINC) y con ECE menor; si falla se cierra el eje precedente
- evidencia-refutacion: 
- origen: spec 2026-10-11 §7 · prior 0.70
- fecha: 2026-10-11
### NODO hip-3
- tipo: hipótesis
- estado: vivo
- texto: G4 riesgo: el umbral selectivo con garantia controla el error real en lo automatizado
- preregistro: En CLINC: error real en decisiones automatizadas <= alfa para alfa en {0.02, 0.05} con al menos 50% de automatizacion; AUROC de fuera de alcance con masa+confianza mayor que con confianza sola
- evidencia-refutacion: 
- origen: spec 2026-10-11 §7 · prior 0.75
- fecha: 2026-10-11
### NODO hip-4
- tipo: hipótesis
- estado: vivo
- texto: G8 huecos: el detector de huecos distingue estados que permiten responder de los que no
- preregistro: En familias trap/ambiguous/adversarial de plumb con items sin respuesta construidos por ablacion de la oracion con la evidencia: precision en respondibles + abstencion correcta en no respondibles supera al p0 de eleccion forzada en al menos 10 pp
- evidencia-refutacion: 
- origen: spec 2026-10-11 §7 · prior 0.65
- fecha: 2026-10-11
### NODO hip-5
- tipo: hipótesis
- estado: vivo
- texto: G3 coherencia: la decodificacion conjunta reduce la incoherencia entre decisiones acopladas sin perder precision
- preregistro: En MASSIVE (intent in scenario) y civil_comments (severe implica toxic): tasa de conjuntos incoherentes baja al menos 50% relativo con precision no inferior
- evidencia-refutacion: 
- origen: spec 2026-10-11 §7 · prior 0.80
- fecha: 2026-10-11
### NODO hip-6
- tipo: hipótesis
- estado: vivo
- texto: G2 phi: la metrica condicionada por la pregunta (Ours-2) supera a la metrica global aprendida (Ours-1)
- preregistro: Ours-2 supera a Ours-1 en al menos 1.5 pp en plumb y civil_comments (multi-pregunta) sin regresion en el resto; si falla se retira phi
- evidencia-refutacion: 
- origen: spec 2026-10-11 §7 · prior 0.35
- fecha: 2026-10-11
### NODO hip-7
- tipo: hipótesis
- estado: vivo
- texto: G5 coste: la accion Bayes-optima reduce el coste realizado frente a argmax y umbral fijo
- preregistro: Bajo 3 matrices de costes en civil_comments y familias routing/judge de plumb: coste total menor que argmax y que umbral fijo
- evidencia-refutacion: 
- origen: spec 2026-10-11 §7 · prior 0.70
- fecha: 2026-10-11
### NODO hip-8
- tipo: hipótesis
- estado: vivo
- texto: G6 publico: la capa en modo benchmark no es peor que el backbone en JevBench
- preregistro: 231 items publicos: precision >= backbone y ECE no peor
- evidencia-refutacion: 
- origen: spec 2026-10-11 §7 · prior 0.50
- fecha: 2026-10-11
### NODO hip-9
- tipo: hipótesis
- estado: vivo
- texto: G7 latencia: toda la capa cabe en el presupuesto
- preregistro: Sobrecoste de la capa < 3 ms p50 medido en H100
- evidencia-refutacion: 
- origen: spec 2026-10-11 §7 · prior 0.90
- fecha: 2026-10-11
### NODO hip-10
- tipo: hipótesis
- estado: vivo
- texto: G9 borde: la memoria ayuda mas cuanto mas debil es el backbone
- preregistro: Qwen3-0.6B + memoria sobre MiniLM: delta de precision a n=1000 mayor que el delta del 4B en al menos 2 de 3 dominios
- evidencia-refutacion: 
- origen: spec 2026-10-11 §7 · prior 0.55
- fecha: 2026-10-11
### NODO int-1
- tipo: intento
- estado: sostenido
- texto: G0 medido: elegido alibiserikbay/JevK5@main:jevk5:bf16
- preregistro: regla binding de hip-1 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"alibiserikbay/JevK5@main:jevk5:bf16": [0.8614718614718615, 0.019165724902958085], "Qwen/Qwen3.5-4B@main:numeric:bf16": [0.6753246753246753, 0.048796301925337175]}
- origen: scripts/gates
- fecha: 2026-10-11
### NODO int-3
- tipo: intento
- estado: sostenido
- texto: G0 medido: elegido alibiserikbay/JevK5@main:jevk5:bf16
- preregistro: regla binding de hip-1 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"alibiserikbay/JevK5@main:jevk5:bf16": [0.8744588744588745, 0.018743954308621306], "Qwen/Qwen3.5-4B@main:numeric:bf16": [0.6926406926406926, 0.05938206580810217]}
- origen: scripts/gates
- fecha: 2026-10-11
### NODO int-4
- tipo: intento
- estado: refutado
- texto: G1 corrido: zona C [reconstruido desde G1.json: {"plumb": false, "massive": false, "clinc": false}]
- preregistro: regla binding de hip-2 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"plumb": false, "massive": false, "clinc": false}
- origen: scripts/gates
- fecha: 2026-10-12
### NODO int-5
- tipo: intento
- estado: refutado
- texto: G4 corrido: zona C [reconstruido desde G4.json: {"risk_ok": false, "oos_ok": true, "reason": "calibración insuficiente"}]
- preregistro: regla binding de hip-3 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"risk_ok": false, "oos_ok": true, "insufficient_seeds": {"0.02": [0, 1, 2], "0.05": [0, 1, 2]}, "reason": "calibraci\u00f3n insuficiente"}
- origen: scripts/gates
- fecha: 2026-10-12
### NODO int-6
- tipo: intento
- estado: sostenido
- texto: G8 corrido: zona A [reconstruido desde G8.json: {"forced": 0.7876, "ours": 0.9793}]
- preregistro: regla binding de hip-4 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"forced": 0.7875647668393783, "ours": 0.9792746113989637}
- origen: scripts/gates
- fecha: 2026-10-12
### NODO int-7
- tipo: intento
- estado: sostenido
- texto: G3 corrido: zona A [reconstruido desde G3.json: {"massive": true, "civil": true}]
- preregistro: regla binding de hip-5 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"massive": true, "civil": true}
- origen: scripts/gates
- fecha: 2026-10-12
### NODO int-8
- tipo: intento
- estado: refutado
- texto: G2 corrido: zona C [reconstruido desde G2.json: {"plumb": -0.0, "civil": -0.0062, "massive": 0.0096, "clinc": 0.0005}]
- preregistro: regla binding de hip-6 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"plumb": -1.5425263072410544e-06, "civil": -0.006158730158730252, "massive": 0.00963909437345889, "clinc": 0.0005454545454545157}
- origen: scripts/gates
- fecha: 2026-10-12
### NODO int-9
- tipo: intento
- estado: refutado
- texto: G5 corrido: zona C [reconstruido desde G5.json: {"A/civil:threat": true, "B/civil:threat": false, "C/civil:threat": true, "A/civil:toxicity": true, "B/civil:toxicity": true, "C/civil:toxicity": false}]
- preregistro: regla binding de hip-7 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"A/civil:threat": true, "B/civil:threat": false, "C/civil:threat": true, "A/civil:toxicity": true, "B/civil:toxicity": true, "C/civil:toxicity": false}
- origen: scripts/gates
- fecha: 2026-10-12
### NODO int-10
- tipo: intento
- estado: refutado
- texto: G6 corrido: zona C [reconstruido desde G6.json: {"ours": {"acc": 0.8744588744588745, "ece": 0.04003155747533362, "brier": 0.18467514216899872}, "backbone": {"acc": 0.8744588744588745, "ece": 0.01874395985623976, "brier": 0.18291589617729187}}]
- preregistro: regla binding de hip-8 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"ours": {"acc": 0.8744588744588745, "ece": 0.04003155747533362, "brier": 0.18467514216899872}, "backbone": {"acc": 0.8744588744588745, "ece": 0.01874395985623976, "brier": 0.18291589617729187}}
- origen: scripts/gates
- fecha: 2026-10-12
### NODO int-11
- tipo: intento
- estado: sostenido
- texto: G7 corrido: zona A [reconstruido desde G7.json: {"p50_ms_cuda_21k": 2.4409, "threshold_ms": 3.0, "gpu": "NVIDIA H100 80GB HBM3", "is_h100": true}]
- preregistro: regla binding de hip-9 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"p50_ms_cuda_21k": 2.4408505123574287, "threshold_ms": 3.0, "gpu": "NVIDIA H100 80GB HBM3", "is_h100": true}
- origen: scripts/gates
- fecha: 2026-10-12
### NODO int-12
- tipo: intento
- estado: sostenido
- texto: G9 corrido: zona A [reconstruido desde G9.json: {"plumb": {"delta_weak": 0.0005858435577401622, "delta_strong": 0.002933270396983323, "weak_gains_more": false}, "massive": {"delta_weak": 0.275835014570724, "delta_strong": 0.04556153328850032, "weak_gains_more": true}, "clinc": {"delta_weak": 0.25054545454545457, "delta_strong": 0.0029696969696968]
- preregistro: regla binding de hip-10 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"plumb": {"delta_weak": 0.0005858435577401622, "delta_strong": 0.002933270396983323, "weak_gains_more": false}, "massive": {"delta_weak": 0.275835014570724, "delta_strong": 0.04556153328850032, "weak_gains_more": true}, "clinc": {"delta_weak": 0.25054545454545457, "delta_strong": 0.0029696969696968445, "weak_gains_more": true}}
- origen: scripts/gates
- fecha: 2026-10-12
### NODO int-13
- tipo: intento
- estado: refutado
- texto: G8 intento 1 (vacuo, sustituido por el intento 2): umbrales ajustados (entropía 0.55, overlap 0.7) escalaron el 100 % de los respondibles porque route() pone el eje oráculo en rojo con thr.mass=0 (sin memoria); ours = 0.000 + 1.000 = 1.000 vs forzado 0.788. Registrado como fallo de instrumentación, no como evidencia
- preregistro: regla binding de hip-4 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"forced": 0.7876, "ours": 1.0, "coverage_answerable": 0.0, "causa": "oraculo rojo con thr.mass=0"}
- origen: scripts/gates
- fecha: 2026-10-12
### NODO hip-11
- tipo: hipótesis
- estado: vivo
- texto: R2-A recalibración: una temperatura ajustada en LOO sobre la salida fusionada (Ours-1-cal) recupera la calibración que pierde la memoria sin perder precisión
- preregistro: A n=1000, 3 semillas, en los 5 dominios: ECE(Ours-1-cal) <= ECE(B1) + 0.005 y acc(Ours-1-cal) >= acc(Ours-1) - 0.002; se compara con las curvas de la ronda 1 (mismas caches y semillas)
- evidencia-refutacion: 
- origen: ronda 2, 2026-10-12 · prior 0.65
- fecha: 2026-10-12
### NODO hip-12
- tipo: hipótesis
- estado: vivo
- texto: R2-B memoria MiniLM + métrica: la memoria sobre embeddings MiniLM con métrica diagonal aprendida (Ours-3) supera a Ours-1 en intents sin perder en el resto
- preregistro: A n=1000, 3 semillas: acc(Ours-3) >= acc(B2) en massive, clinc y banking77, y acc(Ours-3) >= acc(Ours-1) - 0.002 en plumb y civil; con recalibración ECE(Ours-3-cal) <= 0.06 en massive
- evidencia-refutacion: 
- origen: ronda 2, 2026-10-12 · prior 0.55
- fecha: 2026-10-12
### NODO hip-13
- tipo: hipótesis
- estado: vivo
- texto: R2-C coste con prior por frecuencia: corregir p_mem por la frecuencia de clase de la memoria y decidir por coste recupera el recall de clases raras
- preregistro: En civil threat e identity_attack con la matriz A (perder un positivo x10), n=1000 seed 0: recall >= recall de B0 argmax y falsos positivos <= los de B0; coste realizado de la acción Bayes < coste del argmax de B0 en ambas preguntas
- evidencia-refutacion: 
- origen: ronda 2, 2026-10-12 · prior 0.50
- fecha: 2026-10-12
### NODO hip-14
- tipo: hipótesis
- estado: vivo
- texto: R2-D detector de huecos semántico: la similitud MiniLM entre la pregunta y las frases del estado separa respondibles de sin-evidencia mejor que el solapamiento léxico
- preregistro: En plumb, pares (respondible, :noev) de evaluación con umbrales ajustados en el pool: TPR >= 0.6 a FPR <= 0.1; regla G8 estricta: acc_answerable >= forzado - 0.02 y abstención en :noev >= 0.5
- evidencia-refutacion: 
- origen: ronda 2, 2026-10-12 · prior 0.45
- fecha: 2026-10-12
### NODO hip-15
- tipo: hipótesis
- estado: vivo
- texto: R2-E riesgo a alfa 0.10 (informativo): el umbral selectivo automatiza al menos la mitad de clinc con riesgo realizado <= 0.10
- preregistro: clinc, Ours-1 (o Ours-1-cal si R2-A sostiene), n=1000, 3 semillas: automatización >= 0.5 y riesgo realizado <= 0.10 en las 3 semillas; delta 0.05
- evidencia-refutacion: 
- origen: ronda 2, 2026-10-12 · prior 0.60
- fecha: 2026-10-12
### NODO hip-16
- tipo: hipótesis
- estado: vivo
- texto: R2-F modo memoria en dec-serve: con un pool del cliente (plumb train, 1000 etiquetados) y mapeo declarado de question_key, el servidor reproduce la escalera a través del harness HTTP
- preregistro: Fila con el harness real sobre plumb test (131 ítems): acc igual a run_config(Ours-1, n=1000, seed 0) +-0.01, 0 errores de cable, latencia p50 <= 1.3x la del modo benchmark en la misma máquina
- evidencia-refutacion: 
- origen: ronda 2, 2026-10-12 · prior 0.70
- fecha: 2026-10-12
### NODO int-14
- tipo: intento
- estado: refutado
- texto: R2-A corrido: zona C [plumb: ECE 0.025 vs B1 0.026, acc 0.796 vs 0.796; massive: ECE 0.061 vs B1 0.043, acc 0.713 vs 0.713; clinc: ECE 0.048 vs B1 0.045, acc 0.767 vs 0.767; civil: ECE 0.022 vs B1 0.042, acc 0.908 vs 0.908; banking77: ECE 0.052 vs B1 0.039, acc 0.705 vs 0.705]
- preregistro: regla binding de hip-11 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"plumb": {"ece_cal": 0.025360586603552258, "ece_B1": 0.026234963378916686, "acc_cal": 0.7964903894299372, "acc_ours1": 0.7964903894299372, "ok": true}, "massive": {"ece_cal": 0.06116302008662316, "ece_B1": 0.04323481639182852, "acc_cal": 0.7134050661286707, "acc_ours1": 0.7134050661286707, "ok": false}, "clinc": {"ece_cal": 0.04828173072365196, "ece_B1": 0.04531991555474021, "acc_cal": 0.767151515151515, "acc_ours1": 0.767151515151515, "ok": true}, "civil": {"ece_cal": 0.022121067187142768, "ece_B1": 0.04221231463031166, "acc_cal": 0.9077936507936508, "acc_ours1": 0.9077936507936508, "ok": true}, "banking77": {"ece_cal": 0.051597868452857056, "ece_B1": 0.039486833128272385, "acc_cal": 0.7047030775899437, "acc_ours1": 0.7047030775899437, "ok": false}}
- origen: scripts/gates
- fecha: 2026-10-12
### NODO int-15
- tipo: intento
- estado: sostenido
- texto: R2-B corrido: zona A [plumb: ours3 0.796 vs ours1-0.002 0.794, ECE ours3cal 0.025; massive: ours3 0.751 vs B2 0.746, ECE ours3cal 0.041; clinc: ours3 0.768 vs B2 0.768, ECE ours3cal 0.061; civil: ours3 0.910 vs ours1-0.002 0.906, ECE ours3cal 0.020; banking77: ours3 0.728 vs B2 0.728, ECE ours3cal 0.053]
- preregistro: regla binding de hip-12 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"plumb": {"acc_ours3": 0.7964883266486739, "ref": 0.7944903894299372, "ref_name": "ours1-0.002", "ece_ours3cal": 0.025418665887946116, "ok": true}, "massive": {"acc_ours3": 0.7510087424344317, "ref": 0.7462452364940596, "ref_name": "B2", "ece_ours3cal": 0.04103058944812232, "ok": true}, "clinc": {"acc_ours3": 0.7677575757575757, "ref": 0.7676363636363637, "ref_name": "B2", "ece_ours3cal": 0.06114995416998865, "ok": true}, "civil": {"acc_ours3": 0.9103492063492062, "ref": 0.9057936507936508, "ref_name": "ours1-0.002", "ece_ours3cal": 0.019812219018974012, "ok": true}, "banking77": {"acc_ours3": 0.7277850021673169, "ref": 0.7277850021673169, "ref_name": "B2", "ece_ours3cal": 0.05296786316950238, "ok": true}}
- origen: scripts/gates
- fecha: 2026-10-12
### NODO int-16
- tipo: intento
- estado: refutado
- texto: R2-C corrido: zona C [threat: B0 recall 0.32 FP 64 coste 424 | prior+bayes recall 0.94 FP 1464 coste 1494; identity_attack: B0 recall 0.88 FP 1233 coste 1373 | prior+bayes recall 1.00 FP 2486 coste 2486]
- preregistro: regla binding de hip-13 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"civil:threat": {"ok": false, "B0_argmax": {"recall": 0.32075471698113206, "detected": 17, "positives": 53, "false_pos": 64, "cost": 424.0}, "ours1_prior_bayes": {"recall": 0.9433962264150944, "detected": 50, "positives": 53, "false_pos": 1464, "cost": 1494.0}}, "civil:identity_attack": {"ok": false, "B0_argmax": {"recall": 0.8823529411764706, "detected": 105, "positives": 119, "false_pos": 1233, "cost": 1373.0}, "ours1_prior_bayes": {"recall": 1.0, "detected": 119, "positives": 119, "false_pos": 2486, "cost": 2486.0}}}
- origen: scripts/gates
- fecha: 2026-10-12
### NODO int-17
- tipo: intento
- estado: refutado
- texto: R2-D corrido: zona C [TPR 0.570 FPR 0.404 acc 0.508 vs forzado 0.788]
- preregistro: regla binding de hip-14 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"forced_acc": 0.7875647668393783, "ours_acc": 0.5077720207253886, "abstain_unanswerable": 0.5699481865284974, "tpr": 0.5699481865284974, "fpr": 0.40414507772020725, "rule_round1": {"zone": "A", "detail": {"forced": 0.7875647668393783, "ours": 1.077720207253886}}}
- origen: scripts/gates
- fecha: 2026-10-12
### NODO int-18
- tipo: intento
- estado: refutado
- texto: R2-E corrido: zona C [s0: aut 0.523 riesgo 0.1019 theta 0.797; s1: aut 0.433 riesgo 0.0852 theta 0.830; s2: aut 0.524 riesgo 0.1023 theta 0.792]
- preregistro: regla binding de hip-15 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"alpha": 0.1, "config": "ours1", "per_seed": [{"seed": 0, "theta": 0.7974319458007812, "automation": 0.5227272727272727, "realized_risk": 0.10191304347826091, "bound": 0.09832145312894992}, {"seed": 1, "theta": 0.8302433490753174, "automation": 0.4330909090909091, "realized_risk": 0.08522250209907645, "bound": 0.09842366837099242}, {"seed": 2, "theta": 0.7920957803726196, "automation": 0.5243636363636364, "realized_risk": 0.10228848821081826, "bound": 0.09633731393887968}]}
- origen: scripts/gates
- fecha: 2026-10-12
### NODO int-19
- tipo: intento
- estado: sostenido
- texto: R2-F corrido: zona A [acc HTTP memoria 0.824 vs escalera 0.824 (benchmark 0.809); p50 215 vs 214 ms; errores 0]
- preregistro: regla binding de hip-16 en el libro de gates (spec §7), fijada antes de correr
- evidencia-refutacion: {"acc_http_memory": 0.8244274809160306, "acc_ladder_ours1": 0.8244274809160306, "acc_http_benchmark": 0.8091603053435115, "p50_memory_s": 0.2151924967765808, "p50_benchmark_s": 0.21362016536295414, "errors": 0, "memory_hits": 131}
- origen: scripts/gates
- fecha: 2026-10-12
<!-- exploracion:nodos:end -->

## Aristas
<!-- exploracion:aristas:start -->
- int-1 --sostiene--> hip-1
- int-3 --sostiene--> hip-1
- int-4 --refuta--> hip-2
- int-5 --refuta--> hip-3
- int-6 --sostiene--> hip-4
- int-7 --sostiene--> hip-5
- int-8 --refuta--> hip-6
- int-9 --refuta--> hip-7
- int-10 --refuta--> hip-8
- int-11 --sostiene--> hip-9
- int-12 --sostiene--> hip-10
- int-13 --refuta--> hip-4
- hip-11 --deriva-de--> hip-2
- hip-12 --deriva-de--> hip-2
- hip-13 --deriva-de--> hip-7
- hip-14 --deriva-de--> hip-4
- hip-15 --deriva-de--> hip-3
- hip-16 --deriva-de--> hip-8
- int-14 --refuta--> hip-11
- int-15 --sostiene--> hip-12
- int-16 --refuta--> hip-13
- int-17 --refuta--> hip-14
- int-18 --refuta--> hip-15
- int-19 --sostiene--> hip-16
<!-- exploracion:aristas:end -->

## Huecos
_(ninguno todavía)_
