#!/usr/bin/env bash
# Grafo de exploración (ronda 10): CONTRATO y tooling determinista para
# .graph/exploracion.md — la bitácora de hipótesis/intentos/anomalías/
# referencias-SOTA que alimenta el filtro de no-refutación (spec
# 2026-09-06-graph-exploracion-dirigida-ronda-10-design.md, §3-5.bis).
# bash + python3 stdlib, cero dependencias externas, mismo estilo que
# tools/oraculo-map.sh y tools/red.sh.
#
# =============================================================================
# CONTRATO de .graph/exploracion.md (vinculante — cualquier escritor de este
# archivo, humano o agente, debe respetar esta gramática; este tool es la
# única vía de escritura recomendada, pero el formato en sí es lo que hace
# al archivo "parseable" independientemente de quién lo edite)
# =============================================================================
#
# El archivo es markdown con DOS secciones estructuradas delimitadas por
# marcadores HTML-comment (mismo patrón que symbol-map:start/end en map.md),
# más prosa libre alrededor que este tool NUNCA parsea ni toca:
#
#   # Grafo de exploración · <proyecto>
#   > Esquema-exploracion: 1 · Sembrado: <fecha> · Última tarea: <slug>
#
#   ## Resumen
#   <prosa libre — informativa, este tool no la lee ni la escribe>
#
#   ## Nodos
#   <!-- exploracion:nodos:start -->
#   ### NODO <id>
#   - tipo: hipótesis|intento|anomalía|referencia-sota
#   - estado: vivo|refutado|sostenido
#   - texto: <una línea>
#   - preregistro: <una línea, puede ir vacía>
#   - evidencia-refutacion: <una línea, puede ir vacía>
#   - origen: <una línea>
#   - fecha: <YYYY-MM-DD>
#   (solo si tipo=anomalía, además:)
#   - esperado: <una línea>
#   - observado: <una línea>
#   - bifurcacion: pendiente|a|b
#   (0 o más bloques ### NODO seguidos, en cualquier orden)
#   <!-- exploracion:nodos:end -->
#
#   ## Aristas
#   <!-- exploracion:aristas:start -->
#   - <id-origen> --refuta--> <id-destino>
#   - <id-origen> --variante-de--> <id-destino>
#   - <id-origen> --deriva-de--> <id-destino>
#   - <id-origen> --sostiene--> <id-destino>
#   (0 o más líneas así)
#   <!-- exploracion:aristas:end -->
#
#   ## Huecos
#   <prosa libre>
#
# Reglas de la gramática (lo que `validar` exige, fail-closed en cualquier
# desvío):
#   - Los 4 marcadores (nodos:start/end, aristas:start/end) deben existir
#     EXACTAMENTE una vez cada uno, en pares bien anidados (start < end
#     dentro de cada par) y las dos secciones NUNCA se superponen entre sí
#     (Nodos y Aristas pueden ir en cualquier orden relativo, pero no
#     intercaladas). Sin este esqueleto, NINGÚN modo opera sobre el archivo
#     (ver Fase 4.6 de skills/init/SKILL.md: lo siembra init/refresh).
#   - Cada campo es UNA sola línea (`- campo: valor`); un valor con saltos
#     de línea embebidos rompería la gramática, así que `agregar` sanea
#     (reemplaza TAB/CR/LF por espacio) ANTES de escribir — mismo choke
#     point que sanitize_motivo en oraculo-map.sh.
#   - Campos comunes obligatorios en TODO nodo: tipo, estado, texto,
#     preregistro, evidencia-refutacion, origen, fecha. `texto` y `origen`
#     no pueden ir vacíos. `fecha` debe matchear YYYY-MM-DD.
#   - `preregistro` no puede ir vacío si tipo ∈ {hipótesis, intento} (spec
#     §5: sin compromiso previo no hay refutación que cuente). Puede ir
#     vacío en anomalía (que ya declara su propia expectativa en
#     `esperado`) y en referencia-sota (un hecho registrado, no una
#     apuesta falsable).
#   - `evidencia-refutacion` no puede ir vacía si estado=refutado (si no
#     hubo evidencia, no está refutado — es la disciplina de preregistro
#     del spec §5 aplicada a la ESCRITURA, no solo a la intención).
#   - tipo=anomalía exige además `esperado`, `observado` (ambos no vacíos,
#     spec §5.bis: los dos campos obligatorios del disparador de
#     anomalía) y `bifurcacion` ∈ {pendiente, a, b} (spec §5.bis: (a)
#     error nuestro / (b) expectativa equivocada — `pendiente` mientras el
#     humano no decidió). Ningún otro tipo admite estos 3 campos (fail-
#     closed ante copy-paste entre tipos).
#   - El PREFIJO del id debe corresponder al tipo declarado (ver mapa
#     tipo→prefijo abajo) — detecta un id manualmente editado que quedó
#     desincronizado de su propio campo `tipo`.
#   - IDs únicos en TODO el archivo (no solo dentro de un tipo).
#   - Toda arista (`- <origen> --<tipo>--> <destino>`) exige que <origen> Y
#     <destino> existan como ids de nodo ya parseados (integridad
#     referencial) y prohíbe origen==destino (auto-referencia).
#   - Ningún campo desconocido (typo de nombre de campo) se tolera en
#     silencio: fail-closed con diag.
#
# ID: `<prefijo>-<n>` con <n> entero positivo secuencial POR PREFIJO:
#   hipótesis        -> hip-<n>
#   intento          -> int-<n>
#   anomalía         -> anom-<n>
#   referencia-sota  -> sota-<n>
# `agregar` asigna `<n> = max(ids EXISTENTES de ese prefijo en el archivo,
# en ESTE momento) + 1` — determinista dado el estado actual del archivo,
# nunca aleatorio ni basado en reloj, y NUNCA se renumera un nodo
# existente (un id ya escrito no cambia mientras el nodo exista).
#
# Corrección deliberada de contrato (verificación adversarial r10, hallazgo
# #4): una versión anterior de este comentario prometía que un id "nunca
# se reutiliza", lo cual el algoritmo de arriba NO garantiza — escanea los
# ids PRESENTES en el archivo en ese instante, no un contador aparte. Si un
# nodo se borra a mano (el tool no ofrece "borrar"; solo un editor humano
# podría hacerlo) y luego se agrega uno nuevo del mismo prefijo, el hueco
# numérico puede reaparecer con un id igual al del nodo borrado, ahora
# apuntando a un nodo distinto. Se decidió declarar esta implicancia en vez
# de agregar un contador persistente en el encabezado del archivo: un
# contador aparte necesitaría su propia escritura atómica coordinada con la
# del nodo (mismo lock de abajo, pero un punto más de estado que puede
# quedar desincronizado del contenido real si algo edita el archivo a mano
# sin pasar por este tool — el contrato ya exige que CUALQUIER escritor,
# humano o agente, sea válido) y una superficie de fallo nueva en un tool
# que la ronda 10 deja apagado por default. Mientras nadie borre nodos a
# mano, el comportamiento observado SÍ es "nunca se reutiliza"; la garantía
# formal es más débil que eso, y queda documentada acá para que quien lea
# el contrato no confíe en algo que el código no cumple.
#
# Re-normalización: LÍMITE CONOCIDO, no arreglado esta ronda (verificación
# adversarial r10, hallazgo #6 — declarado en vez de resuelto porque es un
# problema de DISEÑO del contrato, no un bug puntual de este tool). La idea
# original (mismo principio que INDEX.md/map.md/oraculo.md, C5 de
# conventions.md) era: si `## Nodos` crece más de ~1 pantalla, quien lo
# edite mueve los bloques `### NODO ...` ÍNTEGROS a un archivo enlazado
# nuevo (p.ej. `.graph/exploracion-nodos.md`, con la MISMA gramática de
# marcadores nodos:start/end) dejando en el raíz un resumen + puntero.
#
# Pero este tool valida y opera SIEMPRE sobre UN solo <archivo> por
# invocación (`parse_file` no sigue punteros ni conoce la existencia de un
# enlazado) — así que partir un grafo en dos archivos ROMPE la integridad
# referencial de las aristas apenas un nodo cruza de archivo mientras una
# arista que lo referencia se queda del otro lado: `validar` sobre
# CUALQUIERA de los dos archivos por separado va a dar rc=1 ("arista
# referencia un id inexistente") para toda arista cuyo origen o destino
# haya migrado al otro archivo. No hay "re-normalización sin pérdida" real
# disponible con el tooling actual: mover nodos a un enlazado sin además
# mover (o duplicar) toda arista que los toca dejará el grafo dividido en
# un estado que este tool reporta como inválido. Consecuencia práctica:
# NO mover bloques `### NODO` a un archivo enlazado mientras tengan
# aristas hacia/desde nodos que se quedan en el otro archivo — o aceptar
# que `validar` quede en rc=1 para ese archivo hasta que el diseño del
# contrato lo resuelva (fuera de alcance de esta ronda).
#
# =============================================================================
# Modos
# =============================================================================
#   exploracion.sh validar <archivo>
#     Formato + integridad referencial de aristas + ids únicos. rc=0 si el
#     archivo cumple TODO el contrato de arriba; rc=1 con una línea `diag:`
#     POR hallazgo (puede haber varias) si no. Archivo con 0 nodos y 0
#     aristas (el esqueleto recién sembrado) es VÁLIDO (rc=0).
#
#   exploracion.sh repite <archivo> <descripcion>
#     ¿<descripcion> repite un nodo YA REFUTADO? Salida: UNA línea TSV a
#     stdout — `veredicto<TAB>id<TAB>motivo` — veredicto ∈ {REPETIDO,
#     NUEVO}; `id` es el nodo refutado más parecido (vacío si NUEVO).
#     rc=0 en ambos veredictos (rc≠0 es solo error de USO o de FORMATO del
#     archivo — el veredicto en sí nunca es un fallo del comando). Nunca
#     escribe nada.
#
#     ¿Qué cuenta como "refutado" a los fines de esta comparación? La
#     UNIÓN de dos conjuntos (regla (2) agregada tras hallazgo de
#     verificación adversarial r10 — antes solo miraba (1) y por eso no
#     veía lo que el propio sistema acababa de refutar):
#       (1) todo nodo cuyo propio campo `estado` valga `refutado`.
#       (2) todo nodo que sea DESTINO de una arista `--refuta-->` cuyo
#           ORIGEN tenga campo `estado`=`refutado`. El patrón que el
#           propio contrato prescribe (spec §5) es registrar un `intento`
#           con `--refuta <id-hipótesis>`: eso deja el `intento` en
#           refutado pero NUNCA toca el `estado` de la hipótesis
#           apuntada, que sigue `vivo` — sin la regla (2) una propuesta
#           calcada de esa hipótesis pasaría como NUEVO pese a estar ya
#           descartada.
#     La regla (2) mira el campo `estado` REAL del origen, NUNCA un nodo
#     que sea "refutado" solo por la regla (2) misma — un solo salto, sin
#     cascada transitiva no pedida por el spec (p.ej. si X refuta a Y por
#     esta regla y algo `--refuta--> X`, eso NO arrastra a Y).
#
#     Heurística (declarada, conservadora, SIN embeddings ni deps): se
#     compara <descripcion> contra el campo `texto` de cada nodo refutado
#     (unión (1)+(2) de arriba; nunca contra un nodo vivo/sostenido que no
#     sea destino de tal arista — "repite" responde EXACTAMENTE la
#     pregunta del spec §3: ¿esto ya se descartó?). Ambos
#     textos se normalizan (NFKD + se descartan marcas combinantes, o sea
#     se "pliegan" acentos: café→cafe) y se tokenizan en palabras
#     [a-z0-9]+ de largo≥3 que no estén en una lista fija de stopwords en
#     español (de/la/el/en/un/que/para/con/... — ver STOPWORDS). Score =
#     Jaccard sobre esos conjuntos de tokens (|intersección|/|unión|).
#     Veredicto REPETIDO solo si Jaccard≥0.40 Y ≥3 tokens compartidos
#     (ambos umbrales a la vez — evita marcar por 1-2 palabras técnicas
#     incidentales en textos cortos); el nodo refutado con MAYOR score
#     gana, empate se rompe por menor id (orden prefijo,número — nunca
#     por orden de aparición en el archivo, para que el resultado no
#     dependa de cómo quedó escrito el archivo).
#
#     Límites CONOCIDOS y declarados (no se ocultan):
#       - Bag-of-words puro, sin semántica: una paráfrasis con vocabulario
#         TOTALMENTE distinto de la misma idea NO se detecta (sesgo hacia
#         el falso negativo en ESE caso — cuando el vocabulario no
#         coincide, conservador a propósito: preferible dejar pasar una
#         propuesta legítima que bloquear una nueva por error).
#       - PERO en el caso contrario (vocabulario técnico compartido, IDEA
#         opuesta o no relacionada) el umbral produce FALSOS POSITIVOS
#         reales, no hipotéticos — esto es una limitación real del
#         heurístico, no "conservador": Jaccard≥0.40 Y ≥3 tokens
#         compartidos no distingue "aumentar X" de "reducir X" si ambos
#         textos comparten el resto del vocabulario técnico. Ejemplo: nodo
#         refutado con texto "aumentar el batch size en el entrenamiento
#         del modelo" (tokens: aumentar/batch/size/entrenamiento/modelo)
#         vs. propuesta NUEVA y legítima "reducir el batch size en el
#         entrenamiento del modelo base" (tokens: reducir/batch/size/
#         entrenamiento/modelo/base) → compartidos={batch,size,
#         entrenamiento,modelo}=4, unión=7, Jaccard=0.57≥0.40 y 4≥3 →
#         REPETIDO, pese a que "reducir" es la propuesta opuesta a
#         "aumentar" y nunca fue refutada. Mitigación posible pero NO
#         implementada (agregaría superficie/deps): comparar también el
#         verbo/operación distinguido del resto del vocabulario técnico.
#         Hasta entonces: un veredicto REPETIDO es una señal a revisar por
#         un humano, no un descarte automático — el `motivo` siempre
#         incluye el id y una vista previa del nodo citado para que ese
#         chequeo sea barato.
#       - Sin stemming: "compartida" y "compartido" NO cuentan como la
#         misma palabra (reduce el score, nunca lo infla).
#       - Plegado de acentos puede colisionar palabras distintas por
#         accidente ortográfico (ej. "año"/"ano" quedan iguales tras
#         plegar) — riesgo real, no eliminado, mitigado por exigir
#         Jaccard alto Y ≥3 tokens (una sola colisión así no alcanza).
#       - Lista de stopwords es de español; texto en otro idioma no se
#         filtra (puede sub- o sobre-contar tokens significativos).
#       - Solo mira `texto`, nunca `preregistro`/`evidencia-refutacion`
#         (esos comparten vocabulario de método entre nodos sin relación,
#         lo que infla falsos positivos si se incluyeran).
#       - Compara contra TODOS los refutados sin límite de escala — en un
#         grafo con miles de nodos esto es O(n) por invocación; aceptable
#         al tamaño esperado de este archivo. NO asumir que partir el
#         archivo en un enlazado resuelve esto sin más: ver el límite
#         conocido de re-normalización declarado más arriba (romper el
#         archivo en dos puede volverlo inválido antes de aliviar la
#         escala).
#
#   exploracion.sh agregar <archivo> <tipo> <texto> [flags...]
#     Agrega UN nodo nuevo (append determinista: inserta el bloque nuevo
#     inmediatamente antes de `<!-- exploracion:nodos:end -->`, sin tocar
#     ni un byte de ningún bloque `### NODO` existente — "jamás pisa
#     nodos ajenos"). Si el archivo no cumple el contrato (mismas
#     validaciones que `validar`), o si algún flag es inválido, o si un id
#     referenciado por --refuta/--variante-de/--deriva-de/--sostiene no
#     existe: rc=1 con diag, el archivo NO se toca (todo o nada). Éxito:
#     imprime el id nuevo asignado (una línea) a stdout, rc=0.
#
#     <tipo> acepta con o sin tilde: hipotesis|hipótesis, intento,
#     anomalia|anomalía, referencia-sota (también el alias "sota").
#
#     Flags (todos opcionales salvo donde el tipo/estado los exige, ver
#     arriba): --refuta ID, --variante-de ID, --deriva-de ID, --sostiene ID
#     (0 o más — cada uno crea una arista `<id-nuevo> --tipo--> ID`; el
#     MISMO flag puede repetirse varias veces — `--refuta hip-1 --refuta
#     hip-2` crea DOS aristas, una por ocurrencia, en el orden en que se
#     pasaron — corregido en r10 tras hallazgo adversarial #3: una versión
#     anterior solo conservaba la ÚLTIMA ocurrencia de cada flag y
#     descartaba las anteriores en silencio, sin diag);
#     --preregistro TXT (alias --prerregistro, por si se invoca con la
#     ortografía castellana estricta del prefijo "pre-"+"registro");
#     --estado vivo|refutado|sostenido (default: vivo); --evidencia TXT;
#     --origen TXT (default: "manual" si se omite); --esperado TXT,
#     --observado TXT, --bifurcacion pendiente|a|b (solo válidos, y
#     esperado/observado obligatorios, con tipo=anomalía). `fecha` la
#     asigna el propio comando (`date -u +%Y-%m-%d`, no es un flag).
#
#   exploracion.sh resumen <archivo>
#     Conteos por tipo/estado y por tipo de arista, para el record (diario
#     de tarea). TSV determinista a stdout, TODAS las combinaciones
#     siempre presentes (aunque valgan 0, para que el resumen sea
#     diffable entre corridas):
#       nodo<TAB><tipo><TAB><estado><TAB><n>      (4 tipos × 3 estados)
#       arista<TAB><tipo-arista><TAB><n>          (4 tipos de arista)
#       total<TAB>nodos<TAB><n>
#       total<TAB>aristas<TAB><n>
#     rc=1 con diag si el archivo no cumple el contrato (mismo criterio
#     que `validar`).
#
# Fail-closed real: los 4 modos comparten `run_py` (ver abajo), que
# distingue "el intérprete embebido corrió y encontró un problema de
# negocio real" (relee su propio diag, ya impreso por python, y no agrega
# nada encima) de "python3 no corrió en absoluto o reventó sin avisar"
# (stderr vacío pese a rc≠0 → se sintetiza un diag genérico) — así
# `validar`/`resumen` con un archivo inválido emiten SOLO los diagnósticos
# específicos y accionables, sin ruido, mientras que una rotura interna
# real sigue siendo fail-closed con aviso.
#
# Locale y determinismo: mismo patrón que oraculo-map.sh/red.sh — el
# script fija LC_ALL=C y fuerza a python3 a decodificar argv/stdin/stdout
# como UTF-8 (PYTHONUTF8, PYTHONIOENCODING), así que dos corridas
# idénticas no divergen según el locale del invocador (incluye ids/textos
# con acentos, como en los tests con paths y descripciones acentuadas).
#
# Fin de línea: el archivo se ESCRIBE siempre LF-only (`write_atomic` junta
# las líneas con "\n"; ningún modo de este tool emite "\r"). Al LEER, sin
# embargo, `read_text` normaliza CRLF/CR-solo a LF ANTES de partir en
# líneas (hallazgo adversarial r10 #5: sin esto, un archivo CRLF —p.ej.
# editado en Windows— dejaba un "\r" colgando al final de cada línea, y
# como los regexes de campo/nodo/arista no lo toleraban, CADA línea fallaba
# el parseo con diagnósticos genéricos de "línea no reconocida" en vez de
# un único diag claro sobre el fin de línea — fail-closed técnicamente
# correcto pero indepurable). Con la normalización, un archivo CRLF válido
# por lo demás valida rc=0 igual que su equivalente LF; `sanitize_field`
# sigue reemplazando cualquier "\r" que sobreviva DENTRO de un valor de
# campo (pegado a mano, no de fin de línea) por un espacio, como red de
# seguridad adicional al escribir.
set -euo pipefail

# Determinismo independiente del invocador (idéntico a oraculo-map.sh).
export LC_ALL=C
export PYTHONUTF8=1
export PYTHONIOENCODING='utf-8:surrogateescape'

DONE=false
TMP_ROOT=""
LOCK_DIR_HELD=""

on_exit() {
  local rc=$?
  release_lock
  if [ -n "$TMP_ROOT" ]; then
    rm -rf "$TMP_ROOT" 2>/dev/null || true
  fi
  if [ "$DONE" != true ]; then
    echo "diag: error interno inesperado (rc=$rc) — fail-closed" >&2
    exit 1
  fi
}
trap on_exit EXIT

fail_closed() {
  echo "diag: $1" >&2
  DONE=true
  exit 1
}

TMP_ROOT="$(mktemp -d)"

# --- lock de archivo (serializa el ciclo read-modify-write de `agregar`) ---
# Hallazgo adversarial r10 #2: `agregar` es parse_file → calcular id nuevo
# → write_atomic, TODO dentro de UN proceso python. `write_atomic` es
# atómico en el rename final, pero eso NO serializa el ciclo COMPLETO entre
# dos invocaciones DISTINTAS de este script sobre el MISMO archivo: dos
# `agregar` concurrentes pueden ambos leer el mismo estado viejo, calcular
# el MISMO id nuevo, y escribir cada uno un `write_atomic` que reemplaza
# por completo al del otro — ambos terminan con rc=0 e imprimen un id,
# pero el archivo final solo tiene el nodo del que escribió último: el
# otro nodo (y su id impreso al llamador) desaparece en silencio. Se
# serializa el ciclo completo con un lock exclusivo por archivo ANTES de
# invocar `run_py agregar`, usando `flock(1)` si está disponible (namespace
# de lock separado del archivo de datos, vía un descriptor sobre
# "<archivo>.lock", liberado automáticamente al cerrar el fd = al salir
# este proceso) y, si no, un fallback 100% portable sin deps: un directorio
# "<archivo>.lockdir" creado con `mkdir` (atómico en cualquier POSIX, a
# diferencia de comprobar-y-crear un archivo con `[ -e ]` + `touch`) con
# espera activa acotada. Solo `agregar` toma este lock: los demás modos son
# de solo lectura y `write_atomic` (rename) garantiza que SIEMPRE ven o el
# archivo viejo completo o el nuevo completo, nunca un estado a medio
# escribir.
release_lock() {
  if [ -n "$LOCK_DIR_HELD" ]; then
    rmdir "$LOCK_DIR_HELD" 2>/dev/null || true
    LOCK_DIR_HELD=""
  fi
}

acquire_lock() {
  local archivo="$1"
  if command -v flock >/dev/null 2>&1; then
    local lockfile="${archivo}.lock"
    exec {EXPLORACION_LOCKFD}>"$lockfile" \
      || fail_closed "agregar: no se pudo abrir '$lockfile' para el lock"
    flock -x "$EXPLORACION_LOCKFD" \
      || fail_closed "agregar: no se pudo adquirir el lock de '$archivo'"
    return 0
  fi
  # Fallback portable (cero deps): mkdir es atómico — exactamente UN
  # invocador concurrente puede crear el directorio; el resto reintenta.
  local lockdir="${archivo}.lockdir"
  local waited=0
  while ! mkdir "$lockdir" 2>/dev/null; do
    sleep 0.05
    waited=$((waited + 1))
    if [ "$waited" -ge 1200 ]; then
      fail_closed "agregar: no se pudo adquirir el lock de '$archivo' (mkdir '$lockdir') tras ~60s — otro proceso lo retiene o quedó huérfano"
    fi
  done
  LOCK_DIR_HELD="$lockdir"
}

# run_py <modo> <args...>: ejecuta el intérprete embebido capturando
# stdout/stderr por separado. Relee stdout tal cual (éxito o no — algunos
# modos igual escriben algo útil antes de fallar, aunque hoy ninguno lo
# hace). Ante rc≠0: si python3 YA imprimió algo a stderr (el caso normal —
# hallazgos de validación, un flag inválido, etc.), lo repite tal cual SIN
# agregar nada encima (evita el "diag" genérico duplicado que ensuciaría
# el caso más común: un archivo simplemente inválido). Si stderr vino
# VACÍO pese a rc≠0 (python3 no llegó a correr — binario roto/reemplazado,
# mismo escenario que el caso (j)/(aa) de test-oraculo-map.sh), sintetiza
# un diag genérico: fail-closed nunca en silencio.
run_py() {
  local out err rc=0
  out="$(mktemp -p "$TMP_ROOT")"
  err="$(mktemp -p "$TMP_ROOT")"
  if python3 -c "$PY_EXPLORACION" "$@" >"$out" 2>"$err"; then
    rc=0
  else
    rc=$?
  fi
  cat "$out"
  if [ "$rc" -ne 0 ]; then
    if [ -s "$err" ]; then
      cat "$err" >&2
    else
      echo "diag: error interno inesperado en exploracion.sh (rc=$rc) — fail-closed" >&2
    fi
  fi
  return "$rc"
}

# Script python embebido (stdlib: re, sys, os, tempfile, unicodedata). Un
# solo cuerpo para los 4 modos: comparten `parse_file` (la ÚNICA función
# que lee y valida la gramática) así que no hay una segunda heurística de
# "qué es válido" que pueda divergir entre `validar` y lo que `agregar`
# exige antes de escribir (mismo principio que scan≡fuerza en
# oraculo-map.sh).
read -r -d '' PY_EXPLORACION <<'PYEOF' || true
import os
import re
import sys
import tempfile
import unicodedata

TIPOS = ("hipótesis", "intento", "anomalía", "referencia-sota")
PREFIX_OF_TIPO = {
    "hipótesis": "hip",
    "intento": "int",
    "anomalía": "anom",
    "referencia-sota": "sota",
}
ESTADOS = ("vivo", "refutado", "sostenido")
BIFURCACIONES = ("pendiente", "a", "b")
EDGE_TIPOS = ("refuta", "variante-de", "deriva-de", "sostiene")

COMMON_FIELDS = ("tipo", "estado", "texto", "preregistro", "evidencia-refutacion", "origen", "fecha")
ANOM_FIELDS = ("esperado", "observado", "bifurcacion")

MARK_NODOS_START = "<!-- exploracion:nodos:start -->"
MARK_NODOS_END = "<!-- exploracion:nodos:end -->"
MARK_ARISTAS_START = "<!-- exploracion:aristas:start -->"
MARK_ARISTAS_END = "<!-- exploracion:aristas:end -->"

DATE_RE = re.compile(r'^[0-9]{4}-[0-9]{2}-[0-9]{2}$')
NODE_HEADER_RE = re.compile(r'^### NODO ([a-z]+-[0-9]+)\s*$')
FIELD_RE = re.compile(r'^- ([a-zA-Z_-]+):(?: (.*))?$')
EDGE_RE = re.compile(r'^- (\S+) --(refuta|variante-de|deriva-de|sostiene)--> (\S+)\s*$')

TIPO_ALIASES = {
    "hipotesis": "hipótesis",
    "intento": "intento",
    "anomalia": "anomalía",
    "referencia-sota": "referencia-sota",
    "referencia_sota": "referencia-sota",
    "sota": "referencia-sota",
}

STOPWORDS = set("""
de la el en un una uno unos unas que para con por los las se su sus del al
y o no es son ser esta esto este estos estas como mas menos sin sobre
entre cuando donde porque pero si ya muy tambien hay han fue era eran sea
sean lo le les nos mi tu eso etc todo toda todos todas the a an of to
""".split())

TOKEN_RE = re.compile(r'[a-z0-9]+')

JACCARD_THRESHOLD = 0.40
MIN_SHARED = 3


def sanitize_field(value):
    """Choke point único de saneo: un valor de campo viaja como UNA línea
    `- campo: valor` — un TAB o salto de línea adentro rompería la
    gramática (mismo hallazgo r9 #7 de oraculo-map.sh, aplicado acá)."""
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def fold(text):
    """Pliega acentos/diacríticos (NFKD + descarta marcas combinantes):
    'café' -> 'cafe', 'año' -> 'ano'. Usado para matching tolerante a
    acentos en `<tipo>` y en `repite` — con el riesgo declarado en el
    encabezado (colisión ortográfica ocasional)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def tokenize(text):
    folded = fold(text).lower()
    raw = TOKEN_RE.findall(folded)
    return [t for t in raw if len(t) >= 3 and t not in STOPWORDS]


def id_sort_key(node_id):
    prefix, _, num = node_id.rpartition("-")
    try:
        n = int(num)
    except ValueError:
        n = 0
    return (prefix, n)


def read_text(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    text = raw.decode("utf-8", errors="replace")
    # Tolera CRLF/CR-solo al LEER (hallazgo adversarial r10 #5): el archivo
    # se ESCRIBE siempre LF-only, pero un editor externo (o alguien en
    # Windows) puede dejarlo en CRLF. Sin esto, cada línea terminaba con un
    # "\r" colgante que los regexes de campo/nodo/arista (anclados con "$")
    # no toleraban, y el archivo fallaba línea por línea con diagnósticos
    # genéricos de "línea no reconocida" en vez de un fallo claro y único.
    # Normalizar ANTES de partir en líneas hace que un archivo CRLF válido
    # por lo demás valide igual que su equivalente LF.
    return text.replace("\r\n", "\n").replace("\r", "\n")


class ParseResult(object):
    def __init__(self):
        self.errors = []
        self.nodes = []
        self.node_by_id = {}
        self.edges = []
        self.lines = []


def parse_file(path):
    pr = ParseResult()
    try:
        text = read_text(path)
    except OSError as exc:
        pr.errors.append("no se pudo leer '%s': %s" % (path, exc))
        return pr

    lines = text.split("\n")
    pr.lines = lines

    def find_all(marker):
        return [i for i, l in enumerate(lines) if l.strip() == marker]

    idx = {}
    for name in (MARK_NODOS_START, MARK_NODOS_END, MARK_ARISTAS_START, MARK_ARISTAS_END):
        found = find_all(name)
        if len(found) == 0:
            pr.errors.append("falta el marcador '%s' (el archivo no tiene el esqueleto esperado — ver contrato en el encabezado de exploracion.sh)" % name)
        elif len(found) > 1:
            pr.errors.append("marcador duplicado '%s' (aparece %d veces, debe aparecer exactamente una)" % (name, len(found)))
        else:
            idx[name] = found[0]

    if pr.errors:
        return pr

    ns, ne = idx[MARK_NODOS_START], idx[MARK_NODOS_END]
    as_, ae = idx[MARK_ARISTAS_START], idx[MARK_ARISTAS_END]

    if ns >= ne:
        pr.errors.append("marcadores de Nodos invertidos (start en línea %d, end en línea %d)" % (ns + 1, ne + 1))
    if as_ >= ae:
        pr.errors.append("marcadores de Aristas invertidos (start en línea %d, end en línea %d)" % (as_ + 1, ae + 1))
    if pr.errors:
        return pr

    nodos_span = (min(ns, ne), max(ns, ne))
    aristas_span = (min(as_, ae), max(as_, ae))
    if nodos_span[0] <= aristas_span[1] and aristas_span[0] <= nodos_span[1]:
        pr.errors.append("las secciones de Nodos y Aristas se superponen o están mal anidadas")
        return pr

    seen_ids = {}
    i = ns + 1
    current = None
    while i < ne:
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue
        m = NODE_HEADER_RE.match(line)
        if m:
            if current is not None:
                _finalize_node(pr, current, seen_ids)
            current = {"id": m.group(1), "fields": {}, "line_no": i + 1}
            i += 1
            continue
        fm = FIELD_RE.match(line)
        if fm and current is not None:
            key, val = fm.group(1), (fm.group(2) or "")
            if key in current["fields"]:
                pr.errors.append("campo repetido '%s' en nodo %s (línea %d)" % (key, current["id"], i + 1))
            current["fields"][key] = val
            i += 1
            continue
        if current is None:
            pr.errors.append("línea no reconocida dentro de la sección de Nodos (línea %d): %r" % (i + 1, line))
        else:
            pr.errors.append("línea no reconocida dentro del nodo %s (línea %d): %r" % (current["id"], i + 1, line))
        i += 1
    if current is not None:
        _finalize_node(pr, current, seen_ids)

    i = as_ + 1
    while i < ae:
        line = lines[i]
        if line.strip() == "":
            i += 1
            continue
        em = EDGE_RE.match(line)
        if not em:
            pr.errors.append("línea no reconocida dentro de la sección de Aristas (línea %d): %r" % (i + 1, line))
            i += 1
            continue
        origen, tipo_arista, destino = em.group(1), em.group(2), em.group(3)
        pr.edges.append({"origen": origen, "tipo": tipo_arista, "destino": destino, "line_no": i + 1})
        i += 1

    for e in pr.edges:
        if e["origen"] == e["destino"]:
            pr.errors.append("arista con origen=destino ('%s', línea %d): auto-referencia no permitida" % (e["origen"], e["line_no"]))
        if e["origen"] not in pr.node_by_id:
            pr.errors.append("arista referencia un id inexistente como origen: '%s' (línea %d)" % (e["origen"], e["line_no"]))
        if e["destino"] not in pr.node_by_id:
            pr.errors.append("arista referencia un id inexistente como destino: '%s' (línea %d)" % (e["destino"], e["line_no"]))

    return pr


def _finalize_node(pr, current, seen_ids):
    nid = current["id"]
    fields = current["fields"]
    prefix = nid.split("-", 1)[0]

    if nid in seen_ids:
        pr.errors.append("id duplicado: '%s' (primera vez línea %d, de nuevo línea %d)" % (nid, seen_ids[nid], current["line_no"]))
    else:
        seen_ids[nid] = current["line_no"]

    tipo = fields.get("tipo")
    missing_common = [k for k in COMMON_FIELDS if k not in fields]
    if missing_common:
        pr.errors.append("nodo %s: falta(n) campo(s) obligatorio(s): %s" % (nid, ", ".join(missing_common)))

    if tipo is not None and tipo not in TIPOS:
        pr.errors.append("nodo %s: tipo desconocido '%s' (válidos: %s)" % (nid, tipo, "|".join(TIPOS)))
    elif tipo is not None:
        expected_prefix = PREFIX_OF_TIPO[tipo]
        if prefix != expected_prefix:
            pr.errors.append("nodo %s: el prefijo del id no corresponde al tipo '%s' (esperado prefijo '%s-')" % (nid, tipo, expected_prefix))

    estado = fields.get("estado")
    if estado is not None and estado not in ESTADOS:
        pr.errors.append("nodo %s: estado desconocido '%s' (válidos: %s)" % (nid, estado, "|".join(ESTADOS)))

    if not (fields.get("texto") or "").strip():
        pr.errors.append("nodo %s: campo 'texto' vacío (obligatorio)" % nid)
    if not (fields.get("origen") or "").strip():
        pr.errors.append("nodo %s: campo 'origen' vacío (obligatorio)" % nid)

    fecha = fields.get("fecha")
    if fecha is not None and not DATE_RE.match(fecha):
        pr.errors.append("nodo %s: fecha con formato inválido '%s' (esperado YYYY-MM-DD)" % (nid, fecha))

    if tipo in ("hipótesis", "intento") and not (fields.get("preregistro") or "").strip():
        pr.errors.append("nodo %s: tipo '%s' exige 'preregistro' no vacío (spec §5)" % (nid, tipo))

    if estado == "refutado" and not (fields.get("evidencia-refutacion") or "").strip():
        pr.errors.append("nodo %s: estado 'refutado' exige 'evidencia-refutacion' no vacía" % nid)

    is_anom = (tipo == "anomalía")
    present_anom_fields = [k for k in ANOM_FIELDS if k in fields]
    if is_anom:
        missing_anom = [k for k in ANOM_FIELDS if k not in fields]
        if missing_anom:
            pr.errors.append("nodo %s: tipo 'anomalía' exige campo(s) %s (spec §5.bis)" % (nid, ", ".join(missing_anom)))
        if not (fields.get("esperado") or "").strip():
            pr.errors.append("nodo %s: campo 'esperado' vacío (obligatorio en anomalía)" % nid)
        if not (fields.get("observado") or "").strip():
            pr.errors.append("nodo %s: campo 'observado' vacío (obligatorio en anomalía)" % nid)
        bif = fields.get("bifurcacion")
        if bif is not None and bif not in BIFURCACIONES:
            pr.errors.append("nodo %s: bifurcacion inválida '%s' (válidas: %s)" % (nid, bif, "|".join(BIFURCACIONES)))
    elif present_anom_fields:
        pr.errors.append("nodo %s: tipo '%s' no admite campo(s) %s (exclusivos de anomalía)" % (nid, tipo, ", ".join(present_anom_fields)))

    allowed = set(COMMON_FIELDS) | (set(ANOM_FIELDS) if is_anom else set())
    unknown = sorted(k for k in fields if k not in allowed)
    if unknown:
        pr.errors.append("nodo %s: campo(s) desconocido(s): %s" % (nid, ", ".join(unknown)))

    record = dict(fields)
    record["id"] = nid
    pr.nodes.append(record)
    if nid not in pr.node_by_id:
        pr.node_by_id[nid] = record


def insert_before_marker(lines, marker, new_lines):
    idx = None
    for i, l in enumerate(lines):
        if l.strip() == marker:
            idx = i
            break
    if idx is None:
        raise RuntimeError("marcador no encontrado al insertar: %s" % marker)
    return lines[:idx] + new_lines + lines[idx:]


def write_atomic(path, text):
    d = os.path.dirname(os.path.abspath(path)) or "."
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".exploracion-", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(text.encode("utf-8"))
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def emit_errors(errors):
    for e in errors:
        sys.stderr.write("diag: %s\n" % e)


def mode_validar(argv):
    (path,) = argv
    pr = parse_file(path)
    if pr.errors:
        emit_errors(pr.errors)
        return 1
    return 0


def mode_resumen(argv):
    (path,) = argv
    pr = parse_file(path)
    if pr.errors:
        emit_errors(pr.errors)
        return 1

    counts = {}
    for n in pr.nodes:
        key = (n.get("tipo"), n.get("estado"))
        counts[key] = counts.get(key, 0) + 1

    edge_counts = {}
    for e in pr.edges:
        edge_counts[e["tipo"]] = edge_counts.get(e["tipo"], 0) + 1

    out = []
    for t in TIPOS:
        for e in ESTADOS:
            out.append("nodo\t%s\t%s\t%d" % (t, e, counts.get((t, e), 0)))
    for t in EDGE_TIPOS:
        out.append("arista\t%s\t%d" % (t, edge_counts.get(t, 0)))
    out.append("total\tnodos\t%d" % len(pr.nodes))
    out.append("total\taristas\t%d" % len(pr.edges))
    sys.stdout.write("\n".join(out) + "\n")
    return 0


def mode_repite(argv):
    path, descripcion = argv
    pr = parse_file(path)
    if pr.errors:
        emit_errors(pr.errors)
        return 1

    desc_tokens = set(tokenize(descripcion))

    # "Refutado" para `repite` = unión de (1) estado=refutado real y (2)
    # destino de una arista --refuta--> cuyo origen tenga estado=refutado
    # REAL (ver regla documentada en el encabezado del tool, hallazgo
    # adversarial r10 #1). Un solo salto: `refutados_reales` es la
    # instantánea fija contra la que se prueba cada origen de arista, así
    # que agregar destinos a `refutado_ids` en este bucle nunca hace que
    # ese destino, a su vez, "refute por arista" a un tercero.
    refutados_reales = set(n["id"] for n in pr.nodes if n.get("estado") == "refutado")
    refutado_ids = set(refutados_reales)
    for e in pr.edges:
        if e["tipo"] == "refuta" and e["origen"] in refutados_reales:
            refutado_ids.add(e["destino"])

    refutados = [pr.node_by_id[nid] for nid in refutado_ids]

    best_id = None
    best_score = 0.0
    best_shared = 0
    for n in sorted(refutados, key=lambda n: id_sort_key(n["id"])):
        node_tokens = set(tokenize(n.get("texto") or ""))
        if not desc_tokens or not node_tokens:
            continue
        shared = len(desc_tokens & node_tokens)
        union = len(desc_tokens | node_tokens)
        score = (float(shared) / union) if union else 0.0
        if score > best_score:
            best_score = score
            best_shared = shared
            best_id = n["id"]

    def preview(texto, maxwords=8):
        words = texto.split()
        p = " ".join(words[:maxwords])
        if len(words) > maxwords:
            p += "…"
        return sanitize_field(p)

    if best_id is not None and best_score >= JACCARD_THRESHOLD and best_shared >= MIN_SHARED:
        node_texto = pr.node_by_id[best_id].get("texto", "")
        motivo = sanitize_field(
            "solapamiento lexico jaccard=%.2f (%d palabras significativas compartidas) con nodo refutado %s: \"%s\""
            % (best_score, best_shared, best_id, preview(node_texto))
        )
        sys.stdout.write("REPETIDO\t%s\t%s\n" % (best_id, motivo))
    else:
        motivo = sanitize_field(
            "sin coincidencia significativa con nodos refutados (mejor jaccard=%.2f con %s, umbral=%.2f, minimo_compartidas=%d)"
            % (best_score, (best_id or "ninguno"), JACCARD_THRESHOLD, MIN_SHARED)
        )
        sys.stdout.write("NUEVO\t\t%s\n" % motivo)
    return 0


EDGE_FLAG_TIPOS = ("refuta", "variante-de", "deriva-de", "sostiene")


def mode_agregar(argv):
    (path, tipo_cli, texto, edge_arg,
     preregistro, estado_cli, evidencia, origen, esperado, observado,
     bifurcacion_cli, fecha) = argv

    pr = parse_file(path)
    if pr.errors:
        emit_errors(pr.errors)
        sys.stderr.write("diag: agregar se detiene sin escribir (el archivo no cumple el contrato)\n")
        return 1

    errors = []

    tipo = TIPO_ALIASES.get(fold(tipo_cli).lower())
    if tipo is None:
        errors.append("tipo desconocido '%s' (válidos: hipotesis|hipótesis, intento, anomalia|anomalía, referencia-sota|sota)" % tipo_cli)

    if not texto.strip():
        errors.append("texto vacío (obligatorio)")

    estado = estado_cli or "vivo"
    if estado not in ESTADOS:
        errors.append("estado desconocido '%s' (válidos: %s)" % (estado, "|".join(ESTADOS)))

    if estado == "refutado" and not evidencia.strip():
        errors.append("estado 'refutado' exige --evidencia no vacía")

    if tipo in ("hipótesis", "intento") and not preregistro.strip():
        errors.append("tipo '%s' exige --preregistro (o --prerregistro) no vacío (spec §5)" % tipo)

    origen_final = origen.strip() or "manual"

    is_anom = (tipo == "anomalía")
    bifurcacion = None
    if is_anom:
        if not esperado.strip():
            errors.append("tipo 'anomalía' exige --esperado no vacío (spec §5.bis)")
        if not observado.strip():
            errors.append("tipo 'anomalía' exige --observado no vacío (spec §5.bis)")
        bifurcacion = bifurcacion_cli or "pendiente"
        if bifurcacion not in BIFURCACIONES:
            errors.append("bifurcacion inválida '%s' (válidas: %s)" % (bifurcacion, "|".join(BIFURCACIONES)))
    elif esperado.strip() or observado.strip() or bifurcacion_cli.strip():
        errors.append("--esperado/--observado/--bifurcacion solo son válidos con tipo anomalía")

    # `edge_arg` trae TODAS las ocurrencias de TODOS los flags de arista,
    # una por línea lógica separada por "\x1e" (record separator), cada una
    # como "<tipo>\x1f<id-destino>" (unit separator) — ver el bash que
    # arma esto en `agregar_mode`. Se preserva UNA tupla por ocurrencia
    # (hallazgo adversarial r10 #3: antes solo llegaba el ÚLTIMO valor de
    # cada flag, así que `--refuta hip-1 --refuta hip-2` perdía la primera
    # arista en silencio) — el orden de las aristas creadas respeta el
    # orden en que se pasaron los flags.
    edge_requests = []
    if edge_arg:
        for entry in edge_arg.split("\x1e"):
            if not entry:
                continue
            tipo_arista, sep, target = entry.partition("\x1f")
            flagname = "--%s" % tipo_arista
            if not sep or tipo_arista not in EDGE_FLAG_TIPOS:
                errors.append("arista interna con formato inválido (bug de invocación): %r" % (entry,))
                continue
            if not target:
                errors.append("%s requiere un id (bug de invocación: llegó vacío)" % flagname)
            elif target not in pr.node_by_id:
                errors.append("%s referencia un id inexistente: '%s'" % (flagname, target))
            else:
                edge_requests.append((tipo_arista, target))

    if not DATE_RE.match(fecha):
        errors.append("fecha interna con formato inválido '%s' (bug de invocación, no del usuario)" % fecha)

    if errors:
        emit_errors(errors)
        return 1

    prefix = PREFIX_OF_TIPO[tipo]
    existing_nums = []
    for n in pr.nodes:
        p, _, num = n["id"].partition("-")
        if p == prefix:
            try:
                existing_nums.append(int(num))
            except ValueError:
                pass
    new_num = (max(existing_nums) + 1) if existing_nums else 1
    new_id = "%s-%d" % (prefix, new_num)

    node_lines = ["### NODO %s" % new_id]
    node_lines.append("- tipo: %s" % tipo)
    node_lines.append("- estado: %s" % estado)
    node_lines.append("- texto: %s" % sanitize_field(texto))
    node_lines.append("- preregistro: %s" % sanitize_field(preregistro))
    node_lines.append("- evidencia-refutacion: %s" % sanitize_field(evidencia))
    node_lines.append("- origen: %s" % sanitize_field(origen_final))
    node_lines.append("- fecha: %s" % fecha)
    if is_anom:
        node_lines.append("- esperado: %s" % sanitize_field(esperado))
        node_lines.append("- observado: %s" % sanitize_field(observado))
        node_lines.append("- bifurcacion: %s" % bifurcacion)

    edge_lines = ["- %s --%s--> %s" % (new_id, tipo_arista, target) for tipo_arista, target in edge_requests]

    lines = list(pr.lines)
    lines = insert_before_marker(lines, MARK_NODOS_END, node_lines)
    if edge_lines:
        lines = insert_before_marker(lines, MARK_ARISTAS_END, edge_lines)

    write_atomic(path, "\n".join(lines))

    sys.stdout.write("%s\n" % new_id)
    return 0


def main():
    mode = sys.argv[1]
    argv = sys.argv[2:]
    if mode == "validar":
        rc = mode_validar(argv)
    elif mode == "repite":
        rc = mode_repite(argv)
    elif mode == "agregar":
        rc = mode_agregar(argv)
    elif mode == "resumen":
        rc = mode_resumen(argv)
    else:
        sys.stderr.write("diag: modo interno desconocido para PY_EXPLORACION: %r\n" % mode)
        rc = 1
    sys.exit(rc)


try:
    main()
except Exception as exc:  # fail-closed: nunca a medio escribir, siempre diag
    sys.stderr.write("diag: error interno en exploracion.sh (python): %s\n" % exc)
    sys.exit(1)
PYEOF

# validar <archivo> ----------------------------------------------------
validar_mode() {
  local archivo="${1:-}"
  [ -n "$archivo" ] || fail_closed "validar: falta <archivo>"
  [ -f "$archivo" ] || fail_closed "validar: '$archivo' no existe o no es un archivo regular"
  local rc=0
  run_py validar "$archivo" || rc=$?
  DONE=true
  exit "$rc"
}

# repite <archivo> <descripcion> ----------------------------------------
repite_mode() {
  local archivo="${1:-}" descripcion="${2:-}"
  [ -n "$archivo" ] || fail_closed "repite: falta <archivo>"
  [ -n "$descripcion" ] || fail_closed "repite: falta <descripcion>"
  [ -f "$archivo" ] || fail_closed "repite: '$archivo' no existe o no es un archivo regular"
  local rc=0
  run_py repite "$archivo" "$descripcion" || rc=$?
  DONE=true
  exit "$rc"
}

# resumen <archivo> ------------------------------------------------------
resumen_mode() {
  local archivo="${1:-}"
  [ -n "$archivo" ] || fail_closed "resumen: falta <archivo>"
  [ -f "$archivo" ] || fail_closed "resumen: '$archivo' no existe o no es un archivo regular"
  local rc=0
  run_py resumen "$archivo" || rc=$?
  DONE=true
  exit "$rc"
}

# agregar <archivo> <tipo> <texto> [flags...] ----------------------------
agregar_mode() {
  local archivo="${1:-}" tipo_cli="${2:-}" texto="${3:-}"
  [ -n "$archivo" ] || fail_closed "agregar: falta <archivo>"
  [ -n "$tipo_cli" ] || fail_closed "agregar: falta <tipo>"
  [ -n "$texto" ] || fail_closed "agregar: falta <texto>"
  [ -f "$archivo" ] || fail_closed "agregar: '$archivo' no existe o no es un archivo regular"
  shift 3 || true

  local preregistro="" estado="" evidencia="" origen="" esperado="" \
        observado="" bifurcacion=""
  # EDGE_REQS acumula CADA ocurrencia de un flag de arista (hallazgo
  # adversarial r10 #3: antes se guardaba en una variable escalar por
  # flag, así que una segunda ocurrencia del MISMO flag pisaba a la
  # primera en silencio). Cada elemento es "<tipo>\x1f<id-destino>"; se
  # codifica así (separador ASCII "unit separator", nunca aparece en texto
  # normal) para poder pasar la lista COMPLETA como UN solo argv al
  # intérprete embebido sin cambiar la forma de invocar `run_py` a "N
  # argumentos variables".
  local -a EDGE_REQS=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --refuta)
        [ $# -ge 2 ] || fail_closed "agregar: --refuta requiere un id"
        EDGE_REQS+=("refuta"$'\x1f'"$2"); shift 2 ;;
      --variante-de)
        [ $# -ge 2 ] || fail_closed "agregar: --variante-de requiere un id"
        EDGE_REQS+=("variante-de"$'\x1f'"$2"); shift 2 ;;
      --deriva-de)
        [ $# -ge 2 ] || fail_closed "agregar: --deriva-de requiere un id"
        EDGE_REQS+=("deriva-de"$'\x1f'"$2"); shift 2 ;;
      --sostiene)
        [ $# -ge 2 ] || fail_closed "agregar: --sostiene requiere un id"
        EDGE_REQS+=("sostiene"$'\x1f'"$2"); shift 2 ;;
      --preregistro|--prerregistro)
        [ $# -ge 2 ] || fail_closed "agregar: --preregistro requiere texto"
        preregistro="$2"; shift 2 ;;
      --estado)
        [ $# -ge 2 ] || fail_closed "agregar: --estado requiere un valor"
        estado="$2"; shift 2 ;;
      --evidencia)
        [ $# -ge 2 ] || fail_closed "agregar: --evidencia requiere texto"
        evidencia="$2"; shift 2 ;;
      --origen)
        [ $# -ge 2 ] || fail_closed "agregar: --origen requiere texto"
        origen="$2"; shift 2 ;;
      --esperado)
        [ $# -ge 2 ] || fail_closed "agregar: --esperado requiere texto"
        esperado="$2"; shift 2 ;;
      --observado)
        [ $# -ge 2 ] || fail_closed "agregar: --observado requiere texto"
        observado="$2"; shift 2 ;;
      --bifurcacion)
        [ $# -ge 2 ] || fail_closed "agregar: --bifurcacion requiere un valor"
        bifurcacion="$2"; shift 2 ;;
      *)
        fail_closed "agregar: flag desconocida '$1'" ;;
    esac
  done

  local fecha
  fecha="$(date -u +%Y-%m-%d)"

  # Une EDGE_REQS en un solo string, entradas separadas por "\x1e" (ASCII
  # "record separator") — junto con "\x1f" de arriba, ninguno de los dos
  # puede aparecer en un id de nodo (que matchea `[a-z]+-[0-9]+`) ni en el
  # nombre fijo de un tipo de arista, así que la codificación es inambigua.
  local edge_arg=""
  if [ "${#EDGE_REQS[@]}" -gt 0 ]; then
    edge_arg="$(printf '%s\x1e' "${EDGE_REQS[@]}")"
    edge_arg="${edge_arg%$'\x1e'}"
  fi

  # Lock de archivo: serializa el ciclo read-modify-write completo (parse +
  # asignación de id + write_atomic, todo dentro de `run_py agregar`) frente
  # a otras invocaciones concurrentes de `agregar` sobre el MISMO archivo
  # (ver comentario junto a `acquire_lock` arriba — hallazgo adversarial
  # r10 #2). Se toma acá, lo más tarde posible: después de validar TODOS
  # los argumentos de uso (arriba), para no bloquear ni competir por el
  # lock ante un error que de todos modos no va a escribir nada.
  acquire_lock "$archivo"

  local rc=0
  run_py agregar "$archivo" "$tipo_cli" "$texto" "$edge_arg" \
    "$preregistro" "$estado" "$evidencia" "$origen" \
    "$esperado" "$observado" "$bifurcacion" "$fecha" || rc=$?
  DONE=true
  exit "$rc"
}

# --- dispatch ---------------------------------------------------------------
USO="uso: exploracion.sh validar <archivo> | repite <archivo> <descripcion> | agregar <archivo> <tipo> <texto> [--refuta id] [--variante-de id] [--deriva-de id] [--sostiene id] [--preregistro txt] [--estado vivo|refutado|sostenido] [--evidencia txt] [--origen txt] [--esperado txt] [--observado txt] [--bifurcacion pendiente|a|b] | resumen <archivo>"

[ $# -ge 1 ] || fail_closed "$USO"
MODE="$1"
shift
case "$MODE" in
  validar)
    [ $# -eq 1 ] || fail_closed "uso: exploracion.sh validar <archivo> (recibí $# argumentos)"
    validar_mode "$1"
    ;;
  repite)
    [ $# -eq 2 ] || fail_closed "uso: exploracion.sh repite <archivo> <descripcion> (recibí $# argumentos)"
    repite_mode "$1" "$2"
    ;;
  agregar)
    [ $# -ge 3 ] || fail_closed "uso: exploracion.sh agregar <archivo> <tipo> <texto> [flags...] (recibí $# argumentos)"
    agregar_mode "$@"
    ;;
  resumen)
    [ $# -eq 1 ] || fail_closed "uso: exploracion.sh resumen <archivo> (recibí $# argumentos)"
    resumen_mode "$1"
    ;;
  *)
    fail_closed "modo desconocido '$MODE' (usar 'validar', 'repite', 'agregar' o 'resumen')"
    ;;
esac
