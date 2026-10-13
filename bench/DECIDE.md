# Capa de decisión local (Fase 4)

Etiquetas: **[medido]**, **[leído en código]**, **[estimado]**, **[propuesto]**, **[reportado por terceros, sin verificar]**.

## 0. Protocolo fijado ANTES de medir (13 oct 2026, commit anterior a cualquier resultado)

**Qué se decide** (`src/decide.py`, interfaz `decide(estado, preguntas, backend) -> probabilidades`, una pregunta por llamada):
- `done`: ¿la evidencia (tarea, diff, compilación, salida de pruebas) muestra que la tarea está terminada? YES/NO. Guard de `DONE` del agente.
- `route`: fuente de la pregunta: NONE / DOCS / WEB / HISTORY.
- `mixer`: modelo que debe responder: fast / medium / slow.

**Datos** (`bench/decide_data/*.jsonl`, generados por `bench/decide_data_done.py` y `bench/decide_data_text.py`):
- `done` (210 estados): finales de las 25 tareas de `bench/agent_tasks.py` (referencia, nada hecho, parciales, mutantes de una línea, cambios inocuos). **La etiqueta es el resultado real del check de la tarea**, no una opinión. Sintético: salen de mutaciones, **no de trazas del 4B real** [limitación].
- `route` (203) y `mixer` (201): frases ES/EN **etiquetadas a mano por mí** [limitación: las etiquetas son mi criterio; la frontera fast/medium/slow es subjetiva].
- Separación **por tarea** (done) y estratificada por hash del texto (route, mixer): 60 / 20 / 20 % entrenamiento / validación / prueba, fija. Las 8 tareas de prueba de `done` (5, 8, 12, 16, 20, 23, 24, 25) también son las reservadas para cualquier ajuste fino posterior.
- Aviso: "≥200 por tarea" se cumple en total, pero el conjunto de prueba es pequeño (done: 75 estados, 12 negativos que pasan las comprobaciones objetivas; route: 40; mixer: 40). Los intervalos serán anchos y se reportan (Wilson 95 %).

**Backends** de esta tanda: `logprob` (línea base; Qwen3.5-2B y 4B ya cargados, sin muestrear) y `embed` (MiniLM multilingüe `models/multi` + regresión logística en numpy/scipy, sin dependencias nuevas). `laya`, `nanojev`, `kev`: se verifican fuentes y licencias (sección 6) y no se descarga nada sin tu confirmación.

**Métricas por backend** (conjunto de prueba): exactitud; Brier; ECE (10 cajas); para `done` además AUC, y **falsos aceptados** (estados NO aceptados) al umbral τ elegido en validación; latencia p50/p95 por decisión; RAM/VRAM extra; tamaño de instalación; licencia.
- Umbral de `done`: τ = el menor valor para el que NO hay falsos aceptados entre los negativos de validación que pasan las comprobaciones objetivas (si hay empate, el máximo p(YES) entre ellos + 0.001). Se reporta la tasa de falsos aceptados en prueba con intervalo.
- `embed` se entrena solo con la partición de entrenamiento; `logprob` no se entrena.

**Regla de adopción (fijada ahora):** un backend distinto de `logprob` **gana** en una tarea solo si, en el conjunto de prueba, mejora la métrica principal en el margen: **exactitud +5 puntos** (route, mixer) o **AUC +0.05** (done). **Empate** = diferencia dentro de 2 puntos de exactitud / 0.02 de AUC: en empate gana el que use menos VRAM+RAM extra; si también empatan, el que no sobrescribe la caché KV del modelo principal. Entre 2 y 5 puntos no se adopta. Si ningún backend supera el margen ni empata con menos memoria, se propone un ajuste LoRA pequeño (con aprobación y coste estimado).

**Guard de `DONE`** (`decide.done_guard`): (1) comprobaciones objetivas primero (cambió algo, compila, las pruebas del usuario pasan si están configuradas); (2) la probabilidad del backend elegido como señal secundaria con τ calibrado en validación; (3) nunca el veredicto del mismo modelo como única prueba.
- El estudio de longitud mide la exactitud de `done` con la evidencia ampliada con 0, ~25 y ~80 líneas de salida irrelevante.

**Condición de la máquina:** la GPU está degradada hoy (`bench/ENERGY.md`, 2.5 TFLOPS), así que las **latencias** de `logprob` serán más altas que en una GPU sana; la exactitud no depende de ello.
