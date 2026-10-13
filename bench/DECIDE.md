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

## 6. Backends de terceros: qué se verificó y qué no (búsqueda web del 13 oct 2026, sin descargar nada)
Todas las cifras de esta sección son **[reportado por terceros, sin verificar]** salvo lo marcado.
- **Laya (multilingüe, 322M)** (`convaiinnovations/laya-multilingual`): licencia **Apache-2.0** (campo de licencia de la página del modelo), 322M parámetros (backbone mmBERT-base de 307M + cabeza de decisión), contexto de **1024 tokens** por defecto (hasta 8192 pasando `max_len`; la página reporta 16-18/20 aciertos hasta ~4000 tokens y 8-17/20 por encima), 51 idiomas. Latencia reportada: **32.8 ms con 1 pregunta**, 72 ms con 10 (hardware no indicado; no hay cifra de CPU). Calibración reportada: **sin calibrar de fábrica** (ECE macro 0.387; recalibrando temperaturas con datos propios bajó a 0.106). Se instala con `pip install laya` (dependencias de torch/transformers: **no es un extra opcional pequeño**). **No encontré un ONNX int8 de la versión multilingüe**: el único ONNX con FP32/FP16/INT8 que apareció (`tozp/laya-onnx`) es de la versión inglesa de 421M, de un autor distinto al oficial. Por tanto "ONNX int8 en CPU, ~33-40 ms" **no está confirmado para el modelo multilingüe**.
- **NanoJev** (Qwen3-0.6B + cabezas de decisión, **MIT** según su página): checkpoint, conjunto de datos y servicio de inferencia local. Sus benchmarks propios son de juegos (Maze, Snake...) con conjuntos de 8-10 episodios, "direccionales"; **no hay evidencia sobre tareas como las nuestras**. Necesita torch + transformers para ejecutarlo; nuestro motor NPU/GPU no carga ese formato.
- **Kev** (modelos de decisión de Jared Palmer sobre bases Qwen3.5, LoRA de rango 16 + cabeza "pointer"): tamaños **0.8B, 4B, 9B** (y 27B según una fuente; mi prompt original citaba solo 0.8B y 4B), licencia **Apache-2.0** según páginas de empaquetado de terceros (no vi la del repositorio original). Latencia reportada: Kev-4B 779 ms para 5 preguntas en un Apple M5 (otra fuente da 721 ms) y 18.1 ms en una H100; autores y reseñas, sin pruebas independientes. Para usarlo aquí haría falta fusionar el LoRA con la base, convertir a nuestro formato int8 e implementar la cabeza "pointer": trabajo de ingeniería no trivial; **no se ha intentado**.
- **Jev** (TypeSafe AI): propietario, solo API, sin pesos; descartado por la regla 9. (`juspay/jev-one` de Hugging Face es otro modelo no relacionado.)

**Estado:** ninguno de los tres se descargó ni se instaló. Antes de hacerlo necesito tu confirmación (regla 5): Laya y NanoJev añaden torch/transformers/laya fuera del núcleo (extra opcional o entorno aparte), y Kev exige la integración descrita. Mi recomendación se da en la sección de conclusiones según los resultados de la línea base.

## 7. Resultados de la comparativa (conjunto de prueba, 13 oct 2026) [medido, GPU degradada]
Tablas completas, barridos de τ y estudio de longitud en `bench/decide_results/REPORT.md` (generado por `bench/decide_report.py`). Intervalos Wilson 95 %. Latencias medidas con la GPU degradada (`bench/ENERGY.md`).

| Backend | done: AUC / exactitud | route: exactitud | mixer: exactitud | Brier route / mixer | ECE route / mixer | Latencia p50 | RAM/VRAM extra | Instalación | Licencia |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
| `logprob` Qwen3.5-4B | 0.912 / 0.747 [0.64, 0.83] | **0.900** [0.77, 0.96] | **0.925** [0.80, 0.97] | 0.140 / 0.101 | 0.056 / 0.094 | 2.9 s (done), 1.8 s (mixer) | 0 (usa el modelo cargado) | ninguna | Apache-2.0 (Qwen) |
| `logprob` Qwen3.5-2B | 0.925 / 0.800 [0.70, 0.87] | 0.250 [0.14, 0.40] (siempre "None") | 0.450 [0.31, 0.60] | 1.324 / 0.805 | 0.689 / 0.384 | 0.8 s | 0 | ninguna | Apache-2.0 |
| `logprob` 2B, etiquetas A-D (bandera `NPU_LAB_ROUTE_LABELS=letters`) | — | 0.525 [0.37, 0.67] | — | 0.645 | 0.184 | 0.8 s | 0 | ninguna | Apache-2.0 |
| `embed` (MiniLM multilingüe + regresión logística) | 0.705 / 0.667 [0.55, 0.76] | 0.850 [0.71, 0.93] | 0.850 [0.71, 0.93] | 0.341 / 0.306 | 0.310 / 0.229 | 0.05-0.22 s | ~790 MB RAM (CPU) | ninguna (numpy, scipy, tokenizers ya del núcleo) | Apache-2.0 (MiniLM) [no verificada aquí] |
| Laya, NanoJev, Kev | no medidos: no se descargaron ni instalaron (§6) | | | | | | | | |

**Aplicando la regla de adopción fijada en §0:**
- `done`: `logprob` (4B o 2B) supera a `embed` en +0.21 AUC (margen +0.05): **gana `logprob`**.
- `route`: con el 4B cargado, `embed` queda 5.0 puntos por debajo (0.85 frente a 0.90); `mixer`, 7.5 puntos (0.85 frente a 0.925). Los intervalos se solapan. `embed` no supera a `logprob` en ninguno, así que **no se adopta**; es una diferencia en zona gris, no una victoria estadística.
- Con solo el 2B cargado la misma regla da la vuelta: `embed` supera a `logprob` por +0.60 (route) y +0.40 (mixer). Es una **condición**, no el valor por defecto: el 2B no sirve para decidir con palabras (§7.1).
- Estudio de longitud (`done`, 0/25/80 líneas de ruido): el AUC de `logprob` 4B pasa de 0.912 a 0.941 y 0.944 (sin degradación hasta ~2900 caracteres); `embed` cae a 0.521 porque su contexto (256 tokens) corta la evidencia. `logprob` mantiene falsos aceptados = 0/12 al τ de validación en los tres largos.

### 7.1 El 2B en `route`: matriz de confusión y sonda [medido]
- `bench/route_confusion.py`: el 2B contesta **None a las 40 preguntas** (probabilidad media 0.94 para NONE; el 4B es casi diagonal: 9/10, 10/10, 10/10, 7/10).
- Sonda `--probe` (12 preguntas de val/test): prompt actual 3/12 (todo None); sin el bloque `<think>` vacío 5/12; letras A-D 8/12. En las 40 preguntas de prueba con letras: 21/40 = 0.525 [0.37, 0.67]. Conclusión: es un sesgo del modelo hacia la palabra "None" como primera palabra, no un error de plantilla; las letras lo reducen sin eliminarlo. No se enciende por defecto y el 2B no se usa para decidir.
- Desviación del protocolo: la primera ejecución usaba etiquetas partidas en varios tokens (`DOCS` = `DO`+`CS`); se corrigió a etiquetas de un token antes de ver ningún resultado de prueba del 4B. El archivo de esa corrida se conserva (`bench/decide_results/logprob_qwen35_2b.firsttoken_bug.jsonl.txt`) y daba el mismo 0.25.

## 8. El guard de `DONE` con trazas reales del 4B [medido, n=1 por tarea, exploración]
Trazas: `bench/agent_traces/` (no se suben), 25 tareas × 1 muestra, 4B, `--run`, guard apagado. La unidad estadística es la tarea; intervalos por bootstrap sobre tareas; `bench/guard_offline.py` reproduce todo desde las trazas.
- Resueltas **22/25**. **`DONE` falso: 2/25 = 0.08 [0.00, 0.20]** (tareas 11 y 18, ambas sin tests propios). Una tarea (4, borrar `legacy.py`) terminó sin `DONE`: 0/8 acciones válidas, nunca usó `DELETE` y los mensajes de error no lo mencionan. El flujo de confirmación y borrado no llegó a ejercitarse.
- **Guard objetivo:** atrapa 0 de los 2 `DONE` falsos (compilan y no tienen tests) y rechaza 3 `DONE` correctos de la fixture con tests (21, 22, 23: "tests no corridos tras el último cambio"). Ejecución real posterior (L3b): en las 3 el agente ejecutó `RUN` y terminó bien; **el guard cuesta un turno, no aciertos** (válidas 3/4, resueltas 3/3).
- **Guard de modelo (4B):** p(done) de los 2 erróneos: 0.19 y 0.08; de los 22 correctos: mediana 0.29, rango 0.09-0.72; AUC 0.89 (2 negativos: ruido). **τ=0.678, calibrado sobre los datos sintéticos de validación, NO sirve con trazas reales** (acepta 1 de 22 correctos): el 4B subestima los correctos en estas trazas respecto a los estados sintéticos. A τ=0.30 rechaza los 2 erróneos y acepta 10 de 22 correctos. Con 1-2 negativos por pliegue no se puede recalibrar leave-one-task-out: **no se recomienda ningún τ**; el guard de modelo se reporta como curva.
- **L3a [humo, τ=0.35 elegido sobre los datos de evaluación con 2 negativos]:** tareas 11 y 18 con guard de modelo: rechazó los 3 `DONE` de cada una y el agente no se recuperó (11: siguió reescribiendo `format_line` en lugar de añadir `table`; 18: repitió la misma edición inválida, y un `DONE` tras una acción fallida se rechaza sin más). Resultado: 0 `DONE` falsos pero 0 resueltas, y 419 s y 360 s frente a ~100 s. El guard convierte un `DONE` falso en un fracaso honesto y más caro; no arregla la tarea.
- **Determinismo [medido]:** decodificación voraz; en las 7 corridas de L3 las respuestas coinciden con las de L1 hasta la primera intervención del guard.
- Nota de condiciones: el `platform_profile` se cambió a `performance` a las 09:39, a mitad de la muestra 0; no se comparan latencias entre tramos.

## 9. Decisiones de la Fase 4
- **Backend por defecto: `logprob` con el 4B** para `done`, `route` y `mixer`. `embed` queda documentado como zona gris (no adoptado) y como alternativa si solo hay un 2B cargado. El 2B no decide con palabras.
- **Guard de `DONE`:** `agent_guard` apagado por defecto. El guard objetivo manda cuando hay check; el modelo es señal secundaria, apagada, y se reporta como curva de τ (sin umbral recomendado).
- **Laya, NanoJev, Kev: no se descargan ni instalan** (§6, [reportado por terceros, sin verificar]).
- `RUN` solo ejecuta el comando configurado, tras confirmación, con la salida visible.

## 10. Propuestas, sin construir
1. **Endurecer el guard objetivo con comprobaciones que no necesitan tests** (los 2 `DONE` falsos están en tareas sin tests): diff no vacío (ya está), archivos tocados coherentes con la instrucción (p. ej. la tarea nombra `report.py`/`format_line` y el diff solo cambia una función existente, sin definir lo pedido), y que lo que la instrucción nombra por identificador (`table`, `cur`) aparezca en el diff.
2. **Mejorar los errores del agente para el borrado:** cuando un bloque de edición tiene SEARCH y REPLACE vacíos, o intenta crear un archivo que ya existe, sugerir `DELETE path` (tarea 4).
3. **Calibración del guard de modelo con trazas reales del 4B** cuando haya >=10 negativos (n>=3 muestras de las tareas sin tests); mientras tanto no hay umbral recomendado.
4. Si el agente reintenta tras un rechazo del modelo, dar contexto del rechazo (qué probabilidad, qué faltaría) en vez de un mensaje genérico.
