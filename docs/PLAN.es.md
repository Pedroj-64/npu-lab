# Plan: un asistente siempre activo para la NPU de primera generación

Versión en inglés: [`PLAN.md`](PLAN.md). Reglas de trabajo: [`WORKING_RULES.es.md`](WORKING_RULES.es.md). Marcas: `[ ]` pendiente, `[~]` en curso, `[x]` hecho. Cada tarea tiene una comprobación de aceptación; no la marques como hecha sin ejecutarla.

## 1. Dirección
La NPU XDNA1 (Ryzen 7040/8040, `npu1`) no puede ganarle a una GPU en velocidad: el decode está limitado por DMA a unos 22 GB/s y rinde 9–20 tok/s con modelos de 0.6–2B (`bench/PERF.md`, `bench/DMA_READ.md`). Su ventaja real es el **bajo consumo y estar siempre disponible sin tocar la GPU**. El producto es entonces un asistente de modelos pequeños que trabaja en segundo plano, más un cliente de escritorio y una página web que lo hagan usable para quien tenga esta generación olvidada de hardware.

Decisiones (dueño, 2026-10-14): abrir el proyecto de verdad (la licencia ya es MIT), primer pilar = asistente siempre activo, cliente de escritorio con Tauri, medición continua, búsqueda continua en web y GitHub de nuevos usos de la NPU y documentación bilingüe completa para el traspaso.

## 2. Qué existe y qué no
| Existe | Falta |
|---|---|
| Qwen3 0.6B/1.7B solo en NPU; Qwen3.5 0.8B–4B asistido | Voz (ASR/VAD), visión, reranker medido |
| Embeddings MiniLM en NPU (coseno ≥ 0.996, ~20 trozos/s) | Acceso remoto / celular |
| Clasificador `decide.py` (ruta 0.85), `router.py` | Cliente de escritorio nativo |
| `search.py`, `documents.py`, `docs.py`, `web.py`, `repomap.py` (solo Python) | Integración con graphify |
| Puente `/v1` compatible con OpenAI, agente con confirmación, `bench/energy.py` | Ciclos continuos de medición e investigación |

## 3. Usos, por prioridad
| # | Uso | Por qué la NPU | Esfuerzo |
|---|---|---|---|
| 1 | Búsqueda semántica local (archivos, docs, notas) en segundo plano | Embeddings baratos y constantes, la GPU queda libre | Bajo: existe, falta chunking y reranker |
| 2 | Mapa de proyecto con graphify (botón) + consultas compactas para otros agentes | La extracción AST es gratis; la NPU resume archivos con un modelo de 0.6–1.7B | Bajo–medio |
| 3 | Centinela: vigila repos, archivos y trabajos, clasifica y avisa; un modelo pequeño responde | Siempre encendido a bajo consumo | Medio |
| 4 | Puente al celular (mensaje de entrada, el PC responde) | El PC de casa es el servidor | Medio |
| 5 | Ayudas diarias de programación: mensaje de commit, resumen de diff, triage de logs, explicar un comando, revisión local de secretos en un diff | 0.6–2B bastan | Bajo |
| 6 | Voz local: dictado (Whisper tiny/base) y comando de activación | Capas densas como MiniLM; requiere kernels nuevos | Alto, incierto |
| – | Videojuegos | No hay API de NPU para juegos en Linux; XDNA1 es lenta. Diferido | – |
| – | Modelado / renderizado | La NPU no sirve para eso; solo aplican los mapas de proyectos grandes (#2) | – |

Descartado de la lista original (usos de NPU de teléfono): desbloqueo facial 3D, fotografía de retrato y reducción de ruido, predicción de apps y batería (el sistema operativo ya lo hace y XDNA1 no es accesible desde esas pilas).

## 4. Panorama externo (consultado el 2026-10-14, ver `RESEARCH_LOG.md`)
FastFlowLM, Lemonade y AMD IRON apuntan a **XDNA2** (Ryzen AI 300/400); la guía de Lemonade para Linux dice que XDNA1 no está soportada. Hay trabajo comunitario para `npu1` pero es de un solo autor y sin verificar (`Scottcjn/npu-linux-kit`, `Scottcjn/open-xdna`, `mrtib/phoenix-npu1-iron-kernels`). Ese hueco es la oportunidad: este proyecto ya tiene kernels propios para `npu1`.

## 5. Tareas

### Fase 0 — Línea base y limpieza
- [ ] **T0.1 Línea base.** Ejecutar las pruebas sin conexión y `bench/energy.py --engine npu --model models/qwen3s` (3 corridas) con la NPU sola y el cargador enchufado antes de arrancar (ver `ROADMAP.md` §2, problema P5 de la GPU). *Aceptación:* una fila nueva en `bench/history.jsonl` y un párrafo en `bench/ENERGY.md`.
- [ ] **T0.2 Primera ejecución de `tests/eval_answers.py`** (escrito, nunca ejecutado). *Aceptación:* resultados guardados en `bench/` y resumidos en `docs/CHANGELOG.md`.
- [ ] **T0.3 Revisar el trabajo sin commitear del puente en `src/serve.py`** (`/v1` con herramientas) con `python tests/test_server.py`. El commit lo hace el dueño.

### Fase 1 — Producto base y ciclos continuos
- [~] **T1.1 Medición continua.** `bench/watch.py` y `bench/history.jsonl` existen como esqueleto. Falta: ejecutarlo dos veces y confirmar que añade dos líneas; simular una regresión (`--simulate-drop 0.2`) y confirmar el código de salida 2; añadir el timer semanal de usuario `~/.config/systemd/user/npu-bench.timer`; añadir a la interfaz un botón «medir» y la gráfica de tendencia. *Aceptación:* las cuatro comprobaciones.
- [~] **T1.2 Investigación continua.** `tools/research_scan.py` existe como esqueleto. Falta: ejecutarlo dos veces y confirmar que la segunda no añade nada; confirmar que solo salen palabras clave; añadir el timer semanal; clasificar los hallazgos con el modelo de la NPU (relevante / no, una línea). *Aceptación:* esas comprobaciones.
- [ ] **T1.3 Valores por defecto solo NPU.** Qwen3-0.6B/1.7B (`src/qwen_npu.py`) como perfil por defecto; el perfil Asistido (Qwen3.5 + CUDA) queda opcional. *Aceptación:* una instalación nueva conversa sin NVIDIA.
- [ ] **T1.4 Botón graphify.** `src/graph.py` ejecuta `graphify <ruta> --update` como subproceso; `POST /project/graph` y `GET /project/graph/query?q=…&budget=N` en `src/serve.py`, tras las comprobaciones de token y Host/Origin existentes; `repomap.py` queda de respaldo. *Aceptación:* en un repo de prueba, `--update` no hace llamadas de red para código, y `graphify query --budget 1500` gasta menos tokens que enviar los archivos (medir ambos).
- [ ] **T1.5 Calidad de la búsqueda semántica** (`ROADMAP.md` §4, en orden): chunking por encabezados en `documents.py`, truncado por tokens, falso positivo typescript/typst, reranker medido. `hybrid.py` y `gate.py` siguen apagados salvo que el recall@4 mejore ≥ 0.05. *Aceptación:* recall@4 antes y después con el conjunto de evaluación existente.
- [ ] **T1.6 Información actual.** `web.py`: fechas de publicación, preferir resultados recientes en preguntas sensibles al tiempo, caché con caducidad. *Aceptación:* tres preguntas con fecha respondidas con fuentes fechadas.

### Fase 2 — Cliente de escritorio y web
- [ ] **T2.1 Cascarón Tauri** en `app/`, siguiendo `launcher/src/main.rs`: envuelve el `serve.py` local, icono de bandeja, arranque con la sesión, indicador de NPU/consumo, paleta global de búsqueda. Sin Electron. *Aceptación:* arranca, muestra la bandeja, cierra limpio y no deja un `serve.py` huérfano.
- [ ] **T2.2 Rediseño de la interfaz** de `web/index.html`, compartida por web y cliente. Pantallas: Buscar, Proyecto (grafo + botón graphify), Centinela, Chat, Mediciones, Ajustes. Editorial, sin ruido, oscuro y claro, usable a 360 px y 1440 px. *Aceptación:* capturas en ambos anchos con la herramienta de navegador; `tests/e2e_agent_panel.mjs` sigue pasando.
- [ ] **T2.3 Página pública**: landing + guía «¿funciona mi NPU XDNA1?» con tabla de compatibilidad. *Aceptación:* el sitio estático compila y cada afirmación enlaza a una medición.

### Fase 3 — Centinela y celular
- [ ] **T3.1 Centinela** (`src/sentinel.py`): bucle de bajo consumo con `router.py` y `decide.py` para clasificar eventos (repos con trabajo sin guardar, archivos nuevos, trabajos largos) y avisar con `notify-send`; el LLM despierta solo si el clasificador lo pide. Consolidación solo con el PC ocioso. *Aceptación:* el consumo en reposo con el centinela está a menos de 1 W del reposo sin él (fase de reposo de `bench/energy.py`).
- [ ] **T3.2 Puente al celular** (`bridge/`): bot de Telegram con lista blanca de IDs, o PWA por Tailscale, como proceso aparte que llama a `/v1/chat/completions` con el token local. `serve.py` sigue ligado a `127.0.0.1`. *Aceptación:* se rechaza sin token, fuera de la lista blanca y con Host/Origin incorrectos.
- [ ] **T3.3 Ayudas de programación** (uso 5) como acciones del centinela. *Aceptación:* cada ayuda tiene una comprobación con entrada fija.

### Fase 4 — Voz e investigación de la NPU de primera generación
- [ ] **T4.1 Viabilidad de Whisper.** Partir de [drakosha/whisper-xdna](https://github.com/drakosha/whisper-xdna) (ver `RESEARCH_LOG.md`). Medir Whisper tiny/base int8 en CPU como referencia; estimar si el encoder cabe en `mv_i8` o en el matmul int16. *Aceptación:* ≥ 3× tiempo real y menos vatios que la CPU; si no, registrar el descarte con cifras en `docs/DECISIONS.md`.
- [ ] **T4.2 Reproducir trabajo `npu1` de la comunidad** (`hnws/xdna1-npu-lab`, `FireBurn/ggml-xdna1`, `hawkpoint-npu-llm`, `phoenix-npu1-iron-kernels`) que señale el ciclo de investigación antes de adoptar alguno. *Aceptación:* ≥ 3 corridas en esta máquina y cambio de estado en `RESEARCH_LOG.md`.
- [ ] **T4.3 Informe «XDNA1: qué funciona y qué no»** con cifras, y abrir issues para quienes tengan este hardware.

## 6. Riesgos
- La NPU puede gastar más energía que la CPU en algunas tareas; T0.1 lo decide con datos.
- Voz y juegos dependen de kernels nuevos; no se prometen fechas.
- Las afirmaciones de GitHub de un solo autor pueden ser falsas; siguen `sin verificar` hasta reproducirlas aquí.
