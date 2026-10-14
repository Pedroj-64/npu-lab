# Reglas de trabajo de npu-lab

Versión en inglés: [`WORKING_RULES.md`](WORKING_RULES.md). Quien contribuya: lee este archivo, luego [`PLAN.es.md`](PLAN.es.md) y elige la primera tarea sin marcar.

## Qué es este proyecto
Un asistente local que corre modelos de lenguaje pequeños en la **NPU AMD XDNA1 de primera generación** (Ryzen 7040/8040, `npu1`) en Linux, con kernels propios de [mlir-aie / IRON](https://github.com/Xilinx/mlir-aie) manejados por XRT/`pyxrt`. La NPU es más lenta que una GPU dedicada; su valor es el **bajo consumo y estar siempre disponible sin usar la GPU**. La dirección de producto (asistente, búsqueda semántica, grafo de proyecto, cliente de escritorio, puente al celular) está en [`PLAN.es.md`](PLAN.es.md).

## Máquina y herramientas
- Máquina de referencia: Ryzen 5 8645HS (XDNA1, 4 columnas), RTX 3050 6 GB (ayuda opcional), 16 GB de RAM compartida, CachyOS, XRT 2.21.75, mlir-aie 1.4.4. Detalles y límites medidos: [`../bench/HANDOFF.md`](../bench/HANDOFF.md), [`../bench/PERF.md`](../bench/PERF.md).
- Python vive en `.venv` (3.14). Usa `.venv/bin/python`. La shell del usuario es fish: escribe los scripts auxiliares en Python, no en bash.
- Para compilar kernels: `PEANO_INSTALL_DIR=.venv/lib/python3.14/site-packages/llvm-aie` y `.venv/bin` en el `PATH`. Compilar usa solo CPU y puede hacerse en paralelo.
- **La NPU carga una sola carga de trabajo a la vez. Nunca arranques dos procesos que la usen** (servidor, `chat_cli.py`, cualquier `bench/*.py`). Comprueba antes con `pgrep -af 'src/(serve|chat_cli|qwen)'`.

## Dónde está cada cosa
| Área | Archivos |
|---|---|
| Servidor, API, puente compatible con OpenAI | `src/serve.py` |
| Motores | `src/qwen_npu.py` (Qwen3, solo NPU), `src/qwen35_npu.py` + `src/gpu_qwen35.py` (Qwen3.5, asistido) |
| Búsqueda y documentos | `src/search.py`, `src/documents.py`, `src/docs.py`, `src/hybrid.py` (apagado), `src/gate.py` (apagado), `src/web.py` |
| Enrutado y decisiones | `src/router.py`, `src/decide.py` |
| Contexto | `src/statecache.py`, `src/pins.py`, `src/repomap.py`, `src/facts.py`, `src/chats.py` |
| Agente | `src/agent.py`, `src/project.py` |
| Interfaz | `web/index.html` (JS puro), `launcher/` (Rust) |
| Mediciones | `bench/` (scripts), `bench/*.md` (informes), `bench/history.jsonl` (registro continuo) |
| Decisiones | `docs/DECISIONS.md` (qué se descartó y por qué), `docs/CHANGELOG.md` (qué se hizo, con cifras) |

## Reglas
1. **Mide antes y después** (mínimo 3 corridas, media ± desviación). Etiqueta cada afirmación: `[medido]`, `[leído en código]`, `[hipótesis]`, `[pendiente]`. No presentes una estimación como resultado.
2. **Los cambios de comportamiento van tras una bandera apagada por defecto.** Se activa solo cuando se cumplen las cifras del plan.
3. **Privacidad primero.** Nada de los archivos, rutas o conversaciones del usuario sale de la máquina. Las consultas externas envían solo palabras clave.
4. **NPU primero.** Las funciones nuevas deben funcionar solo con la NPU; la ayuda de la GPU es opcional. Anota las ideas descartadas, con sus cifras, en `docs/DECISIONS.md`.
5. **Git:** los commits, push y PRs los hace solo el dueño del proyecto, con su propia identidad.
6. **Pruebas:** `python tests/test_server.py` y los demás `tests/test_*.py` sin conexión antes y después de un cambio. Los `tests/eval_*.py` necesitan un modelo cargado y la NPU libre.
7. No borres datos del usuario (`~/.config/npu-lab/`, `~/.local/share/npu-lab/`, `models/`, `bench/*_runs.jsonl`). Los resultados crudos de `bench/.*` los citan los informes: consérvalos.
8. La documentación es bilingüe: cada documento para el usuario tiene archivo en inglés y gemelo `.es.md` en español. Actualiza ambos.

## Ciclos continuos (se montan en `docs/PLAN.es.md`, Fase 1)
- `bench/watch.py` añade una medición corta a `bench/history.jsonl` y marca regresiones.
- `tools/research_scan.py` busca trabajo nuevo sobre XDNA1/NPU en GitHub y la web y lo anota en `docs/RESEARCH_LOG.md` como `sin verificar` hasta reproducirlo aquí.
