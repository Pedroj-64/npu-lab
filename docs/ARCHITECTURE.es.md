# Arquitectura

Versión en inglés: [`ARCHITECTURE.md`](ARCHITECTURE.md). Etiquetas: `[medido]`, `[leído en código]`.

## Reparto del procesamiento
- **NPU** (XDNA1, 4 columnas): decodifica cada token con kernels matriz-vector int8 (`src/kernels/`, construidos con `scripts/build_*.py`). Un xclbin y un contexto de hardware de 4 columnas por modelo. El decode está limitado por DMA: 6.6 / 12.3 / 21.8 GB/s con 1 / 2 / 4 columnas `[medido, bench/DMA_READ.md]`. Los embeddings MiniLM ejecutan aquí sus capas densas.
- **GPU** (opcional, NVIDIA/CUDA): lee prompts largos, guarda el KV y el estado de DeltaNet, puede calcular del 0 al 75 % de cada matriz y construye índices grandes de embeddings.
- **CPU**: pegamento, atención del prefill solo-NPU (numpy de un hilo), tokenización, búsqueda, servidor.

Perfiles: **Solo NPU** (Qwen3-0.6B ≈ 19.8 tok/s, 1.7B ≈ 10.1 tok/s, contexto 4k) y **Asistido** (Qwen3.5 0.8B/2B/4B con la GPU) `[medido, bench/PERF.md]`.

## Flujo de una petición
```
cliente (web/index.html, launcher, app Tauri, VS Code por /v1, puente al celular)
   -> src/serve.py  (solo 127.0.0.1; token bearer + código de acceso de un solo uso; comprobación Host/Origin)
        -> router.py / decide.py   elegir: docs | web | historial | chat simple
        -> search.py / documents.py / docs.py / web.py   recuperar extractos (MiniLM en NPU, GPU para índices grandes)
        -> statecache.py / pins.py   reutilizar el estado del prefijo, conservar literales las líneas con datos al compactar
        -> qwen_npu.py | qwen35_npu.py (+gpu_qwen35.py)   generar tokens
        <- respuesta en flujo, fuentes visibles para el usuario
```

## Endpoints del servidor (`src/serve.py`, `[leído en código]`)
GET: `/status`, `/v1/models`, `/routing`, `/documents`, `/rules`, `/facts`, `/chats`, `/doctor`, `/settings`.
POST: `/login`, `/mode`, `/profile`, `/rules`, `/documents/upload`, `/documents/delete`, `/agent/run` (una acción; solo escribe con `confirm=true`), `/project/apply`, `/facts`, `/chats/save`, `/chats/load`, `/routing`, `/settings`, `/settings/models`, `/settings/test_web`, `/chat`, `/v1/chat/completions`, `/search`, `/index`, `/index/stop`.
Previstos (ver `PLAN.es.md`): `POST /project/graph`, `GET /project/graph/query`.

## Datos en disco
| Ruta | Contenido |
|---|---|
| `~/.config/npu-lab/token`, `providers.json` (0600), `rules.json`, ajustes | Token de acceso, claves de proveedores (solo escritura desde la interfaz), reglas y límites |
| `~/.local/share/npu-lab/` | Índices de documentos y docs, chats, hechos, caché de estado, `trash/` (borrados recuperables) |
| `models/` (ignorado por git) | Pesos de los modelos, descargados aparte (modelos Apache-2.0) |
| `build/` (ignorado por git) | xclbins compilados y flujos de instrucciones |
| `bench/history.jsonl` | Una línea por corrida de la medición continua |
| `docs/RESEARCH_LOG.md` | Hallazgos externos y su estado de verificación |

## Componentes previstos
```
sentinel.py  --eventos (git status, archivos nuevos, trabajos largos)--> clasificador decide.py
      |                                                                   | pregunta solo si hace falta
      +--notify-send / panel de la UI <---------------- modelo pequeño (NPU)

bridge/ (proceso aparte)  celular -> bot de Telegram / PWA por Tailscale -> lista blanca -> POST /v1/chat/completions (token local)

graph.py  -> subproceso `graphify <ruta> --update` -> graphify-out/ -> `graphify query --budget N` -> contexto corto
```

## Modelo de seguridad
Servidor solo en loopback; token + comprobación Host/Origin; `web.py` rechaza destinos privados o de loopback; los proveedores en la nube son opcionales, por mensaje y rotulados; los archivos indexados nunca se envían a un proveedor; el texto web no es de fiar (el modelo no tiene herramientas de escritura sin confirmación del usuario). El puente al celular y el ciclo de investigación no deben ampliar esto: el puente es un cliente aparte con autenticación y el ciclo de investigación envía solo palabras clave.
