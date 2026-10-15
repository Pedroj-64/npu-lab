# npu-lab

English: [`README.md`](README.md).

Experimentos y asistente local para la NPU AMD XDNA1 (Ryzen 7040 / 8040 «Phoenix» / «Hawk Point», `1022:1502`) en Linux, con kernels propios de [mlir-aie / IRON](https://github.com/Xilinx/mlir-aie) manejados por XRT y `pyxrt`. Ninguna pila de los fabricantes soporta esta generación en Linux, así que todo se construye con piezas de bajo nivel.

## Dirección
La NPU de primera generación es más lenta que una GPU, pero es eficiente y está siempre disponible. npu-lab se está convirtiendo en un asistente de modelos pequeños para ella: búsqueda semántica privada, un mapa de proyecto que se consulta barato, ayuda de fondo para quien programa y, más adelante, un cliente de escritorio y un puente al celular. Plan, tareas y comprobaciones de aceptación: [`docs/PLAN.es.md`](docs/PLAN.es.md). Arquitectura: [`docs/ARCHITECTURE.es.md`](docs/ARCHITECTURE.es.md). Reglas de trabajo: [`CONTRIBUTING.md`](CONTRIBUTING.md), [`docs/WORKING_RULES.es.md`](docs/WORKING_RULES.es.md). El trabajo ajeno sobre este hardware se sigue en [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md).

## Qué funciona hoy
- Chat local con la NPU generando cada token (Qwen3, 0.6B y 1.7B), con ayuda opcional de la GPU para contexto largo, documentos, documentación local y búsqueda web.
- Modo agente con confirmación en cada escritura, puente compatible con OpenAI para VS Code, embeddings MiniLM en la NPU.
- Cifras y cómo reproducirlas: [`bench/RESULTS.md`](bench/RESULTS.md), [`bench/PERF.md`](bench/PERF.md).

## Inicio rápido
```
cd launcher && cargo build --release && cd ..
./launcher/target/release/npu
```
La instalación completa (Arch / CachyOS), los modos y los límites están en la sección «Setup» de [`README.md`](README.md).

## Compatibilidad
Linux con `amdxdna` + XRT: sí (probado en CachyOS, Ryzen 5 8645HS). Windows: no soportado todavía. XDNA2 (Ryzen AI 300): sin probar; los kernels apuntan a `npu1`.

## Licencia
MIT. Los kernels adaptados de mlir-aie conservan sus cabeceras Apache-2.0 (ver [`NOTICE.md`](NOTICE.md)).
