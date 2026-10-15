# NPU de primera generación (XDNA1): qué funciona y qué no

Versión en inglés: [`XDNA1_STATUS.md`](XDNA1_STATUS.md). Estado a 2026-10-15, máquina Ryzen 5 8645HS (Hawk Point), Linux, `amdxdna` + XRT. `[medido]` = medido en este repositorio; `[reportado]` = afirmado por un proyecto que leímos y no reprodujimos.

## Qué funciona
| Uso | Resultado | Evidencia |
|---|---|---|
| Modelos de lenguaje pequeños, generación | Qwen3-0.6B 18.5 tok/s a ~30 W, Qwen3-1.7B 9.2 tok/s a ~27 W (toda la máquina) `[medido]` | `bench/ENERGY.md`, `bench/PERF.md` |
| Embeddings de frases | MiniLM coincide con la CPU (coseno >= 0.996), ~20 trozos/s `[medido]` | `bench/RESULTS.md` |
| Encoder de Whisper (tiny/base/small) | correcto (misma transcripción), en tiempo real, pero más lento y con más energía que la CPU en estos tamaños; deja libres los núcleos `[medido]` | `bench/WHISPER_XDNA.md` |
| Detección de objetos (YOLO-NAS-L int8, Frigate) | 18.8 ms, +6 W, 0.11 J por detección frente a 1.86 J en CPU `[reportado]`, necesita el runtime propietario Vitis AI de AMD | [hnws/xdna1-npu-lab](https://github.com/hnws/xdna1-npu-lab), sin reproducir |
| Prefill de llama.cpp | de 1.04x a 1.41x más rápido que el backend de CPU `[reportado]` | [FireBurn/ggml-xdna1](https://github.com/FireBurn/ggml-xdna1), sin reproducir |

## Qué no
- **La generación está limitada por memoria, no por cálculo.** Cada token hace pasar todo el modelo por el DMA de la NPU a unos 22 GB/s `[medido]` (un backend de la comunidad reporta 13.5 GB/s `[reportado]`). Más cálculo no ayuda; menos bytes por token sí. Una GPU gana en velocidad; la NPU solo gana en ser barata de dejar encendida.
- **Pesos int4 sin calibración** perdían demasiada calidad; la **decodificación especulativa** no cabe en el kernel; un **reranker cross-encoder** no mejoró nada medible. Todo con cifras en `docs/DECISIONS.md`.
- **Las pilas grandes de los fabricantes apuntan a XDNA2** (FastFlowLM, Lemonade, ejemplos de IRON). XDNA1 en Linux se construye con piezas de bajo nivel: mlir-aie, XRT y el driver del kernel.
- **Los modelos híbridos (Qwen3.5) necesitan la GPU** para sus capas de atención lineal, así que quedan fuera del perfil solo NPU.

## Dónde probablemente está la ganancia real
Trabajo siempre encendido y de bajo consumo: embeddings y búsqueda en segundo plano, clasificadores pequeños y (reportado) detección con cámara. Los modelos de lenguaje grandes encajan mal; los pequeños sirven para respuestas cortas.

## Cómo ayudar
¿Tienes un Ryzen 7040, 8040 u 8000G? Lo más útil es una medición: abre un issue con el formulario **XDNA1 report**. Ejecuta `bench/watch.py` (un minuto) y, si quieres, `bench/energy.py`; pega la salida y las versiones de tu kernel, driver y firmware. Los fallos sirven tanto como los éxitos.
