# Research log / Registro de investigación

Findings from outside the project about using the first-generation NPU (XDNA1 / `npu1`). / Hallazgos externos sobre el uso de la NPU de primera generación.

Status / Estado: `unverified` (sin verificar) → `reproduced here` (reproducido aquí, ≥ 3 runs on this machine) → `rejected` (descartado). Nothing is adopted while `unverified`. / Nada se adopta mientras esté `sin verificar`.

`tools/research_scan.py` appends new GitHub hits under "Automatic scan" and never edits the manual table. / agrega hallazgos nuevos de GitHub en «Automatic scan» y no toca la tabla manual.

## Manual findings / Hallazgos manuales
| Date | Source | What it says | Status | Related task |
|---|---|---|---|---|
| 2026-10-14 | [Linux kernel docs, amdxdna](https://docs.kernel.org/accel/amdxdna/amdnpu.html) | Phoenix and Hawk Point NPUs: 4x5 tile topology, 2560 KB L2; upstream `amdxdna` driver | unverified (docs) | T0.1 |
| 2026-10-14 | [Lemonade FLM on Linux](https://lemonade-server.ai/flm_npu_linux.html) | LLMs and Whisper on **XDNA2** only; Ryzen AI 7000/8000 (XDNA1) "not supported" | unverified | PLAN §4 |
| 2026-10-14 | [Phoronix, Ryzen AI NPUs on Linux](https://www.phoronix.com/news/AMD-Ryzen-AI-NPUs-Linux-LLMs) | FastFlowLM 0.9.35 native Linux, needs Lemonade 10.0 and Linux 7.0 or backports; targets Ryzen AI 300/400 | unverified | PLAN §4 |
| 2026-10-14 | [FastFlowLM](https://github.com/shdu-amd/FastFlowLM) | README lists vision, audio, embedding and MoE support on Ryzen AI NPUs (XDNA2) | unverified | T4.1 |
| 2026-10-14 | [Scottcjn/npu-linux-kit](https://github.com/Scottcjn/npu-linux-kit) | Community kit; webcam module listed as working, embeddings/audio/ggml as roadmap; single author | unverified | T4.2 |
| 2026-10-14 | [mrtib/phoenix-npu1-iron-kernels](https://huggingface.co/mrtib/phoenix-npu1-iron-kernels) | int16 GEMM and bf16 add kernels for Phoenix NPU1, built with mlir-aie 1.3.2 on a Ryzen 7 7840HS; do not load on NPU2/4 | unverified | T4.2 |
| 2026-10-14 | [arXiv 2512.13282](https://arxiv.org/pdf/2512.13282) | GEMM performance across Ryzen AI NPU generations using IRON | unverified | T4.1 |
| 2026-10-14 | [amd/iron](https://github.com/amd/iron) | Official IRON repo; targets NPU1/NPU2, bfloat16 primary; no Whisper example found | unverified | T4.1 |

## Candidate dense models for NPU-only (2026-10-15) / Modelos densos candidatos para solo NPU
Decode is DMA-bound, so tok/s ≈ 15.8 GB/s (effective, from Qwen3-1.7B: 9.2 tok/s × 1.72 GB) ÷ int8 bytes per token. Figures below are **estimates from parameter counts, not measurements**; architecture and licence details come from web search and must be checked on each model card. / tok/s estimados, sin medir.

| Model | Params | Est. tok/s | Licence (per sources) | Engine work | Status |
|---|---|---|---|---|---|
| Qwen3-1.7B / 0.6B (current) | 1.7B / 0.6B | 9.2 / 18.5 measured | Apache-2.0 | none | in use |
| Llama 3.2 1B | 1.24B | ~12 | Llama 3.2 Community (not Apache) | small: no q/k-norm (engine already has the non-Qwen3 path), llama3 RoPE scaling | unverified |
| Gemma 3 1B | ~1.0B | ~15 | check card (sources conflict) | medium: sliding-window layers, extra norms, GeLU | unverified |
| Gemma 4 E2B | ~2B effective | unknown | Apache-2.0 per Wikipedia, conflicting | unknown: "E" models use per-layer embeddings | unverified |
| SmolLM3-3B | 3.08B | ~5 | Apache-2.0 | small-medium: GQA 4 groups, NoPE every 4th layer | unverified |
| Ministral 3 3B | 3.4B text + 0.4B vision | ~4.6 | Apache-2.0 | small-medium: dense Mistral-style, text checkpoints load as `Ministral3Model` | unverified |
| Phi-4-mini | 3.8B | ~4 | MIT | small-medium | unverified |
| Granite 4.x 3B | ~3B | ~5 | Apache-2.0 | check: some Granite 4 models are hybrid (Mamba) | unverified |

Sources: [premai.io](https://www.premai.io/blog/best-lightweight-language-models-worth-running/), [Turing Post](https://www.turingpost.com/p/slmslist), [tinyweights.dev](https://tinyweights.dev/posts/best-small-language-models-2026/), [Ministral 3 on Hugging Face](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512-BF16), [SmolLM3 blog](https://www.huggingface.co/blog/smollm3), [Gemma 3 in NeMo docs](https://docs.nvidia.com/nemo/megatron-bridge/latest/models/gemma/gemma3.html).

## Priority leads from the first scan / Pistas prioritarias del primer escaneo
First automatic scan (2026-10-14) found XDNA1-specific work that changes the plan; all are single-author or small and stay `unverified` until reproduced here. / El primer escaneo encontró trabajo específico para XDNA1; todo sigue `sin verificar` hasta reproducirlo.
- [drakosha/whisper-xdna](https://github.com/drakosha/whisper-xdna): Whisper encoder on XDNA1 with the open toolchain (XRT + MLIR-AIE/IRON). Directly relevant to **T4.1** (voice). Check kernels, dtype, speed, and licence first.
- [hnws/xdna1-npu-lab](https://github.com/hnws/xdna1-npu-lab): Frigate object detection, LLMs and GEMMs on first-gen XDNA. Relevant to **T4.2** and to vision use cases.
- [FireBurn/ggml-xdna1](https://github.com/FireBurn/ggml-xdna1) and [c8dhjp4tyv-bit/hawkpoint-npu-llm](https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm): ggml backend / LLM stack for XDNA1; compare decode speed against `bench/PERF.md`. **T4.2**.
- [tibrezus/xdna-npu-toolkit](https://github.com/tibrezus/xdna-npu-toolkit) and [michael-greider/xdna-stat](https://github.com/michael-greider/xdna-stat): detection and monitoring tools; candidates for the NPU indicator in the desktop client (**T2.1**).
- [amd/xdna-driver](https://github.com/amd/xdna-driver): official driver and XRT SHIM, watch releases for `npu1` regressions (**T0.1**).

## Automatic scan / Escaneo automático
<!-- research_scan:begin -->
- 2026-10-14 · unverified · [FireBurn/ggml-xdna1 — A GGML backend for the AMD Ryzen AI **XDNA1 / AIE2** NPU, talking directly to the upstream Linux `amdxdna` DRM (updated 2026-09-15, 0 stars)](https://github.com/FireBurn/ggml-xdna1)
- 2026-10-14 · unverified · [Joassens/bazarr-subtitles-xdna1 — Bazarr subtitles with XDNA1: Whisper on the AMD XDNA1 NPU (Ryzen 7 8845HS) with word timestamps, VAD, lyrics,  (updated 2026-10-13, 0 stars)](https://github.com/Joassens/bazarr-subtitles-xdna1)
- 2026-10-14 · unverified · [Jonas-Augustinus-Linus/ryzen-npu-linux — Open, reproducible AMD XDNA1 + Strix Point XDNA2 compute on Linux: pinned builds, CPU-reference checks, persis (updated 2026-08-16, 2 stars)](https://github.com/Jonas-Augustinus-Linus/ryzen-npu-linux)
- 2026-10-14 · unverified · [FastFlowLM/FastFlowLM release v1.0.7 (2026-09-30)](https://github.com/ROCm/FastFlowLM/releases/tag/v1.0.7)
- 2026-10-14 · unverified · [FastFlowLM/FastFlowLM release v1.0.8 (2026-10-12)](https://github.com/ROCm/FastFlowLM/releases/tag/v1.0.8)
- 2026-10-14 · unverified · [Xilinx/mlir-aie release v1.4.3 (2026-09-11)](https://github.com/Xilinx/mlir-aie/releases/tag/v1.4.3)
- 2026-10-14 · unverified · [Xilinx/mlir-aie release v1.4.4 (2026-10-06)](https://github.com/Xilinx/mlir-aie/releases/tag/v1.4.4)
- 2026-10-14 · unverified · [amd/iron release v0.0.0 (2025-10-30)](https://github.com/amd/IRON/releases/tag/v0.0.0)
- 2026-10-14 · unverified · [amd/xdna-driver — AMD XDNA Driver (amdxdna.ko) for Linux and XRT SHIM library (updated 2026-10-14, 622 stars)](https://github.com/amd/xdna-driver)
- 2026-10-14 · unverified · [bastienbrasseur/whisper-npu-dictation — Global push-to-talk dictation on Windows, Whisper Large-v3-Turbo running entirely on the AMD Ryzen AI NPU (XDN (updated 2026-04-26, 0 stars)](https://github.com/bastienbrasseur/whisper-npu-dictation)
- 2026-10-14 · unverified · [buckster123/ryzenai-lab — Experiments, notes, and benchmarks for AMD Ryzen AI (Krackan Point NPU + Radeon iGPU) on Linux (updated 2026-04-21, 1 stars)](https://github.com/buckster123/ryzenai-lab)
- 2026-10-14 · unverified · [c8dhjp4tyv-bit/hawkpoint-npu-llm — Run LLMs and Ollama Qwen across AMD XDNA1 NPU, CPU, and GPU on Ryzen AI Linux systems. (updated 2026-10-14, 25 stars)](https://github.com/c8dhjp4tyv-bit/hawkpoint-npu-llm)
- 2026-10-14 · unverified · [c8dhjp4tyv-bit/xdna1-npu-emulator — Functional AMD XDNA1 (Phoenix/Hawk Point) NPU emulator for QEMU, targeting stock amdxdna + XRT guests. (updated 2026-09-09, 0 stars)](https://github.com/c8dhjp4tyv-bit/xdna1-npu-emulator)
- 2026-10-14 · unverified · [craigchandler/ryzen-ai-arch — Run AMD Ryzen AI / XDNA2 NPUs on Arch Linux and CachyOS using the in-tree amdxdna driver, XRT, Ryzen AI SDK, a (updated 2026-10-01, 0 stars)](https://github.com/craigchandler/ryzen-ai-arch)
- 2026-10-14 · unverified · [drakosha/whisper-xdna — Whisper encoder on the AMD XDNA1 NPU (Ryzen AI, Phoenix) under Linux — open toolchain only: XRT, MLIR-AIE/IRON (updated 2026-07-26, 0 stars)](https://github.com/drakosha/whisper-xdna)
- 2026-10-14 · unverified · [hnws/xdna1-npu-lab — Frigate object detection, LLMs and GEMMs on the first-gen AMD XDNA NPU (Ryzen 7040/8040/8000G, Phoenix) on Lin (updated 2026-09-28, 0 stars)](https://github.com/hnws/xdna1-npu-lab)
- 2026-10-14 · unverified · [huppiflupp/strix-halo-npu-linux — AMD NPU (XDNA2) on Nobara/Fedora Linux for Ryzen AI Max+ 395 (Strix Halo): from-source XRT + amdxdna build, pa (updated 2026-09-19, 0 stars)](https://github.com/huppiflupp/strix-halo-npu-linux)
- 2026-10-14 · unverified · [jac-76/npu-whisper — Whisper ASR on the AMD Ryzen AI NPU — zero-dependency Bash wrapper over FastFlowLM, plus a RAPL power/latency  (updated 2026-09-30, 0 stars)](https://github.com/jac-76/npu-whisper)
- 2026-10-14 · unverified · [julianmb/npuhalo-speech — NPU-accelerated speech transcription & speaker diarization on AMD Strix Halo: Whisper-v3-turbo on XDNA 2 + pya (updated 2026-08-31, 12 stars)](https://github.com/julianmb/npuhalo-speech)
- 2026-10-14 · unverified · [lemonade-sdk/lemonade release candidate-v2026.43.0 (2026-10-14)](https://github.com/lemonade-sdk/lemonade/releases/tag/candidate-v2026.43.0)
- 2026-10-14 · unverified · [lemonade-sdk/lemonade release v2026.42.1 (2026-10-14)](https://github.com/lemonade-sdk/lemonade/releases/tag/v2026.42.1)
- 2026-10-14 · unverified · [michael-greider/xdna-stat — GUI NPU Monitoring in glorious and pure C only using IOCTL_DRM_AMDXDNA_GET_INFO, Linux kernel mainline recomme (updated 2026-08-07, 1 stars)](https://github.com/michael-greider/xdna-stat)
- 2026-10-14 · unverified · [morta5/talos-kernel-amdxdna —  (updated 2026-10-12, 0 stars)](https://github.com/morta5/talos-kernel-amdxdna)
- 2026-10-14 · unverified · [tibrezus/xdna-npu-toolkit — Detect, validate, enable and assess the AMD XDNA NPU on Linux. Dependency-free. Gives an honest, machine-speci (updated 2026-06-15, 2 stars)](https://github.com/tibrezus/xdna-npu-toolkit)
<!-- research_scan:end -->
