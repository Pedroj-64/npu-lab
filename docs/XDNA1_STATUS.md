# First-generation NPU (XDNA1): what works and what does not

Spanish version: [`XDNA1_STATUS.es.md`](XDNA1_STATUS.es.md). Status date 2026-10-15, machine Ryzen 5 8645HS (Hawk Point), Linux, `amdxdna` + XRT. `[measured]` = measured in this repository; `[reported]` = claimed by a project we read and did not reproduce.

## What works
| Use | Result | Evidence |
|---|---|---|
| Small language models, decode | Qwen3-0.6B 18.5 tok/s at ~30 W, Qwen3-1.7B 9.2 tok/s at ~27 W (whole machine) `[measured]` | `bench/ENERGY.md`, `bench/PERF.md` |
| Sentence embeddings | MiniLM matches the CPU (cosine >= 0.996), ~20 chunks/s `[measured]` | `bench/RESULTS.md` |
| Whisper encoder (tiny/base/small) | correct (same transcription), real-time capable, but slower and more energy hungry than the CPU at these sizes; leaves the CPU cores free `[measured]` | `bench/WHISPER_XDNA.md` |
| Object detection (YOLO-NAS-L int8, Frigate) | 18.8 ms, +6 W, 0.11 J per detection against 1.86 J on the CPU `[reported]`, needs AMD's proprietary Vitis AI runtime | [hnws/xdna1-npu-lab](https://github.com/hnws/xdna1-npu-lab), not reproduced |
| llama.cpp prefill | 1.04x to 1.41x faster than the CPU backend `[reported]` | [FireBurn/ggml-xdna1](https://github.com/FireBurn/ggml-xdna1), not reproduced |

## What does not
- **Decode is capped by memory, not compute.** One token streams the whole model through the NPU's DMA at about 22 GB/s `[measured]` (a community backend reports 13.5 GB/s `[reported]`). More compute does not help; fewer bytes per token does. A GPU wins on speed; the NPU wins only on being cheap to leave on.
- **int4 weights without calibration** lost too much quality; **speculative decoding** does not fit the kernel; a **cross-encoder reranker** gained nothing measurable. All with numbers in `docs/DECISIONS.md`.
- **The big vendor stacks target XDNA2** (FastFlowLM, Lemonade, IRON examples). XDNA1 on Linux is built from low-level pieces: mlir-aie, XRT, the in-tree driver.
- **Hybrid models (Qwen3.5) need the GPU** for their linear-attention layers, so they are out of the NPU-only profile.

## Where the real gain probably is
Always-on, low-power work: embeddings and search in the background, small classifiers, and (reported) camera detection. Large language models are a poor fit; small ones are fine for short answers.

## How to help
You own a Ryzen 7040, 8040 or 8000G? The most useful thing is a measurement: open an issue with the **XDNA1 report** form. Run `bench/watch.py` (about a minute) and, if you like, `bench/energy.py`; paste the output and your kernel, driver and firmware versions. Failures are as useful as successes.
