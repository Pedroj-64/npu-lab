# npu-lab

Experiments with the AMD XDNA1 NPU (Ryzen 7040 / 8040 "Phoenix" / "Hawk Point", `1022:1502`)
on Linux: custom kernels with [mlir-aie / IRON](https://github.com/Xilinx/mlir-aie),
run through XRT and `pyxrt`. No vendor LLM stack supports this generation on Linux,
so everything here is built from the low-level pieces.

## What works
- `bench/matmul_npu.py`: int16 matmul on the NPU vs numpy on the CPU (up to ~1.4 TOPS at 1024^3).
- `src/minilm.py` + `src/npu_linear.py`: all-MiniLM-L6-v2 sentence embeddings in numpy,
  with every dense layer offloaded to the NPU. Embeddings match the CPU ones (cosine >= 0.996).
- Numbers and how to reproduce them: [`bench/RESULTS.md`](bench/RESULTS.md).

## Modes
The NPU is the center of the project; GPU/CPU are optional helpers, never required.
- **LLM chat**: `python src/qwen_npu.py "prompt"` runs Qwen2.5-0.5B-Instruct with every dense layer on the NPU
  (~8 tok/s, ~0.3 CPU cores). The assistant answers in Spanish by default.
- **Semantic file search**: `python src/search.py index` then `python src/search.py search "query"`.
  Embeddings run on the NPU. What may be read is controlled by `~/.config/npu-lab/rules.json`
  (allowed roots/extensions, denied folders and file names, secret-looking chunks are skipped);
  the index stays in `~/.local/share/npu-lab/`, never in the repo.
- **Optional GPU boost for the first big index**: `python src/search.py --device gpu index`
  (needs `torch` with CUDA; VRAM is hard-capped by `gpu_vram_gb` in the rules, default 1.5 GB).
  ~1100 chunks/s vs ~20 on the NPU; vectors are interchangeable. Day-to-day updates use the NPU.

## Setup (Arch / CachyOS)
1. `sudo pacman -S xrt xrt-plugin-amdxdna`
2. Raise the locked-memory limit (XRT needs it), then log out and in:
   `echo -e '* soft memlock unlimited\n* hard memlock unlimited' | sudo tee /etc/security/limits.d/99-npu.conf`
3. Check the device: `xrt-smi examine` should list `RyzenAI-npu1`.
4. Python venv with the toolchain:
   `uv venv .venv && uv pip install mlir_aie llvm-aie numpy scipy tokenizers safetensors -f https://github.com/Xilinx/mlir-aie/releases/expanded_assets/v1.4.4 -f https://github.com/Xilinx/llvm-aie/releases/expanded_assets/nightly`
   and link the system `pyxrt` into it (`.pth` file pointing at the XRT python dir).
5. Clone mlir-aie into `third_party/` and build the `whole_array` matmul kernels
   (see the reproduce section in `bench/RESULTS.md`).
6. Download `sentence-transformers/all-MiniLM-L6-v2` into `models/minilm/`.

## Status / ideas
- Next: int8 matvec kernel (half the weight bytes), more cores per column, a 1.5B model,
  and a small launcher to pick the mode (LLM / search) and a resource profile (NPU-only / assisted).

License: MIT.

## Credits
`src/kernels/matvec.py` is adapted from the [mlir-aie](https://github.com/Xilinx/mlir-aie) matrix_vector example
(Apache-2.0 with LLVM exception; original header kept). Models: Qwen2.5-0.5B-Instruct (Qwen), paraphrase-multilingual-MiniLM-L12-v2 and all-MiniLM-L6-v2 (sentence-transformers) are downloaded separately and not redistributed here.
