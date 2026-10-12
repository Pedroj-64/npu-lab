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
- Latency vs CPU is a tie for a model this small; next is measuring CPU load and power,
  a multilingual embedding model, and a local semantic file search.
- Stretch: a ~0.5-1B LLM on the same kernels (memory-bandwidth bound, so expectations are low).

License: MIT.
