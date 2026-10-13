# Licensing

The original code in this repository is released under the **MIT License** (see `LICENSE`).

## Files derived from other projects
These keep their original license and copyright header; our modifications to them are offered under the same terms.

| File | Derived from | License |
|---|---|---|
| `src/kernels/matvec.py`, `src/kernels/matvec_i8.py` | [mlir-aie](https://github.com/Xilinx/mlir-aie) `programming_examples/basic/matrix_multiplication/matrix_vector` | Apache-2.0 WITH LLVM-exception, (c) Advanced Micro Devices, Inc. |
| `src/kernels/matmul_whole_array.py` | mlir-aie `programming_examples/basic/matrix_multiplication/whole_array` | Apache-2.0 WITH LLVM-exception, (c) Advanced Micro Devices, Inc. |
| `src/kernels/mv_i8.cc` | mlir-aie `aie_kernels/linalg/mv_i16.cc` | Apache-2.0 WITH LLVM-exception, (c) Advanced Micro Devices, Inc. |

## Not included, downloaded by the user
Nothing below is redistributed in this repository; check each license before reuse.

- mlir-aie / IRON and the `llvm-aie` (Peano) compiler: Apache-2.0 with LLVM exception.
- XRT and the `amdxdna` driver: Apache-2.0 / GPL-2.0 (kernel module).
- Qwen3-0.6B and Qwen3-1.7B (also Qwen3-4B/8B): Apache-2.0.
- paraphrase-multilingual-MiniLM-L12-v2 and all-MiniLM-L6-v2: Apache-2.0.
- The Gated DeltaNet chunked scan in `src/gpu_qwen35.py` is adapted from Hugging Face transformers (`modeling_qwen3_5.py`): Apache-2.0, Copyright The Qwen team and The HuggingFace Inc. team.
