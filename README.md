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

## Why local documentation (frozen knowledge)
A small model's knowledge stops at its training date (Qwen2.5: around 2023-24) and it confidently invents commands.
`python src/docs.py --device gpu build` indexes the man pages and `tldr` pages installed on **your** machine (and the
Arch wiki if `arch-wiki-docs` is installed); in the Chat tab, "use local documentation" retrieves the best excerpts for
your question (query embedding runs on one CPU core, ~0.1 s), hands them to the model and shows the sources.
Example (Qwen2.5-1.5B, "how do I install an AUR package with paru"): without docs it invents that AUR is the
"Arch Linux App Repository" and that `paru` comes from `sudo pacman -S paru`; with docs it answers `paru -S package`.
Run `tldr --update` first (tealdeer or tldr client). Index build: ~1.5 min with a GPU (CPU part: rendering man pages).
It is a mitigation, not magic: a 1.5B model can still misread a correct page, so read the cited source and run commands yourself.

## Web search and cloud assist (opt-in, configured in the Settings tab)
- **Web search** (Chat tab, "web search"): the server searches, reads the top pages and gives the model short excerpts, then shows the
  sources. Providers: DuckDuckGo and Wikipedia (no key), your own SearXNG, or Brave (API key). The server refuses to fetch anything
  that resolves to a private/loopback address, so it cannot be used to reach your LAN. Treat web text as untrusted: the model has no
  tools, it can only be misled, so check the cited links.
- **Cloud assist** (Chat tab, backend "Cloud"): bring your own key for any OpenAI-compatible endpoint (NVIDIA NIM / Nemotron,
  OpenRouter, Ollama...) or Gemini; "List models" asks the provider which models exist so you do not guess names. It is per message and
  clearly labelled: your question and the conversation leave the PC. Your indexed files are never sent; docs/web excerpts only if you tick
  "also send the excerpts". Keys live in `~/.config/npu-lab/providers.json` (mode 0600), are write-only from the UI and never appear in
  responses or errors. Free tiers often train on your data: read your provider's terms.
- The chat renders Markdown (code blocks with a copy button and light highlighting, lists, links), shows what it is doing (searching docs,
  searching the web, reading N tokens, writing), folds reasoning (`<think>`) into a collapsible block, and remembers the conversation.

## Platform support (read this first)
| Part | Linux | Windows |
|---|---|---|
| NPU backends (kernels, LLM, embeddings) | **Yes**, tested on CachyOS with `amdxdna` + XRT | **Not supported yet.** The NPU is driven by AMD's Ryzen AI stack there, which this project does not use or test |
| GPU embedding helper (`--device gpu`) | Yes (NVIDIA/CUDA tested) | Untested |
| CPU fallback (`--device cpu`) | Yes | Should work (plain numpy), untested |
| Launcher (planned, Rust) | Planned | Planned (portable code; it can only use the backends above that exist on that OS) |

Hardware tested: AMD Ryzen 5 8645HS (XDNA1 / Hawk Point). XDNA2 (Ryzen AI 300) is untested here; the kernels target `npu1`.

## Quick start (launcher + web UI)
```
cd launcher && cargo build --release && cd ..      # one small binary (Rust)
./launcher/target/release/npu                     # starts the local server and opens your browser
```
The page (`http://127.0.0.1:<random port>/`) walks you through a **system check** (NPU, XRT, memory limit, kernels,
models, with the exact fix for anything missing), lets you pick the **resource profile** (NPU only / assisted),
**load one mode at a time** (LLM chat or semantic search; switching frees the NPU), chat, search, run the index
and choose which folders may be read. Press Ctrl+C in the terminal to stop it.

Security: the server listens on `127.0.0.1` only and every API call needs a random token; the link the launcher
prints carries a **one-time code** (it works once, then it is useless); foreign `Host`/`Origin` headers are refused;
the API never accepts file paths for models and only lets you read folders inside your home.
`python tests/test_server.py` checks all of this plus a real NPU round trip.

## Modes
The NPU is the center of the project; GPU/CPU are optional helpers, never required.
- **LLM chat**: `python src/qwen_npu.py "prompt"` runs Qwen2.5-Instruct with every dense layer on the NPU
  (int8 weights): 0.5B ~15.7 tok/s, 1.5B (`--model models/qwen15`) ~6.3 tok/s, ~0.3 CPU cores.
  The assistant answers in Spanish by default. Kernels for a model: `python src/qwen_npu.py --model DIR --shapes`
  lists them, `scripts/build_matvec.py --shared ...` builds them; `scripts/build_prefill.py DIR` builds the
  prompt-reading kernels (bf16 matmuls, ~170 tok/s on the 0.5B, ~56 tok/s on the 1.5B instead of reading token by token).
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

See [`ROADMAP.md`](ROADMAP.md) for what is next (per-question routing for docs/web, better memory, evaluation).

License: MIT.

## Credits
`src/kernels/matvec.py` is adapted from the [mlir-aie](https://github.com/Xilinx/mlir-aie) matrix_vector example
(Apache-2.0 with LLVM exception; original header kept). Models: Qwen2.5-0.5B-Instruct (Qwen), paraphrase-multilingual-MiniLM-L12-v2 and all-MiniLM-L6-v2 (sentence-transformers) are downloaded separately and not redistributed here.
