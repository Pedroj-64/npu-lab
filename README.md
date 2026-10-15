# npu-lab

Experiments with the AMD XDNA1 NPU (Ryzen 7040 / 8040 "Phoenix" / "Hawk Point", `1022:1502`)
on Linux: custom kernels with [mlir-aie / IRON](https://github.com/Xilinx/mlir-aie),
run through XRT and `pyxrt`. No vendor LLM stack supports this generation on Linux,
so everything here is built from the low-level pieces.

Spanish: [`README.es.md`](README.es.md).

## Direction
The first-generation NPU (XDNA1, Ryzen 7040/8040) is slower than a GPU, but it is efficient and always available. npu-lab is turning into a small-model assistant for it: private semantic search, a project map you can query cheaply, background help for developers and, later, a desktop client and a phone bridge. Plan, tasks and acceptance checks: [`docs/PLAN.md`](docs/PLAN.md). Architecture: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). Contributing and rules: [`CONTRIBUTING.md`](CONTRIBUTING.md), [`docs/WORKING_RULES.md`](docs/WORKING_RULES.md). What works and what does not on this chip: [`docs/XDNA1_STATUS.md`](docs/XDNA1_STATUS.md). Outside work on this hardware is tracked in [`docs/RESEARCH_LOG.md`](docs/RESEARCH_LOG.md). If you own an XDNA1 machine, measurements and reports are the most useful contribution.

## What works
- **Local LLM chat with the NPU generating every token** (Qwen3, 0.6B and 1.7B), optional GPU assistance for long context, documents,
  local docs and web search: see Quick start and Modes below.
- **Agent mode** (project folder + "agent mode" in the Chat tab, or the terminal chat `python src/chat_cli.py --model qwen35l --project DIR --agent`):
  the model proposes ONE action per reply (read, list, edit, create, delete); reads run alone, anything that writes shows a preview and waits
  for your click (or y/N in the terminal); deletes go to a recoverable trash in `~/.local/share/npu-lab/trash/`. With the 4B, 94% of the calls were
  valid but only 3-4 of 5 test tasks were solved and a false "DONE" still happens, so it is a confirm-each-step helper, not an autonomous agent
  (numbers in `docs/CHANGELOG.md`). Optional, off by default: `RUN` (runs only the test command you configure, `agent_test_command`, after your confirmation) and a `DONE` guard (`agent_guard`: `objective` or `model`) set through the `/rules` API. The web panel (headless browser, scripted model) and the terminal chat (real 4B) pass their end-to-end tests.
- `bench/matmul_npu.py`: int16 matmul on the NPU vs numpy on the CPU (up to ~1.4 TOPS at 1024^3).
- `src/minilm.py` + `src/npu_linear.py`: all-MiniLM-L6-v2 sentence embeddings in numpy,
  with every dense layer offloaded to the NPU. Embeddings match the CPU ones (cosine >= 0.996).
- Numbers and how to reproduce them: [`bench/RESULTS.md`](bench/RESULTS.md).

## Why local documentation (frozen knowledge)
A small model's knowledge stops at its training date (a year or more ago) and it confidently invents commands.
`python src/docs.py --device gpu build` indexes the man pages and `tldr` pages installed on **your** machine (and the
Arch wiki if `arch-wiki-docs` is installed); in the Chat tab, "use local documentation" retrieves the best excerpts for
your question (query embedding runs on one CPU core, ~0.1 s), hands them to the model and shows the sources.
Example (a 1.5B model, "how do I install an AUR package with paru"): without docs it invents that AUR is the
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
| Launcher (`launcher/`, Rust, one small binary) | **Yes**, built and used on CachyOS | Has Windows branches in the code, never built or run there; it can only use the backends above that exist on that OS |
| Web UI and terminal chat (`src/chat_cli.py`) | Yes | Untested |

Hardware tested: AMD Ryzen 5 8645HS (XDNA1 / Hawk Point). XDNA2 (Ryzen AI 300) is untested here; the kernels target `npu1`.

## Quick start (launcher + web UI)
```
cd launcher && cargo build --release && cd ..      # one small binary (Rust)
./launcher/target/release/npu                     # starts the local server and opens your browser
```
The page (`http://127.0.0.1:<random port>/`) has four tabs:
- **Setup**: a **system check** (NPU, XRT, memory limit, kernels, models, tools, with the exact fix for anything missing), the **resource
  profile** (NPU only / Assisted) and its limits (GPU memory the model may use, context window, conversation compaction, GPU share, document cache).
- **Chat**: pick a model and press **Load** (one at a time; Unload frees the NPU), then chat. Drag files onto the page (PDF, Markdown, text,
  HTML, DOCX, images with OCR) to ask about them; sources are shown for every answer: *local docs* (man/tldr/Arch wiki), *web search*,
  *my documents*, each Auto / On / Off, and a line says what was used and why with a one-click override.
- **Search**: semantic search over the folders you allow, with the index controls.
- **Settings**: web search provider, the words that route a question (Auto mode), and **cloud assist** (pick NVIDIA / OpenRouter /
  OpenAI / Gemini / Ollama, paste a key, Connect).
Press Ctrl+C in the terminal to stop it. For VS Code Copilot see [Use the local models from VS Code](#use-the-local-models-from-vs-code-copilot-or-any-openai-style-client).

Security: the server listens on `127.0.0.1` only and every API call needs a random token; the link the launcher
prints carries a **one-time code** (it works once, then it is useless); foreign `Host`/`Origin` headers are refused;
the API never accepts file paths for models and only lets you read folders inside your home.
`python tests/test_server.py` checks all of this plus real NPU and GPU round trips (loads each profile, compaction);
`python tests/eval_routing.py` checks the routing rules offline.

## Use the local models from VS Code (Copilot) or any OpenAI-style client
The server also speaks the OpenAI chat format on the same local address, so an editor can use the model you have loaded (same NPU/GPU
engine, profile, GPU share and cache as the web page). No account and no cloud key is involved: the only secret is an access token **you invent
for your own machine**.

1. Make a token (32+ characters) and start the launcher with a fixed port. Keep the token private; do not commit it:
   ```
   export NPU_LAB_TOKEN=$(openssl rand -hex 32)
   NPU_LAB_PORT=8765 ./launcher/target/release/npu          # add --no-browser if you will not use the web page
   ```
   (A shorter or missing `NPU_LAB_TOKEN` is ignored and a random token is used, which you could not give to the editor.)
2. Load a model: in the web page (Chat tab, pick the profile, **Load**), or from the terminal with `curl -H "Authorization: Bearer $NPU_LAB_TOKEN" -d '{"mode":"llm","model":"qwen3"}' http://127.0.0.1:8765/mode`
   (the default profile is NPU only; `-d '{"profile":"assisted"}' .../profile` lets the GPU help). One model at a time; the editor talks to whichever is loaded.
3. In VS Code add the model to `settings.json` (adjust the numbers to the context window shown in the Chat tab):
   ```json
   "github.copilot.chat.customOAIModels": {
     "qwen3": {
       "name": "Qwen3 1.7B (NPU, local)",
       "url": "http://127.0.0.1:8765/v1/chat/completions",
       "toolCalling": false,
       "vision": false,
       "maxInputTokens": 32000,
       "maxOutputTokens": 2048,
       "requiresAPIKey": true
     }
   }
   ```
   Then open the chat model picker, choose **Manage Models**, and when it asks for the API key paste the value of `$NPU_LAB_TOKEN`.
   The menu names change between Copilot versions; if the entry does not show up, check the setting name in your version.
4. Check without the editor: `curl -H "Authorization: Bearer $NPU_LAB_TOKEN" http://127.0.0.1:8765/v1/models` lists the models.

What it is and is not: `GET /v1/models` and `POST /v1/chat/completions` (streaming or not). Plain chat and code questions work. **Agent mode and tool
calls do not** (the local model cannot call tools; `tools` is ignored), and images are not read. This path skips the web page's routing, web search and
*my documents*: it is the bare model with the editor's own prompt. The same rules as the rest of the API hold: `127.0.0.1` only, token on every call,
foreign `Host`/`Origin` refused; a request while the model is busy answers 409, and with no model loaded, 409 too.

**Terminal chat** (`python src/chat_cli.py`) is separate: it starts its own server on a random port, so it cannot be shared with the editor or the web
page. Close one before opening the other (one program at a time on the NPU/GPU). For the editor, use the launcher as above.

## Modes
The NPU is the center of the project; GPU/CPU are optional helpers, never required.
- **LLM chat**: the NPU generates every token (int8 weights, one shared 4-column kernel); all models are Qwen3 (Apache-2.0),
  downloaded separately into `models/` (see the table). The assistant answers in Spanish by default.

  | Role | Model (folder) | Needs | Decode on a Ryzen 5 8645HS + RTX 3050 6 GB |
  |---|---|---|---|
  | medium | Qwen3-1.7B (`models/qwen3`) | nothing | ~9-10 tok/s at short context, 4k window (see below) |
  | fast | Qwen3-0.6B (`models/qwen3s`) | nothing | ~18-20 tok/s at short context, 4k window (see below) |

  One model is loaded at a time (Unload, then Load). **Profiles:** *NPU only* never touches the GPU (Qwen3 models, 4096 tokens of
  context, prompts read by the NPU at ~90-220 tok/s that fall as the prompt grows, because attention runs on one CPU thread; decode of the 0.6B goes from ~18-20 tok/s
  at short context to ~7 at 3500 tokens; numbers in [`bench/PERF.md`](bench/PERF.md)). *Assisted* lets the GPU read prompts (1000-1800 tok/s), keep the KV cache and run attention, which gives **long context** (default window 80k tokens, up to 262k). Limits are set in the Setup tab:
  GPU memory the model may use (default 4 GB), context window, and when to compact the conversation (the model summarises old turns).
  Kernels for a model: `python src/qwen_npu.py --model DIR --shapes` lists them and `python scripts/build_matvec.py --shared4 ...` builds them (the System check prints the exact command).
  `src/qwen_npu.py` is the engine (dense Qwen-style decoder). The hybrid Qwen3.5 models were removed on 2026-10-15 because they need the GPU for every token; see [`docs/DECISIONS.md`](docs/DECISIONS.md).
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
   and link the system `pyxrt` into it (`.pth` file pointing at the XRT python dir). For the Assisted profile also install PyTorch with CUDA
   (`uv pip install torch`). Documents: `sudo pacman -S poppler` (PDF text) and
   optionally `tesseract tesseract-data-spa tesseract-data-eng` (OCR of scans and images).
5. Clone mlir-aie into `third_party/` and build the `whole_array` matmul kernels
   (see the reproduce section in `bench/RESULTS.md`).
6. Download the models you want into `models/` (any of them; the System check lists what is missing):
   ```
   python - <<'PY'
   from huggingface_hub import snapshot_download
   for repo, d in [("Qwen/Qwen3-1.7B", "models/qwen3"),          # NPU only
                   ("Qwen/Qwen3-0.6B", "models/qwen3s"),         # NPU only, fast
                   ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "models/multi")]:
       snapshot_download(repo, local_dir=d, allow_patterns=["*.json", "*.safetensors", "merges.txt", "vocab.json", "*.jinja"])
   PY
   ```
7. Build the NPU kernels for each model you downloaded (a minute or two per model; add `--share 0.5` for a 50% GPU share):
   ```
   export PEANO_INSTALL_DIR=$PWD/.venv/lib/python3.*/site-packages/llvm-aie PATH=$PWD/.venv/bin:$PATH
   python scripts/build_matvec.py --shared4 $(python src/qwen35_npu.py --model models/qwen35_2b --shapes)
   python scripts/build_prefill.py models/qwen3        # only the NPU-only Qwen3 models read prompts with NPU kernels
   ```
   GPU share kernels: every share needs its own kernels (the build folder `build/` is not in git, so a fresh checkout has to build them). The defaults are 2B 50 %, everything else 0 %; the 2B "max speed" profile (75 %) and the 4B at 25 % are opt-in. If the kernels of a share are missing the Settings page greys that option out and shows the command, and loading the model refuses with the same command (about 80 s for the 2B at 75 %, CPU only):
   ```
   python scripts/build_matvec.py --shared4 $(python src/qwen35_npu.py --model models/qwen35_2b --shapes --share 0.75)
   ```
   Loading also refuses when it would leave less GPU memory free for the desktop than the margin in Settings (700 MiB by default; the 4B at 50 % does not fit).
8. Optional: local documentation index (`python src/docs.py --device gpu build`, after `tldr --update`).

## Status
**Known issue (open):** on the tested laptop the NVIDIA GPU ran at ~2.5 TFLOPS (state P5, memory at 780 MHz) after a boot WITHOUT the charger and at ~15 TFLOPS (P0) after boots WITH it (3 boots, a correlation; root cause not found, see `ROADMAP.md`). Boot with the charger plugged in and check with `sh bench/block1_preflight.sh` before trusting any GPU number.

Usable day to day on the tested machine (Ryzen 5 8645HS + RTX 3050 6 GB, Linux): chat with five models, long context (80k tokens by default; with the 4B
the 6 GB card holds about 22k tokens, the GPU budget is now capped automatically so a value that is too high no longer runs out of memory), documents,
local docs, web search, optional cloud assist, agent mode with confirmation. Known limits: Linux only, XDNA1 only, the Assisted profile needs an NVIDIA GPU with CUDA,
OCR needs `tesseract`, kernels must be built per model (and per GPU share) on your machine.

Docs: [`ROADMAP.md`](ROADMAP.md) (only what is still open, by priority), [`docs/CHANGELOG.md`](docs/CHANGELOG.md) (what is built, with the measured numbers),
[`docs/DECISIONS.md`](docs/DECISIONS.md) (why, and what was looked at and not built), [`bench/PERF.md`](bench/PERF.md) (performance measurements).
The offline tests (`tests/test_agent.py`, `test_project.py`, `test_compact.py`, `test_pins.py`, `test_documents.py`, `test_chats.py`) run in CI on every push.

License: MIT.

## Credits
`src/kernels/matvec.py` is adapted from the [mlir-aie](https://github.com/Xilinx/mlir-aie) matrix_vector example
(Apache-2.0 with LLVM exception; original header kept). Models: Qwen3 (Qwen team, Apache-2.0), paraphrase-multilingual-MiniLM-L12-v2 and all-MiniLM-L6-v2 (sentence-transformers) are downloaded separately and not redistributed here.

The Gated DeltaNet chunked scan in `src/gpu_qwen35.py` is adapted from Hugging Face transformers (`modeling_qwen3_5.py`, Apache-2.0).
