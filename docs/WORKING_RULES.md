# Working rules for npu-lab

Spanish version: [`WORKING_RULES.es.md`](WORKING_RULES.es.md). Contributors: read this file, then [`PLAN.md`](PLAN.md), then pick the first unchecked task.

## What this project is
A local assistant that runs small language models on the **first-generation AMD XDNA1 NPU** (Ryzen 7040/8040, `npu1`) on Linux, with custom [mlir-aie / IRON](https://github.com/Xilinx/mlir-aie) kernels driven through XRT/`pyxrt`. The NPU is slower than a discrete GPU; its value is **low power and always-on availability without using the GPU**. The product direction (assistant, semantic search, project graph, desktop client, phone bridge) is in [`PLAN.md`](PLAN.md).

## Machine and tooling
- Reference machine: Ryzen 5 8645HS (XDNA1, 4 columns), RTX 3050 6 GB (optional helper), 16 GB shared RAM, CachyOS, XRT 2.21.75, mlir-aie 1.4.4. Details and measured limits: [`../bench/HANDOFF.md`](../bench/HANDOFF.md), [`../bench/PERF.md`](../bench/PERF.md).
- Python lives in `.venv` (3.14). Use `.venv/bin/python`. The user's shell is fish: write helper scripts in Python, not bash.
- To compile kernels: `PEANO_INSTALL_DIR=.venv/lib/python3.14/site-packages/llvm-aie` and `.venv/bin` on `PATH`. Compiling uses only the CPU and can run in parallel.
- **The NPU loads one workload at a time. Never start two processes that use it** (server, `chat_cli.py`, any `bench/*.py`). Check first with `pgrep -af 'src/(serve|chat_cli|qwen)'`.

## Where things are
| Area | Files |
|---|---|
| Server, API, OpenAI-compatible bridge | `src/serve.py` |
| Engines | `src/qwen_npu.py` (Qwen3, NPU only), `src/qwen35_npu.py` + `src/gpu_qwen35.py` (Qwen3.5, assisted) |
| Search and documents | `src/search.py`, `src/documents.py`, `src/docs.py`, `src/hybrid.py` (off), `src/gate.py` (off), `src/web.py` |
| Routing and decisions | `src/router.py`, `src/decide.py` |
| Context | `src/statecache.py`, `src/pins.py`, `src/repomap.py`, `src/facts.py`, `src/chats.py` |
| Agent | `src/agent.py`, `src/project.py` |
| UI | `web/index.html` (vanilla JS), `launcher/` (Rust) |
| Measurements | `bench/` (scripts), `bench/*.md` (reports), `bench/history.jsonl` (continuous log) |
| Decisions | `docs/DECISIONS.md` (what was rejected and why), `docs/CHANGELOG.md` (what was done, with numbers) |

## Rules
1. **Measure before and after** (at least 3 runs, report mean ± std). Label every claim: `[measured]`, `[read in code]`, `[hypothesis]`, `[pending]`. Do not present an estimate as a result.
2. **Behavior changes go behind a flag that is off by default.** Turn it on only after the numbers in the plan are met.
3. **Privacy first.** Nothing from the user's files, paths or conversations leaves the machine. External lookups send search keywords only.
4. **NPU first.** New features must work on the NPU alone; GPU assistance is optional. Record rejected ideas with their numbers in `docs/DECISIONS.md`.
5. **Git:** commits, pushes and PRs are made only by the project owner, under their own identity.
6. **Tests:** `python tests/test_server.py` and the other offline `tests/test_*.py` before and after a change. `tests/eval_*.py` need a running model and the NPU alone.
7. Do not delete user data (`~/.config/npu-lab/`, `~/.local/share/npu-lab/`, `models/`, `bench/*_runs.jsonl`). Raw results in `bench/.*` are referenced by reports: keep them.
8. Documentation is bilingual: every user-facing document has an English file and a Spanish `.es.md` twin. Update both.

## Continuous loops (set up in `docs/PLAN.md`, Phase 1)
- `bench/watch.py` appends a short benchmark to `bench/history.jsonl` and flags regressions.
- `tools/research_scan.py` looks for new work on XDNA1/NPU in GitHub and the web and logs it in `docs/RESEARCH_LOG.md` as `unverified` until reproduced here.
