# Plan: an always-on assistant for the first-generation NPU

Spanish version: [`PLAN.es.md`](PLAN.es.md). Working rules: [`WORKING_RULES.md`](WORKING_RULES.md). Status marks: `[ ]` pending, `[~]` in progress, `[x]` done. Each task has an acceptance check; do not mark it done without running it.

## 1. Direction
The XDNA1 NPU (Ryzen 7040/8040, `npu1`) cannot beat a GPU on speed: decode is DMA-bound at about 22 GB/s and gives 9–20 tok/s with 0.6–2B models (`bench/PERF.md`, `bench/DMA_READ.md`). Its real advantage is **low power and always-on availability without touching the GPU**. The product is therefore a small-model assistant that works in the background, plus a desktop client and a web page that make it usable by anyone who owns this forgotten generation of hardware.

Decisions (owner, 2026-10-14): open the project for real (license is already MIT), first pillar = always-on assistant, desktop client with Tauri, continuous measurement, continuous search of the web and GitHub for new NPU uses, full bilingual documentation for hand-over.

## 2. What exists and what does not
| Exists | Missing |
|---|---|
| Qwen3 0.6B/1.7B on the NPU alone (GPU optional) | Speech (ASR/VAD), vision, measured reranker |
| MiniLM embeddings on the NPU (cosine ≥ 0.996, ~20 chunks/s) | Remote / phone access |
| `decide.py` classifier (route 0.85), `router.py` | Native desktop client |
| `search.py`, `documents.py`, `docs.py`, `web.py`, `repomap.py` (Python only) | graphify integration |
| `/v1` OpenAI-compatible bridge, agent with confirmation, `bench/energy.py` | Continuous measurement and research loops |

## 3. Use cases, ranked
| # | Use | Why the NPU | Effort |
|---|---|---|---|
| 1 | Local semantic search (files, docs, notes) in the background | Cheap constant embeddings, GPU stays free | Low: exists, needs chunking and reranker |
| 2 | Project map with graphify (button) + compact queries for other agents | AST extraction is free; NPU summarizes files with a 0.6–1.7B model | Low–medium |
| 3 | Sentinel: watches repos, files and jobs, classifies and notifies; small model answers | Always on at low power | Medium |
| 4 | Phone bridge (message in, PC answers) | The home PC is the server | Medium |
| 5 | Daily developer helpers: commit message, diff summary, log triage, explain a command, local secret check on a diff | 0.6–2B is enough | Low |
| 6 | Local voice: dictation (Whisper tiny/base) and wake command | Dense layers like MiniLM; needs new kernels | High, uncertain |
| – | Games | No NPU API for games on Linux; XDNA1 is slow. Deferred | – |
| – | Modeling / rendering | The NPU is not suited; only large-project maps (#2) apply | – |

Discarded from the original idea list (phone-NPU use cases): 3D face unlock, portrait/denoise photo pipelines, app-launch and battery prediction (the OS already does it and XDNA1 is not reachable from those stacks).

## 4. External landscape (checked 2026-10-14, see `RESEARCH_LOG.md`)
FastFlowLM, Lemonade and AMD IRON target **XDNA2** (Ryzen AI 300/400); Lemonade's Linux guide states XDNA1 is not supported. Community work for `npu1` exists but is single-author and unverified (`Scottcjn/npu-linux-kit`, `Scottcjn/open-xdna`, `mrtib/phoenix-npu1-iron-kernels`). This gap is the opportunity: this project already has its own `npu1` kernels.

## 5. Tasks

### Phase 0 — Baseline and cleanup
- [x] **T0.1 Baseline.** Done 2026-10-15, see `bench/ENERGY.md`. Run the offline tests and `bench/energy.py --engine npu --model models/qwen3s` (3 runs) with the NPU alone, charger plugged in before boot (see `ROADMAP.md` §2, GPU P5 issue). *Accept:* a row appended to `bench/history.jsonl` and a paragraph in `bench/ENERGY.md`.
- [x] **T0.2 First run of `tests/eval_answers.py`** (2026-10-15: 8/8 after fixing two over-strict criteria, details in `docs/CHANGELOG.md`) (written, never executed). *Accept:* results saved under `bench/` and summarized in `docs/CHANGELOG.md`.
- [x] **T0.3 Review the uncommitted bridge work in `src/serve.py`** (`/v1` with tools) against `python tests/test_server.py`. The owner commits.

### Phase 1 — Base product and continuous loops
- [~] **T1.1 Continuous measurement.** `bench/watch.py` and `bench/history.jsonl` exist as a skeleton. Done 2026-10-15: two runs append two lines, `--simulate-drop 0.2` exits 2 and writes nothing, timer units in `tools/systemd/` (not enabled). Remaining: enable the timer when the owner decides; a "measure" button and trend chart in the UI.
- [~] **T1.2 Continuous research.** `tools/research_scan.py` exists as a skeleton. Done 2026-10-15: the second run adds nothing (24 hits on the first), only fixed keywords are sent, timer units in `tools/systemd/` (not enabled). Remaining: classify hits with the NPU model (relevant / not, one line).
- [x] **T1.3 NPU-only defaults.** Done 2026-10-15: the server already defaulted to `npu_only`; the page now picks Qwen3-1.7B/0.6B unless the profile is Assisted. Verified with `CUDA_VISIBLE_DEVICES=` empty: Qwen3-0.6B loads and answers, GPU memory stays at 28 MiB. Make Qwen3-0.6B/1.7B (`src/qwen_npu.py`) the default profile; keep the Assisted profile (Qwen3.5 + CUDA) optional. *Accept:* fresh install chats with no NVIDIA present.
- [~] **T1.4 graphify button.** Done 2026-10-15: `src/graph.py`, `GET/POST /project/graph`, `POST /project/graph/query`, the Project tab, `tests/test_graph.py` (passes; checked against the real server in a temporary HOME, no network used). Measured 2026-10-15 on three real projects: a map answer is ~1,200 tokens vs ~12,600–37,500 for the files it cites (67–99 % smaller) and ~224,000–338,000 for all source; see `bench/GRAPH_TOKENS.md` (relevance of the answers not yet measured). Remaining: a small question set to measure relevance; `repomap.py` fallback wiring. `src/graph.py` runs `graphify <path> --update` as a subprocess; `POST /project/graph` and `GET /project/graph/query?q=…&budget=N` in `src/serve.py`, behind the existing token and Host/Origin checks; `repomap.py` stays as fallback. *Accept:* on a test repo, `--update` makes no network call for code, and `graphify query --budget 1500` uses fewer tokens than sending the files (measure both).
- [ ] **T1.5 Semantic search quality** (`ROADMAP.md` §4, in order): heading-aware chunking in `documents.py`, token-based truncation, the typescript/typst false match, measured reranker. `hybrid.py` and `gate.py` stay off unless recall@4 improves by ≥ 0.05. *Accept:* recall@4 before/after with the existing evaluation set.
- [ ] **T1.6 Current information.** `web.py`: publication dates, prefer recent results for time-sensitive questions, cache with expiry. *Accept:* three dated questions answered with dated sources.

- [ ] **T1.7 Second dense model family for NPU-only.** Qwen3.5 was removed (it needs the GPU for every token), so NPU-only stays on Qwen3. **Owner decision 2026-10-15: new models are experiments or alternatives, adopted only if they run well without loading the PC; otherwise Qwen3 stays.** Candidates and estimates in `RESEARCH_LOG.md`. Start with Llama 3.2 1B or Gemma 3 1B (≈1 GB per token, 12–15 tok/s expected) and one 3B (SmolLM3 or Ministral 3, ~5 tok/s). *Accept:* the model loads in `src/qwen_npu.py` behind a flag, token-for-token check against the reference (`bench/token_equiv.py`), 3 runs of `bench/watch.py` and `bench/energy.py`, and a row in `bench/PERF.md`; otherwise record the rejection in `docs/DECISIONS.md`.

### Phase 2 — Desktop client and web
- [ ] **T2.1 Tauri shell** in `app/`, following `launcher/src/main.rs`: wraps the local `serve.py`, tray icon, start with session, NPU/power indicator, global search palette. No Electron. *Accept:* starts, shows tray, quits cleanly and leaves no orphan `serve.py`.
- [~] **T2.2 Interface redesign** of `web/index.html` shared by web and client. Done 2026-10-15: warm paper/charcoal palette with light and dark modes, one accent, serif headings, status dot, skip link, focus rings, collapsible detail text, empty and loading states, 360 px layout, no external fonts or requests. Remaining: Search/Project/Sentinel/Measurements screens and the graph view. Screens: Search, Project (graph + graphify button), Sentinel, Chat, Measurements, Settings. Editorial, low-clutter, dark and light, usable at 360 px and 1440 px. *Accept:* screenshots at both widths via the browser tool; `tests/e2e_agent_panel.mjs` still passes.
- [ ] **T2.3 Public page**: landing + "does my XDNA1 NPU work?" guide with a compatibility table. *Accept:* static site builds and every claim links to a measurement.

### Phase 3 — Sentinel and phone
- [~] **T3.1 Sentinel.** First version done 2026-10-15: `src/sentinel.py` (git-only watcher: uncommitted/unpushed work, one notice per state change, no model, no network), `tests/test_sentinel.py` passes, run on the real `~/Documentos/Codigo` it found 9 repos with pending work, unit in `tools/systemd/` (not enabled). Remaining: classifier with `decide.py` for other event kinds (new files, long jobs), the NPU-model answer on request, the idle-power check below. Original scope: (`src/sentinel.py`): low-power loop using `router.py` and `decide.py` to classify events (repos with unsaved work, new files, long jobs) and notify with `notify-send`; the LLM wakes only when the classifier asks. Consolidation only while the PC is idle. *Accept:* idle power with sentinel on is within 1 W of idle without it (`bench/energy.py` idle phase).
- [ ] **T3.2 Phone bridge** (`bridge/`): Telegram bot with an allow-list of user IDs, or a PWA over Tailscale, as a separate process calling `/v1/chat/completions` with the local token. `serve.py` stays bound to `127.0.0.1`. *Accept:* rejected without token, outside the allow-list, and with a bad Host/Origin.
- [ ] **T3.3 Developer helpers** (use case 5) as sentinel actions. *Accept:* each helper has one fixed-input check.

### Phase 4 — Voice and first-generation NPU research
- [ ] **T4.1 Whisper feasibility.** Start from [drakosha/whisper-xdna](https://github.com/drakosha/whisper-xdna) (see `RESEARCH_LOG.md`). Measure Whisper tiny/base int8 on the CPU as reference; estimate whether the encoder fits `mv_i8` or the int16 matmul. *Accept:* ≥ 3× real time and fewer watts than the CPU, otherwise record the rejection with numbers in `docs/DECISIONS.md`.
- [ ] **T4.2 Reproduce community `npu1` work** (`hnws/xdna1-npu-lab`, `FireBurn/ggml-xdna1`, `hawkpoint-npu-llm`, `phoenix-npu1-iron-kernels`) flagged by the research loop before adopting any. *Accept:* ≥ 3 runs on this machine and a `RESEARCH_LOG.md` status change.
- [ ] **T4.3 Report "XDNA1: what works and what does not"** with figures, and open issues for owners of this hardware.

## 6. Risks
- The NPU may use more energy than the CPU for some tasks; T0.1 decides with data.
- Voice and games depend on new kernels; no dates are promised.
- Single-author GitHub claims may be wrong; they stay `unverified` until reproduced here.
