# Plan: an always-on assistant for the first-generation NPU

Spanish version: [`PLAN.es.md`](PLAN.es.md). Working rules: [`WORKING_RULES.md`](WORKING_RULES.md). Status marks: `[ ]` pending, `[~]` in progress, `[x]` done. Each task has an acceptance check; do not mark it done without running it.

## 1. Direction
The XDNA1 NPU (Ryzen 7040/8040, `npu1`) cannot beat a GPU on speed: decode is DMA-bound at about 22 GB/s and gives 9–20 tok/s with 0.6–2B models (`bench/PERF.md`, `bench/DMA_READ.md`). Its real advantage is **low power and always-on availability without touching the GPU**. The product is therefore a small-model assistant that works in the background, plus a desktop client and a web page that make it usable by anyone who owns this forgotten generation of hardware.

Decisions (owner, 2026-10-14): open the project for real (license is already MIT), first pillar = always-on assistant, desktop client with Tauri, continuous measurement, continuous search of the web and GitHub for new NPU uses, full bilingual documentation for hand-over.

## 2. What exists and what does not
| Exists | Missing |
|---|---|
| Qwen3 0.6B/1.7B on the NPU alone; Qwen3.5 0.8B–4B assisted | Speech (ASR/VAD), vision, measured reranker |
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
- [ ] **T0.1 Baseline.** Run the offline tests and `bench/energy.py --engine npu --model models/qwen3s` (3 runs) with the NPU alone, charger plugged in before boot (see `ROADMAP.md` §2, GPU P5 issue). *Accept:* a row appended to `bench/history.jsonl` and a paragraph in `bench/ENERGY.md`.
- [ ] **T0.2 First run of `tests/eval_answers.py`** (written, never executed). *Accept:* results saved under `bench/` and summarized in `docs/CHANGELOG.md`.
- [ ] **T0.3 Review the uncommitted bridge work in `src/serve.py`** (`/v1` with tools) against `python tests/test_server.py`. The owner commits.

### Phase 1 — Base product and continuous loops
- [~] **T1.1 Continuous measurement.** `bench/watch.py` and `bench/history.jsonl` exist as a skeleton. Remaining: run it twice and confirm two lines are appended; simulate a regression (`--simulate-drop 0.2`) and confirm exit code 2; add the weekly user timer `~/.config/systemd/user/npu-bench.timer`; add a "measure" button and trend chart to the UI. *Accept:* all four checks.
- [~] **T1.2 Continuous research.** `tools/research_scan.py` exists as a skeleton. Remaining: run it twice and confirm the second run adds nothing; confirm only keywords leave the machine; add the weekly timer; classify hits with the NPU model (relevant / not, one line). *Accept:* those checks.
- [ ] **T1.3 NPU-only defaults.** Make Qwen3-0.6B/1.7B (`src/qwen_npu.py`) the default profile; keep the Assisted profile (Qwen3.5 + CUDA) optional. *Accept:* fresh install chats with no NVIDIA present.
- [ ] **T1.4 graphify button.** `src/graph.py` runs `graphify <path> --update` as a subprocess; `POST /project/graph` and `GET /project/graph/query?q=…&budget=N` in `src/serve.py`, behind the existing token and Host/Origin checks; `repomap.py` stays as fallback. *Accept:* on a test repo, `--update` makes no network call for code, and `graphify query --budget 1500` uses fewer tokens than sending the files (measure both).
- [ ] **T1.5 Semantic search quality** (`ROADMAP.md` §4, in order): heading-aware chunking in `documents.py`, token-based truncation, the typescript/typst false match, measured reranker. `hybrid.py` and `gate.py` stay off unless recall@4 improves by ≥ 0.05. *Accept:* recall@4 before/after with the existing evaluation set.
- [ ] **T1.6 Current information.** `web.py`: publication dates, prefer recent results for time-sensitive questions, cache with expiry. *Accept:* three dated questions answered with dated sources.

### Phase 2 — Desktop client and web
- [ ] **T2.1 Tauri shell** in `app/`, following `launcher/src/main.rs`: wraps the local `serve.py`, tray icon, start with session, NPU/power indicator, global search palette. No Electron. *Accept:* starts, shows tray, quits cleanly and leaves no orphan `serve.py`.
- [ ] **T2.2 Interface redesign** of `web/index.html` shared by web and client. Screens: Search, Project (graph + graphify button), Sentinel, Chat, Measurements, Settings. Editorial, low-clutter, dark and light, usable at 360 px and 1440 px. *Accept:* screenshots at both widths via the browser tool; `tests/e2e_agent_panel.mjs` still passes.
- [ ] **T2.3 Public page**: landing + "does my XDNA1 NPU work?" guide with a compatibility table. *Accept:* static site builds and every claim links to a measurement.

### Phase 3 — Sentinel and phone
- [ ] **T3.1 Sentinel** (`src/sentinel.py`): low-power loop using `router.py` and `decide.py` to classify events (repos with unsaved work, new files, long jobs) and notify with `notify-send`; the LLM wakes only when the classifier asks. Consolidation only while the PC is idle. *Accept:* idle power with sentinel on is within 1 W of idle without it (`bench/energy.py` idle phase).
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
