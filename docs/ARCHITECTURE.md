# Architecture

Spanish version: [`ARCHITECTURE.es.md`](ARCHITECTURE.es.md). Labels: `[measured]`, `[read in code]`.

## Processing split
- **NPU** (XDNA1, 4 columns): decodes every token with int8 matrix-vector kernels (`src/kernels/`, built by `scripts/build_*.py`). One xclbin and one 4-column hardware context per model. Decode is DMA-bound: 6.6 / 12.3 / 21.8 GB/s with 1 / 2 / 4 columns `[measured, bench/DMA_READ.md]`. MiniLM embeddings run their dense layers here.
- **GPU** (optional, NVIDIA/CUDA): reads long prompts, holds KV cache and DeltaNet state, can compute a 0–75 % share of each matrix, builds big embedding indexes.
- **CPU**: glue, attention for the NPU-only prefill (single-thread numpy), tokenization, search, server.

Profiles: **NPU only** (Qwen3-0.6B ≈ 19.8 tok/s, 1.7B ≈ 10.1 tok/s, 4k context) and **Assisted** (Qwen3.5 0.8B/2B/4B with the GPU) `[measured, bench/PERF.md]`.

## Request flow
```
client (web/index.html, launcher, Tauri app, VS Code via /v1, phone bridge)
   -> src/serve.py  (127.0.0.1 only; bearer token + one-time login code; Host/Origin checks)
        -> router.py / decide.py   choose: docs | web | history | plain chat
        -> search.py / documents.py / docs.py / web.py   retrieve excerpts (MiniLM on NPU, GPU for big indexes)
        -> statecache.py / pins.py   reuse prefix state, keep fact-bearing lines verbatim through compaction
        -> qwen_npu.py | qwen35_npu.py (+gpu_qwen35.py)   generate tokens
        <- streamed reply, sources shown to the user
```

## Server endpoints (`src/serve.py`, `[read in code]`)
GET: `/status`, `/v1/models`, `/routing`, `/documents`, `/rules`, `/facts`, `/chats`, `/doctor`, `/settings`.
POST: `/login`, `/mode`, `/profile`, `/rules`, `/documents/upload`, `/documents/delete`, `/agent/run` (one action; writes only with `confirm=true`), `/project/apply`, `/facts`, `/chats/save`, `/chats/load`, `/routing`, `/settings`, `/settings/models`, `/settings/test_web`, `/chat`, `/v1/chat/completions`, `/search`, `/index`, `/index/stop`.
Project map (`src/graph.py`, graphify, code only): `GET /project/graph` (status), `POST /project/graph` (build or refresh in the background), `POST /project/graph/query` (`{q, budget}`).

## Data on disk
| Path | Content |
|---|---|
| `~/.config/npu-lab/token`, `providers.json` (0600), `rules.json`, settings | Auth token, provider keys (write-only from the UI), allow/deny rules and limits |
| `~/.local/share/npu-lab/` | Document and docs indexes, chats, facts, state cache, `trash/` (recoverable deletes) |
| `models/` (git-ignored) | Model weights, downloaded separately (Apache-2.0 models) |
| `build/` (git-ignored) | Compiled xclbins and instruction streams |
| `bench/history.jsonl` | One line per continuous-measurement run |
| `docs/RESEARCH_LOG.md` | External findings and their verification status |

## Planned components
```
sentinel.py (today: git status/ahead only, no model)  --events (git status, new files, long jobs)--> decide.py classifier
      |                                                         | asks only if needed
      +--notify-send / UI feed <---------------- small model (NPU)

bridge/ (separate process)  phone -> Telegram bot / Tailscale PWA -> allow-list -> POST /v1/chat/completions (local token)

graph.py  -> subprocess `graphify <path> --update` -> graphify-out/ -> `graphify query --budget N` -> short context
```

## Security model
Server bound to loopback only; token + Host/Origin checks; `web.py` refuses private/loopback targets; cloud providers are opt-in, per message and labelled; indexed files are never sent to a provider; web text is untrusted (the model has no write tools without user confirmation). The phone bridge and the research loop must not widen this: the bridge is a separate authenticated client, and the research loop sends only search keywords.
