"""Theoretical bytes per token against what is measured (Phase 2 step 1). In ONE process, for one model:
 - the int8 bytes the NPU really streams per decoded token, counted from the NpuLin objects the engine builds (padding included), and the number of NPU calls;
 - the decode time split (PROF, same as src/qwen_npu.py --profile): NPU wait, submit, host prep, post, CPU glue, step total;
 - the DMA floor: bytes / BW with BW = 22.3 GB/s (slope of the 4-column pure-read sweep, bench/DMA_READ.md; 4 columns) and, as an upper bound, + 0.080 ms per call
   (intercept of the same fit: a synchronous call; the engine submits the calls of one matrix together, so the real overhead is between the two).
usage: .venv/bin/python bench/bytes_floor.py --model models/qwen3 [--engine npu|assisted] [--share 0.5] [--tokens 48] [--runs 3]   (NPU, and the GPU for assisted: run alone)
Prints one markdown row and appends a JSON line to bench/bytes_floor.jsonl."""
import argparse, json, os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--engine", default="assisted", choices=["assisted", "npu"])
ap.add_argument("--share", type=float, default=0.0)
ap.add_argument("--tokens", type=int, default=48)
ap.add_argument("--runs", type=int, default=3)
a = ap.parse_args()
if a.engine == "npu":
    os.environ["NPU_GPU"] = "0"
import numpy as np
import qwen_npu as Q

BW, CALL_MS = 22.3e9, 0.080
made = []
_init = Q.NpuLin.__init__
def init(self, *args, **kw):
    _init(self, *args, **kw); made.append(self)
Q.NpuLin.__init__ = init
if a.engine == "npu":
    m = Q.Qwen(a.model, gpu=False)
else:
    import qwen_npu as Q35
    m = Q35.Qwen(a.model, gpu=True, vram_gb=4.5, max_ctx=4096)
K0 = m.npu.K
bytes_tok = sum(mm * K0 * len(pieces) for lin in made for (mm, s, bi, n, pieces) in lin.chunks)
calls_tok = sum(len(pieces) for lin in made for (mm, s, bi, n, pieces) in lin.chunks)
rng = np.random.default_rng(0)
rows = []
for run in range(a.runs + 1):                                   # first run = warm-up, dropped
    Q.PROF = {}
    ids = rng.integers(1000, 100000, 16).tolist(); m.cache_ids = []
    logits = m.prefill(ids, 0); Q.PROF = {}; pos = len(ids); t = time.perf_counter()
    for _ in range(a.tokens):
        logits = m.step(int(np.argmax(logits)), pos); pos += 1
    wall = (time.perf_counter() - t) / a.tokens * 1e3
    P = Q.PROF; n = P.get("steps", a.tokens)                    # the engine may not count steps itself: the loop above ran a.tokens
    parts = [v for v in P.values() if isinstance(v, dict) and "wait" in v]
    g = lambda k: sum(v[k] for v in parts) / n * 1e3
    rows.append(dict(wall=wall, step=P["step_total"] / n * 1e3 if "step_total" in P else wall, wait=g("wait"), submit=g("submit"), prep=g("host_prep"), post=g("post"), calls=sum(v["calls"] for v in parts) / n, disp=sum(v["dispatches"] for v in parts) / n))
rows = rows[1:]
ms = lambda k: (np.mean([r[k] for r in rows]), np.std([r[k] for r in rows], ddof=1))
floor_lo = bytes_tok / BW * 1e3; floor_hi = floor_lo + calls_tok * CALL_MS
label = f"{os.path.basename(a.model)} {a.engine}" + (f" share {a.share:.0%}" if a.engine == "assisted" and a.share else "")
w, s = ms("wait"), ms("step")
print(f"| {label} | {bytes_tok / 1e6:.0f} | {calls_tok} | {floor_lo:.1f} - {floor_hi:.1f} | {w[0]:.1f} ± {w[1]:.1f} | {w[0] / floor_lo:.2f} | {bytes_tok / (w[0] / 1e3) / 1e9:.1f} | "
      f"{ms('submit')[0] + ms('prep')[0] + ms('post')[0]:.1f} | {s[0] - w[0] - ms('submit')[0] - ms('prep')[0] - ms('post')[0]:.1f} | {s[0]:.1f} ± {s[1]:.1f} | {1e3 / ms('wall')[0]:.2f} |", flush=True)
open("bench/bytes_floor.jsonl", "a").write(json.dumps(dict(label=label, model=a.model, engine=a.engine, share=a.share, bytes=bytes_tok, calls=calls_tok, floor_lo_ms=floor_lo, floor_hi_ms=floor_hi, runs=rows, K0=K0)) + "\n")
