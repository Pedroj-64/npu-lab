"""Where does one decoded token go in the Qwen3.5 engine? (Phase 2 step 2, batch A; bench/FASE2.md). An instrumented COPY of Qwen35.step (src/qwen35_npu.py, same maths, clocks around
each part, nothing changed in src/) splits the step into: NPU wait / NPU call overhead (from the engine's own PROF) / GPU steps of the DeltaNet and attention layers (they block until
the result is back as NumPy, so the clock is honest) / the output head / CPU glue. A microbenchmark of an EMPTY CUDA-graph round trip (copy in, replay, synchronise, copy out) gives the
fixed cost of one GPU step; what remains of the measured GPU step is compute plus the real data movement.
usage: .venv/bin/python bench/step_breakdown.py --model models/qwen35_2b [--tokens 48] [--runs 3]    (NPU + GPU: run alone)   [measured with the degraded GPU]"""
import argparse, json, os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/qwen35_2b")
ap.add_argument("--tokens", type=int, default=48)
ap.add_argument("--runs", type=int, default=3)
a = ap.parse_args()
import numpy as np, torch
import qwen_npu as Q
import qwen35_npu as Q35

m = Q35.load_engine(a.model, gpu=True, gpu_share=0.0, vram_gb=4.5, max_ctx=4096)
T = {}
pc = time.perf_counter


def add(k, t0):
    T[k] = T.get(k, 0.0) + (pc() - t0)


def step_timed(self, tok, pos):                                   # copy of Qwen35.step with clocks
    d, eps = self.d, self.eps
    t = pc(); x = Q.f32(self.embed_raw[tok]).copy(); add("cpu glue", t)
    for i, L in enumerate(self.layers):
        t = pc(); h = Q35.zrms_np(x, L["ln1"], eps); add("cpu glue", t)
        if L["type"] == "linear_attention":
            t = pc(); y = L["inp"](h)[:d["inp"]]; add("npu calls", t)
            t = pc(); core = self.g.linear_step(i, y); add("gpu linear step", t)
            t = pc(); o = L["out"](core); add("npu calls", t)
            t = pc(); x = x + o; add("cpu glue", t)
        else:
            t = pc(); qkvg = L["qkv"](h); add("npu calls", t)
            n, nk = d["NH"] * d["HD"] * 2, d["NKV"] * d["HD"]
            t = pc(); att = self.g.attn_step(i, qkvg[:n], qkvg[n:n + nk], qkvg[n + nk:], pos); add("gpu attention step", t)
            t = pc(); o = L["o"](att); add("npu calls", t)
            t = pc(); x = x + o; add("cpu glue", t)
        t = pc(); hh = Q35.zrms_np(x, L["ln2"], eps); add("cpu glue", t)
        t = pc(); gu = L["gu"](hh); add("npu calls", t)
        t = pc(); g_, u_ = gu[:d["FF"]], gu[d["FF"]:]; act = (g_ / (1 + np.exp(-np.clip(g_, -60, 60)))) * u_; add("cpu glue", t)
        t = pc(); dn = L["down"](act); add("npu calls", t)
        t = pc(); x = x + dn; add("cpu glue", t)
    t = pc(); hn = Q35.zrms_np(x, self.norm, eps); add("cpu glue", t)
    t = pc(); out = self.lm(hn); add("output head (npu)", t)
    return out


orig_step = Q35.Qwen35.step
rng = np.random.default_rng(0)
ids0 = rng.integers(1000, 100000, 16).tolist()


def run_steps(stepfn, n, keep=False):
    m.cache_ids = []; lg = m.prefill(ids0, 0); pos = len(ids0); outs = []
    for _ in range(n):
        lg = stepfn(m, int(np.argmax(lg)), pos); pos += 1
        if keep: outs.append(np.array(lg, copy=True))
    return outs


ref = run_steps(orig_step, 5, True); got = run_steps(step_timed, 5, True)
same = all(np.array_equal(x, y) for x, y in zip(ref, got))
print("instrumented step returns the same logits on 5 fixed steps:", same, flush=True)

# empty GPU round trip: what one linear_step/attn_step costs even if it computed nothing
buf_in = torch.zeros(4096, dtype=torch.float32).pin_memory(); buf_out = torch.zeros(4096, dtype=torch.float32).pin_memory(); gbuf = torch.zeros(4096, device="cuda")
gr = torch.cuda.CUDAGraph()
with torch.cuda.graph(gr):
    gbuf.add_(1.0)
def empty_roundtrip():
    buf_in.numpy()[:2048] = 1.0
    gbuf[:2048].copy_(buf_in[:2048], non_blocking=True); gr.replay(); buf_out[:2048].copy_(gbuf[:2048], non_blocking=True)
    torch.cuda.current_stream().synchronize(); return buf_out[:1024].numpy().copy()
for _ in range(50): empty_roundtrip()
t = pc()
for _ in range(500): empty_roundtrip()
empty_us = (pc() - t) / 500 * 1e6

rows = []
for r in range(a.runs + 1):
    T.clear(); Q.PROF = {}
    m.cache_ids = []; lg = m.prefill(ids0, 0); Q.PROF = {}; T.clear(); pos = len(ids0); t0 = pc()
    for _ in range(a.tokens):
        lg = step_timed(m, int(np.argmax(lg)), pos); pos += 1
    wall = (pc() - t0) / a.tokens * 1e3
    parts = [v for v in Q.PROF.values() if isinstance(v, dict) and "wait" in v]
    wait = sum(v["wait"] for v in parts) / a.tokens * 1e3
    rows.append(dict(wall=wall, wait=wait, **{k: v / a.tokens * 1e3 for k, v in T.items()}))
rows = rows[1:]
mean = lambda k: np.mean([r[k] for r in rows]); sd = lambda k: np.std([r[k] for r in rows], ddof=1)
nlin = sum(1 for L in m.layers if L["type"] == "linear_attention"); natt = len(m.layers) - nlin
npu_total = mean("npu calls") + mean("output head (npu)")
gpu_total = mean("gpu linear step") + mean("gpu attention step")
tot = mean("wall"); sumparts = npu_total + gpu_total + mean("cpu glue")
print(f"\n## {os.path.basename(a.model)} [medido, GPU degradada], {a.runs} runs x {a.tokens} tokens; {nlin} DeltaNet layers, {natt} attention layers")
print("| part | ms/token | sd | % of step |\n|---|---:|---:|---:|")
print(f"| NPU: wait for the weight stream | {mean('wait'):.1f} | {sd('wait'):.1f} | {mean('wait') / tot * 100:.0f} % |")
print(f"| NPU: call overhead (prepare, submit, collect; layers + output head) | {npu_total - mean('wait'):.1f} | | {(npu_total - mean('wait')) / tot * 100:.0f} % |")
print(f"| GPU: DeltaNet steps ({nlin} per token) | {mean('gpu linear step'):.1f} | {sd('gpu linear step'):.1f} | {mean('gpu linear step') / tot * 100:.0f} % |")
print(f"| GPU: attention steps ({natt} per token) | {mean('gpu attention step'):.1f} | {sd('gpu attention step'):.1f} | {mean('gpu attention step') / tot * 100:.0f} % |")
print(f"| CPU glue (embedding, norms, SiLU, residuals) | {mean('cpu glue'):.1f} | {sd('cpu glue'):.1f} | {mean('cpu glue') / tot * 100:.0f} % |")
print(f"| sum of parts | {sumparts:.1f} | | {sumparts / tot * 100:.0f} % |\n| step wall (with clocks) | {tot:.1f} | {sd('wall'):.1f} | 100 % |")
print(f"\nempty GPU round trip (copy in, graph replay, sync, copy out): {empty_us:.0f} us -> fixed cost of the {nlin + natt} GPU steps per token: {empty_us * (nlin + natt) / 1e3:.1f} ms")
print(f"per call: DeltaNet step {mean('gpu linear step') / nlin * 1e3:.0f} us, attention step {mean('gpu attention step') / max(natt, 1) * 1e3:.0f} us")

# pure graph time of the DeltaNet layers (replay + synchronise, no copies) and a kernel profile of one token
g = m.g
graphs = [(i, g.graphs.get(i)) for i in range(len(m.layers)) if m.layers[i]["type"] == "linear_attention"]
ok = [(i, gr) for i, gr in graphs if gr]
for i, gr in ok: gr.replay()
torch.cuda.synchronize(); t = pc()
for _ in range(50):
    for i, gr in ok: gr.replay()
torch.cuda.synchronize(); replay_ms = (pc() - t) / 50 * 1e3
print(f"pure replay of the {len(ok)} captured DeltaNet graphs back to back: {replay_ms:.1f} ms per token ({replay_ms / max(len(ok), 1) * 1e3:.0f} us per layer)")
from torch.profiler import profile, ProfilerActivity
m.cache_ids = []; lg = m.prefill(ids0, 0); pos = len(ids0)
for _ in range(3): lg = step_timed(m, int(np.argmax(lg)), pos); pos += 1
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    step_timed(m, int(np.argmax(lg)), pos)
ev = prof.key_averages()
kern = [e for e in ev if e.device_type.name == "CUDA"]
gpu_us = sum(e.self_device_time_total for e in ev if e.device_type.name == "CUDA" or e.self_device_time_total > 0) if False else sum(e.self_device_time_total for e in ev)
ncu = sum(e.count for e in ev if e.device_type.name == "CUDA")
launches = sum(e.count for e in ev if e.key in ("cudaLaunchKernel", "cudaGraphLaunch", "cuLaunchKernel"))
print(f"one token profiled: {ncu} CUDA kernels, summed kernel time {gpu_us / 1e3:.1f} ms, {launches} launch calls (kernel/graph)")
open("bench/step_breakdown.jsonl", "a").write(json.dumps(dict(model=a.model, same_logits=same, empty_us=empty_us, nlin=nlin, natt=natt, replay_ms=replay_ms, kernels=ncu, kernel_ms=gpu_us / 1e3, launches=launches, runs=rows)) + "\n")
