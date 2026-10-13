"""Decode cost per token as the context grows (NPU only): where does the CPU attention start to hurt?
usage: .venv/bin/python bench/attn_vs_ctx.py [--model models/qwen3s] [--ctx 16,256,1024,2048,3500]"""
import argparse, os, sys, time
os.environ["NPU_GPU"] = "0"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
import qwen_npu as Q

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/qwen3s")
ap.add_argument("--ctx", default="16,256,1024,2048,3500")
ap.add_argument("--steps", type=int, default=24)
a = ap.parse_args()
m = Q.Qwen(a.model, gpu=False)
rng = np.random.default_rng(0)
print("| context | decode tok/s | step ms | attention ms | NPU wait ms | other ms |\n|---:|---:|---:|---:|---:|---:|")
for n in map(int, a.ctx.split(",")):
    ids = rng.integers(1000, 100000, n).tolist()
    m.cache_ids = []; logits = m.prefill(ids, 0)
    Q.PROF = {}
    pos = n
    for _ in range(a.steps):
        logits = m.step(int(np.argmax(logits)), pos); pos += 1
    P, k = Q.PROF, a.steps
    tot = P["step_total"] / k * 1e3; att = P["attention"] / k * 1e3
    wait = sum(P[p]["wait"] for p in ("qkv", "o", "gate_up", "down", "lm_head")) / k * 1e3
    Q.PROF = None
    print(f"| {n} | {1e3 / tot:.2f} | {tot:.1f} | {att:.1f} | {wait:.1f} | {tot - att - wait:.1f} |", flush=True)
