"""How much of NPU-only prefill is the numpy attention? Runs the attention block of Qwen.prefill (src/qwen_npu.py, same code, QBLOCK from the module)
alone on random data with the model's shapes, no NPU, one thread like the engine, and prints seconds for all layers next to tokens/s.
usage: .venv/bin/python bench/prefill_attn.py [--model models/qwen3s] [--tokens 256,1024,2048] [--reps 3]"""
import argparse, json, os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/qwen3s")
ap.add_argument("--tokens", default="256,1024,2048")
ap.add_argument("--reps", type=int, default=3)
a = ap.parse_args()
c = json.load(open(os.path.join(ROOT, a.model, "config.json")))
NH, NKV, HD, NL = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"], c["num_hidden_layers"]
rep = NH // NKV
QBLOCK = int(os.environ.get("QBLOCK", 0)) or __import__("qwen_npu").QBLOCK     # imports the engine module only for its constant (no NPU is opened)


def attention(q, kc, vc, T):                      # copy of the loop in Qwen.prefill, start = 0
    mask = np.where(np.arange(T)[None, :] > np.arange(T)[:, None], -1e30, 0).astype(np.float32)
    att = np.empty((T, NKV, rep, HD), np.float32)
    for s in range(0, T, QBLOCK):
        e, Sb = min(s + QBLOCK, T), min(s + QBLOCK, T)
        Kt, V = kc[:, :Sb].transpose(0, 2, 1), vc[:, :Sb]
        qh = q[s:e].reshape(e - s, NKV, rep, HD).transpose(1, 2, 0, 3).reshape(NKV, rep * (e - s), HD)
        sc = (qh @ Kt / np.sqrt(HD)).reshape(NKV, rep, e - s, Sb) + mask[s:e, :Sb]
        sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
        att[s:e] = (sc.reshape(NKV, rep * (e - s), Sb) @ V).reshape(NKV, rep, e - s, HD).transpose(2, 0, 1, 3)
    return att


rng = np.random.default_rng(0)
print(f"## {a.model}  NH={NH} NKV={NKV} HD={HD} layers={NL} QBLOCK={QBLOCK}\n| tokens (rep) | attention, all layers (s) | one layer (ms) |\n|---:|---:|---:|")
for T in map(int, a.tokens.split(",")):
    q = rng.standard_normal((T, NH, HD), dtype=np.float32)
    kc = rng.standard_normal((NKV, T, HD), dtype=np.float32); vc = rng.standard_normal((NKV, T, HD), dtype=np.float32)
    attention(q, kc, vc, T)                       # warm-up
    for r in range(a.reps):
        t = time.perf_counter(); attention(q, kc, vc, T); one = time.perf_counter() - t
        print(f"| {T} ({r}) | {one * NL:.2f} | {one * 1e3:.1f} |", flush=True)
