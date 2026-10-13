"""NPU_FAST_VOCAB on Qwen3.5 (Phase 2 step 2, batch B; criteria in bench/FASE2.md). One process, one engine, one mode:
  speed   --fast 0|1 : 3 runs x 48 decoded tokens (short context, after one warm-up run), tok/s and ms per token
  quality --fast 0|1 : the 10 fixed prompts, greedy, 80 tokens each (the engine's own stream(): chat template and repetition penalty); stores the texts and token ids.
                       With --fast 0 it also counts how many of the generated tokens are NOT in the fast subset (tokens the reduced head cannot write).
  compare            : reads both quality runs and prints the agreement per prompt.
usage: .venv/bin/python bench/fast_vocab_eval.py speed --model models/qwen35_2b --fast 1      (NPU+GPU: run alone)   [Qwen3.5 rows: measured with the degraded GPU]
Results are appended to bench/fast_vocab_runs.jsonl."""
import argparse, json, os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
ap = argparse.ArgumentParser()
ap.add_argument("mode", choices=["speed", "quality", "compare"])
ap.add_argument("--model", default="models/qwen35_2b")
ap.add_argument("--fast", type=int, default=0)
ap.add_argument("--tokens", type=int, default=48)
ap.add_argument("--runs", type=int, default=3)
ap.add_argument("--new", type=int, default=80)
a = ap.parse_args()
OUT = "bench/fast_vocab_runs.jsonl"
doc = " ".join(f"Section {i}: the NPU streams int8 weights while the GPU keeps the cache of key and value vectors for attention." for i in range(30))
PROMPTS = ["What is the capital of France? Answer in one sentence.",
           "Escribe una función en Python que invierta una cadena y explica cómo funciona.",
           "Lista tres comandos de pacman para instalar, quitar y buscar paquetes.",
           "Explain in two sentences why the sky is blue.",
           f"Read this and summarise it in one sentence: {doc}",
           "Explícame con tus palabras qué es una red neuronal y para qué sirve, en un párrafo corto.",
           "Write a SQL query that returns the three customers with the highest total order amount.",
           "Dame una receta sencilla de arepas colombianas con los pasos numerados.",
           "Give me a JSON object describing a book with title, author, year and two tags.",
           "Hola, ¿cómo estás? Cuéntame algo interesante sobre los pulpos."]

if a.mode == "compare":
    rows = [json.loads(l) for l in open(OUT)]
    full = {r["i"]: r for r in rows if r.get("kind") == "quality" and r["fast"] == 0}
    fast = {r["i"]: r for r in rows if r.get("kind") == "quality" and r["fast"] == 1}
    same = 0; miss = tot = 0
    print("| # | identical | first differing token | per-token agreement | tokens outside the subset | prompt |\n|---:|---|---:|---:|---:|---|")
    for i in sorted(full):
        f, g = full[i], fast.get(i)
        if g is None: continue
        k = next((j for j, (x, y) in enumerate(zip(f["ids"], g["ids"])) if x != y), min(len(f["ids"]), len(g["ids"])))
        eq = f["ids"] == g["ids"]; same += eq
        agree = sum(x == y for x, y in zip(f["ids"], g["ids"])) / max(len(f["ids"]), 1)
        miss += f["outside"]; tot += len(f["ids"])
        print(f"| {i} | {'yes' if eq else 'no'} | {'-' if eq else k} | {agree:.2f} | {f['outside']} / {len(f['ids'])} | {f['prompt'][:50]} |")
    print(f"\nidentical text on {same} of {len(full)} prompts; tokens of the full-head output outside the fast subset: {miss} of {tot} = {100 * miss / max(tot, 1):.2f} %")
    sys.exit()

import numpy as np
import qwen_npu as Q
import qwen35_npu as Q35
m = Q35.load_engine(a.model, gpu=True, gpu_share=0.0, vram_gb=4.5, max_ctx=4096, fast_head=bool(a.fast))
label = f"{os.path.basename(a.model)} fast={a.fast}"
if a.mode == "speed":
    rng = np.random.default_rng(0); res = []
    for run in range(a.runs + 1):
        ids = rng.integers(1000, 100000, 16).tolist(); m.cache_ids = []
        lg = m.prefill(ids, 0); pos = len(ids); t = time.perf_counter()
        for _ in range(a.tokens):
            lg = m.step(int(np.argmax(lg)), pos); pos += 1
        res.append((time.perf_counter() - t) / a.tokens * 1e3)
    res = res[1:]
    print(f"{label}: {np.mean(res):.1f} ± {np.std(res, ddof=1):.1f} ms/token = {1e3 / np.mean(res):.2f} tok/s  (runs {[round(x, 1) for x in res]})  logits width {len(lg)}")
    open(OUT, "a").write(json.dumps(dict(kind="speed", fast=a.fast, model=a.model, ms=res)) + "\n")
else:
    V = m.c["vocab_size"]
    sub = set(Q.fast_vocab(m.tok, V, cache=os.path.expanduser(f"~/.local/share/npu-lab/vocab_fast_qwen35_{V}.npy")).tolist())
    for i, p in enumerate(PROMPTS):
        text = "".join(m.stream(p, a.new))
        ids = m.tok.encode(text, add_special_tokens=False).ids
        outside = sum(1 for x in ids if x not in sub) if a.fast == 0 else 0
        print(f"[{i}] fast={a.fast} {len(ids)} tokens, {outside} outside the subset | {text[:70]!r}", flush=True)
        open(OUT, "a").write(json.dumps(dict(kind="quality", fast=a.fast, model=a.model, i=i, prompt=p, text=text, ids=ids, outside=outside)) + "\n")
