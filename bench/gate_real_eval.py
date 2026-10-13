"""Injection gate on REAL technical questions (local file, never committed): bench/real_questions.jsonl, one line each:
  {"q": "text as asked", "inject": true|false, "gold": ["page_title", ...], "note": "optional"}     (gold only when inject is true)
Uses the model already fitted on bank v2 (bench/gate_model.json, A and B), refits NOTHING, and compares with the baseline docs.retrieve. CPU only.
usage: .venv/bin/python bench/gate_real_eval.py [file]       Wilson 95 % intervals (n is small); the margins of bench/FASE56.md are not moved."""
import json, math, os, sys
os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "bench")]
import numpy as np
import docs, hybrid, gate, search

path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "bench/real_questions.jsonl")
Q = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
assert all("q" in x and isinstance(x.get("inject"), bool) for x in Q), "each line needs q and inject"
assert all(x["gold"] for x in Q if x["inject"]), "inject:true needs a non-empty gold list"
idx, models = docs.load(), json.load(open(os.path.join(ROOT, "bench/gate_model.json")))
bm, m = hybrid.build(idx, stem=True), search.make_model("cpu")
emb = m.encode([x["q"] for x in Q])


def wilson(k, n, z=1.96):
    if n == 0: return (float("nan"),) * 3
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


class C:
    def encode(self, qs): return np.array([emb[[x["q"] for x in Q].index(q)] for q in qs])


def view(f, name): return f if name == "A" else np.array([f[0], f[1], f[2] + f[3], f[4], f[5], f[6]])


rows = {"baseline": [], "A": [], "B": []}
for x, e in zip(Q, emb):
    base = docs.retrieve(C(), x["q"], k=4, idx=idx); rows["baseline"].append(base)
    f = gate.features(idx, bm, e, x["q"])
    for n in ("A", "B"):
        rows[n].append(docs.retrieve(C(), x["q"], k=4, idx=idx, gate=False) if gate.prob(models[n], view(f, n)) >= models[n]["t"] else [])
neg, pos = [i for i, x in enumerate(Q) if not x["inject"]], [i for i, x in enumerate(Q) if x["inject"]]
print(f"{len(Q)} questions: {len(pos)} should inject, {len(neg)} should not")
for name, r in rows.items():
    k = sum(len(r[i]) > 0 for i in neg); h = sum(any(t["title"].lower() in [g.lower() for g in Q[i]["gold"]] for t in r[i]) for i in pos)
    p, lo, hi = wilson(k, len(neg)); q, lo2, hi2 = wilson(h, len(pos))
    print(f"{name:9s} injection on should-not {k}/{len(neg)} = {p:.3f} [{lo:.3f}, {hi:.3f}]   recall@4 on should-inject {h}/{len(pos)} = {q:.3f} [{lo2:.3f}, {hi2:.3f}]")
print("\nquestions where the gate A injects although it should not:"); [print("  -", Q[i]["q"][:100]) for i in neg if rows["A"][i]]
print("questions where the gate A cuts a hit the baseline had:"); [print("  -", Q[i]["q"][:100]) for i in pos if not rows["A"][i] and any(t["title"].lower() in [g.lower() for g in Q[i]["gold"]] for t in rows["baseline"][i])]
