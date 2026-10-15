"""Why does the 2B answer 'None' to every routing question? Two parts.
1) confusion matrix of a stored result file (CPU only, no model):  .venv/bin/python bench/route_confusion.py bench/decide_results/logprob_<model>.jsonl [route|mixer]
2) --probe (NPU+GPU, run alone, NOT run yet): loads the model and, for a few questions, prints the 10 most likely next tokens and the probability of each label under
   variants of the prompt: A) as in decide.py (thinking switched off with an empty <think> block), B) without the empty think block, C) the label list also in the system prompt
   as 'Answer: <label>', D) letters A-D instead of words.  The question it answers: is the bias a property of the model (it prefers 'None' as a first word) or of the template/tokens?
   .venv/bin/python bench/route_confusion.py --probe --model models/qwen3 [--n 12]"""
import json, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); os.chdir(ROOT)


def confusion(path, task):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("task") == task and r.get("split") == "test" and r.get("pad", 0) == 0 and "probs" in r]
    opts = list(rows[0]["probs"]); M = {a: {b: 0 for b in opts} for a in opts}
    for r in rows:
        M[r["label"]][max(r["probs"], key=r["probs"].get)] += 1
    print(f"## {os.path.basename(path)} / {task} (test split, n={len(rows)}); rows = true label, columns = predicted\n")
    print("| true \\ pred | " + " | ".join(opts) + " | mean p(true) |\n|---|" + "---:|" * (len(opts) + 1))
    for a in opts:
        ps = [r["probs"][a] for r in rows if r["label"] == a]
        print(f"| {a} | " + " | ".join(str(M[a][b]) for b in opts) + f" | {sum(ps) / max(len(ps), 1):.3f} |")
    mean = {o: sum(r["probs"][o] for r in rows) / len(rows) for o in opts}
    print("\nmean probability given to each option over all questions: " + ", ".join(f"{o} {v:.3f}" for o, v in mean.items()))


def probe(model, n):
    import numpy as np, decide, qwen_npu as Q35
    eng = Q35.Qwen(model, gpu=True, vram_gb=4.5, max_ctx=4096)
    data = [json.loads(l) for l in open("bench/decide_data/route.jsonl")]
    data = [r for r in data if r["split"] in ("val", "test")]
    seen = {}
    for r in data:
        seen.setdefault(r["label"], []).append(r)
    pick = [x for i in range(max(1, n // 4)) for lab in ("NONE", "DOCS", "WEB", "HISTORY") for x in seen[lab][i:i + 1]]
    sys_a = decide.SYSTEM_ROUTE
    variants = {
        "A decide.py": dict(system=sys_a, labels=["None", "Docs", "Web", "History"], think=True),
        "B no empty think": dict(system=sys_a, labels=["None", "Docs", "Web", "History"], think=False),
        "D letters": dict(system=sys_a.replace("None:", "A:").replace("Docs:", "B:").replace("Web:", "C:").replace("History:", "D:"), labels=["A", "B", "C", "D"], think=True),
    }
    for name, v in variants.items():
        ok = 0; pred = {}
        for r in pick:
            txt = (f"<|im_start|>system\n{v['system']}<|im_end|>\n<|im_start|>user\nQuestion: {r['state']}\nReply with one word: {', '.join(v['labels'])}.<|im_end|>\n<|im_start|>assistant\n"
                   + ("<think>\n\n</think>\n\n" if v["think"] else ""))
            ids = eng.tok.encode(txt, add_special_tokens=False).ids
            eng.cache_ids = []; lg = np.asarray(eng.prefill(ids), np.float64)
            top = np.argsort(lg)[::-1][:10]; z = np.exp(lg - lg.max()); z /= z.sum()
            first = [eng.tok.encode(l, add_special_tokens=False).ids[0] for l in v["labels"]]
            p = np.array([z[t] for t in first]); p /= p.sum()
            guess = ["NONE", "DOCS", "WEB", "HISTORY"][int(p.argmax())]; ok += guess == r["label"]; pred[guess] = pred.get(guess, 0) + 1
            print(f"[{name}] {r['label']:7} -> {guess:7} p={np.round(p, 3).tolist()}  top tokens: {[eng.tok.decode([int(t)]) for t in top[:6]]}  | {r['state'][:50]}")
        print(f"== {name}: {ok}/{len(pick)} correct, predictions {pred}\n")


if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe(sys.argv[sys.argv.index("--model") + 1] if "--model" in sys.argv else "models/qwen3", int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 12)
    else:
        confusion(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "route")
