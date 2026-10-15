"""Local-docs retrieval on the labelled bank (bench/retrieval_bank.py): hit@k on positives, and how often an excerpt is injected for a negative.
usage: .venv/bin/python bench/docs_gate_eval.py [--split ajuste|prueba|all] [--k 4]     (CPU only: loads the embedder on the CPU, no NPU)
Prints, per split: positives answered (gold title in the top k), mean recall, negatives with a spurious excerpt (hard = names that are also commands).
Spanish: calidad de la recuperación de documentación local: aciertos en positivos y excerptos falsos en negativos."""
import argparse, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "bench"))
import docs, gate, hybrid, search, retrieval_bank

ap = argparse.ArgumentParser(); ap.add_argument("--split", default="all"); ap.add_argument("--k", type=int, default=4); ap.add_argument("--gate", action="store_true", help="apply src/gate.py first, as serve.py does"); a = ap.parse_args()
model, idx = search.make_model("cpu"), docs.load()
bm = hybrid.build(idx, stem=True) if a.gate else None


def get(q):
    if a.gate and not gate.allow(idx, bm, model.encode([q])[0], q):
        return []
    return docs.retrieve(model, q, a.k, idx)

rows = [b for b in retrieval_bank.bank() if a.split in ("all", b["split"])]
pos = [b for b in rows if b["kind"] == "pos"]; neg = [b for b in rows if b["kind"] == "neg"]
hit = 0
for b in pos:
    r = get(b["query"])
    hit += any(x["title"].lower() in b["gold"] for x in r)
fp, fph, nh, nn = 0, 0, 0, 0
for b in neg:
    r = get(b["query"])
    bad = bool(r); fp += bad
    if b["hard"]: nh += 1; fph += bad; (print("  hard FP:", b["query"][:60], "->", [x["title"] for x in r][:3]) if bad else None)
    else: nn += 1
print(f"gate={a.gate} split={a.split} k={a.k}: positives {hit}/{len(pos)} = {hit/max(1,len(pos)):.3f}   negatives with an excerpt {fp}/{len(neg)} (hard {fph}/{nh}, easy {fp-fph}/{nn})")
