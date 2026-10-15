"""Cross-encoder reranker on the local-docs bank: re-order the top 20 candidates of docs.retrieve and compare hit@4 with the plain order (paired bootstrap by query).
Model: cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 (multilingual), CPU. Rule written before measuring: adopt only if hit@4 improves by >= +0.05 with the 95 % CI above 0.
usage: .venv/bin/python bench/rerank_eval.py [--model DIR] [--pool 20]
Spanish: ¿un reranker cross-encoder mejora recall@4 en la documentación local? Criterio fijado antes: +0.05 con IC95 % por encima de 0."""
import argparse, os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "bench"))
import numpy as np, torch, docs, search, retrieval_bank
from transformers import AutoModelForSequenceClassification, AutoTokenizer
ap = argparse.ArgumentParser(); ap.add_argument("--model", default=os.path.expanduser("~/.cache/npu-lab/rerank-mmarco")); ap.add_argument("--pool", type=int, default=20); a = ap.parse_args()
torch.set_num_threads(4)
tok, ce = AutoTokenizer.from_pretrained(a.model), AutoModelForSequenceClassification.from_pretrained(a.model).eval()
emb, idx = search.make_model("cpu"), docs.load()
pos = [b for b in retrieval_bank.bank() if b["kind"] == "pos"]
base, rer, lat = [], [], []
for b in pos:
    cand = docs.retrieve(emb, b["query"], a.pool, idx, gate=False)
    base.append(any(c["title"].lower() in b["gold"] for c in cand[:4]))
    t = time.perf_counter()
    enc = tok([b["query"]] * len(cand), [f"{c['title']}\n{c['text']}" for c in cand], padding=True, truncation=True, max_length=256, return_tensors="pt")
    with torch.no_grad():
        s = ce(**enc).logits.squeeze(-1).numpy()
    lat.append(time.perf_counter() - t)
    rer.append(any(cand[j]["title"].lower() in b["gold"] for j in np.argsort(-s)[:4]))
base, rer = np.array(base, float), np.array(rer, float)
rng = np.random.default_rng(0); d = rer - base
bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
print(f"n={len(pos)} pool={a.pool}: hit@4 plain {base.mean():.3f}  reranked {rer.mean():.3f}  diff {d.mean():+.3f} 95% CI [{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  rerank latency median {np.median(lat)*1000:.0f} ms")
for sp in ("ajuste", "prueba"):
    m = np.array([b["split"] == sp for b in pos]); print(f"  {sp}: n={m.sum()} plain {base[m].mean():.3f} reranked {rer[m].mean():.3f}")
