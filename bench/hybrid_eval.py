"""Lote 1 (bench/FASE56.md): baseline `docs.retrieve` vs `hybrid.retrieve` on the retrieval bank. CPU only (MiniLM CPU, numpy, scipy).
  python bench/hybrid_eval.py ajuste    # grid on the tuning split only -> bench/.hybrid_choice.json
  python bench/hybrid_eval.py prueba    # test split, ONCE, with the chosen parameters"""
import itertools, json, os, resource, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "bench")]
import numpy as np
import docs, hybrid, search
from retrieval_bank import bank

CACHE, CHOICE = os.path.join(ROOT, "bench/.hybrid_qemb.npz"), os.path.join(ROOT, "bench/.hybrid_choice.json")
phase = sys.argv[1]
B = [x for x in bank() if x["split"] == phase]
idx = docs.load()


class Cached:                      # same query embedding for every method: only the ranking differs
    def __init__(self, qs):
        z = np.load(CACHE, allow_pickle=True) if os.path.exists(CACHE) else None
        self.d = dict(zip(z["q"], z["e"])) if z is not None else {}
        miss = [q for q in qs if q not in self.d]
        if miss:
            m = search.make_model("cpu")
            for q, e in zip(miss, m.encode(miss)):
                self.d[q] = e
            np.savez(CACHE, q=np.array(list(self.d)), e=np.array(list(self.d.values())))

    def encode(self, qs):
        return np.array([self.d[q] for q in qs])


model = Cached([x["query"] for x in bank()])


def run(fn):
    """-> per-query dict(hit1, hit4, rr, inj, ms)."""
    out = []
    for x in B:
        t = time.perf_counter(); r = fn(x["query"]); ms = (time.perf_counter() - t) * 1e3
        titles = [h["title"].lower() for h in r]
        rank = next((i for i, t_ in enumerate(titles) if t_ in x["gold"]), None)
        out.append(dict(kind=x["kind"], hard=x["hard"], hit1=rank == 0, hit4=rank is not None, rr=1 / (rank + 1) if rank is not None else 0.0, inj=len(r) > 0, ms=ms))
    return out


def summ(o):
    pos, neg = [r for r in o if r["kind"] == "pos"], [r for r in o if r["kind"] == "neg"]
    f = lambda rs, k: float(np.mean([r[k] for r in rs])) if rs else float("nan")
    ms = [r["ms"] for r in o]
    return dict(n_pos=len(pos), n_neg=len(neg), r1=f(pos, "hit1"), r4=f(pos, "hit4"), mrr=f(pos, "rr"), inj=f(neg, "inj"),
                inj_hard=f([r for r in neg if r["hard"]], "inj"), inj_easy=f([r for r in neg if not r["hard"]], "inj"), ms_med=float(np.median(ms)), ms_p95=float(np.percentile(ms, 95)))


def show(name, s):
    print(f"{name:34s} r@1 {s['r1']:.3f} r@4 {s['r4']:.3f} MRR {s['mrr']:.3f} | inj {s['inj']:.3f} (hard {s['inj_hard']:.3f}, easy {s['inj_easy']:.3f}) | {s['ms_med']:.0f}/{s['ms_p95']:.0f} ms med/p95")


rss0 = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
base = run(lambda q: docs.retrieve(model, q, k=4, idx=idx))
sb = summ(base); show("baseline docs.retrieve", sb)
bms = {}
for stem in (False, True):
    t = time.perf_counter(); bms[stem] = hybrid.build(idx, stem=stem)
    M = bms[stem]["M"]; print(f"bm25 stem={stem}: vocab {len(bms[stem]['vocab'])}, {M.nnz} nnz, {(M.data.nbytes + M.indices.nbytes + M.indptr.nbytes) / 1e6:.0f} MB, build {time.perf_counter() - t:.1f}s")
mk = lambda c: (lambda q: hybrid.retrieve(model, q, k=4, idx=idx, bm=bms[c[0]], fusion=c[1], w=c[2], gate_bm25=c[3]))

if phase == "ajuste":
    grid = list(itertools.product((False, True), (("wsum", 0.15), ("wsum", 0.3), ("rrf", 0)), (None, 8.0, 14.0)))
    res = []
    for stem, (fu, w), g in grid:
        c = (stem, fu, w, g); s = summ(run(mk(c))); res.append((c, s)); show(f"stem={int(stem)} {fu} w={w} gate={g}", s)
    ok = [(c, s) for c, s in res if s["inj"] <= sb["inj"] + 1e-9 and s["inj_hard"] <= sb["inj_hard"] + 1e-9]
    pick = max(ok, key=lambda cs: (cs[1]["r4"], cs[1]["mrr"], -cs[0][2])) if ok else None
    json.dump(dict(baseline=sb, choice=pick[0] if pick else None, summary=pick[1] if pick else None), open(CHOICE, "w"))
    print("CHOICE:", pick[0] if pick else "none meets injection <= baseline")
else:
    ch = json.load(open(CHOICE))["choice"]
    assert ch, "no candidate chosen on ajuste"
    cand = run(mk(tuple(ch))); sc = summ(cand); show(f"candidate {ch}", sc)
    rng = np.random.default_rng(0)

    def boot(key, kind):
        d = np.array([c[key] - b[key] for b, c in zip(base, cand) if b["kind"] == kind], float)
        m = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(10000)]
        return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
    for lab, key, kind in (("recall@4", "hit4", "pos"), ("recall@1", "hit1", "pos"), ("MRR", "rr", "pos"), ("injection (neg)", "inj", "neg")):
        m, lo, hi = boot(key, kind); print(f"diff {lab}: {m:+.3f}  IC95 [{lo:+.3f}, {hi:+.3f}]")
    hard = lambda o: [r["inj"] for r in o if r["kind"] == "neg" and r["hard"]]
    print("injection hard: baseline", np.mean(hard(base)), "candidate", np.mean(hard(cand)))
    print(f"added latency median {sc['ms_med'] - sb['ms_med']:+.1f} ms, p95 {sc['ms_p95'] - sb['ms_p95']:+.1f} ms")
    json.dump(dict(choice=ch, baseline=sb, candidate=sc), open(os.path.join(ROOT, "bench/hybrid_result.json"), "w"), indent=1)
print(f"RSS peak {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB (start {rss0:.0f} MB)")
