"""Lote 2 (bench/FASE56.md): injection gate on bank v2. CPU only.
  python bench/gate_eval.py ajuste   # features, weights and threshold from the tuning split only -> bench/gate_model.json
  python bench/gate_eval.py prueba   # test split, ONCE, with that model"""
import hashlib, json, os, sys, time
os.environ.setdefault("OMP_NUM_THREADS", "4")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [os.path.join(ROOT, "src"), os.path.join(ROOT, "bench")]
import numpy as np
from scipy.optimize import minimize
import docs, hybrid, gate, search
from retrieval_bank2 import bank

phase = sys.argv[1]
CACHE, MODEL_F = os.path.join(ROOT, "bench/.gate_qemb.npz"), os.path.join(ROOT, "bench/gate_model.json")
B = bank(); idx = docs.load()
z = np.load(CACHE, allow_pickle=True) if os.path.exists(CACHE) else None
Q = dict(zip(z["q"], z["e"])) if z is not None else {}
miss = [x["query"] for x in B if x["query"] not in Q]
if miss:
    m = search.make_model("cpu")
    for i in range(0, len(miss), 64):
        for q, e in zip(miss[i:i + 64], m.encode(miss[i:i + 64])):
            Q[q] = e
    np.savez(CACHE, q=np.array(list(Q)), e=np.array(list(Q.values())))


class Cached:
    def encode(self, qs):
        return np.array([Q[q] for q in qs])


M = Cached()
bm = hybrid.build(idx, stem=True)
HARD = ("neg_word", "neg_tech")


def feats(xs):
    return np.array([gate.features(idx, bm, Q[x["query"]], x["query"]) for x in xs])


def view(X, name):
    """A = the 8 pre-registered features; B = ablation without the usage-cue features (bank templates for pos_named contain 'terminal'/'comando', so cues are a possible artifact)."""
    return X if name == "A" else np.column_stack([X[:, 0], X[:, 1], X[:, 2] + X[:, 3], X[:, 4], X[:, 5], X[:, 6]])


def hit_rank(r, gold):
    t = [h["title"].lower() for h in r]
    return next((i for i, v in enumerate(t) if v in gold), None)


def fit(X, y, w, lam=1.0):
    mu, sd = X.mean(0), X.std(0) + 1e-9; Z = (X - mu) / sd

    def loss(p):
        s = Z @ p[:-1] + p[-1]; return float(np.sum(w * (np.logaddexp(0, s) - y * s)) + lam * p[:-1] @ p[:-1])

    def grad(p):
        s = Z @ p[:-1] + p[-1]; r = w * (1 / (1 + np.exp(-s)) - y); return np.append(Z.T @ r + 2 * lam * p[:-1], r.sum())
    p = minimize(loss, np.zeros(Z.shape[1] + 1), jac=grad, method="L-BFGS-B").x
    return dict(mu=mu.tolist(), sd=sd.tolist(), w=p[:-1].tolist(), b=float(p[-1]))


def weights(xs):
    """each group gets a fixed total weight (set before looking at any result): positives 1 per query, hard/easy negatives normalised to 150 / 100 / 80 per group."""
    tot = {"neg_word": 150, "neg_tech": 100, "neg_easy": 80}
    n = {g: sum(x["group"] == g for x in xs) for g in tot}
    return np.array([1.0 if x["kind"] == "pos" else tot[x["group"]] / n[x["group"]] for x in xs])


if phase == "ajuste":
    A = [x for x in B if x["split"] == "ajuste"]; X0 = feats(A); y = np.array([x["kind"] == "pos" for x in A], float); w = weights(A)
    keys = sorted({x["key"] for x in A}); fold = {k: int(hashlib.md5(k.encode()).hexdigest(), 16) % 5 for k in keys}; f = np.array([fold[x["key"]] for x in A])
    hard = np.array([x["group"] in HARD for x in A]); pos = y == 1; models = {}
    for name in ("A", "B"):
        X = view(X0, name); oof = np.zeros(len(A))
        for k in range(5):
            tr = f != k; mdl = fit(X[tr], y[tr], w[tr]); oof[~tr] = [gate.prob(mdl, r) for r in X[~tr]]
        best = None
        for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
            inj, cov = float((oof[hard] >= t).mean()), float((oof[pos] >= t).mean())
            if inj <= 0.10 and (best is None or cov > best[1]): best = (float(t), cov, inj)
        sem = np.array([x["group"] in ("pos_sem", "pos_old") for x in A])
        print(name, "out-of-fold (t, coverage all pos, coverage sem+old, injection hard):", [(float(t), round(float((oof[pos] >= t).mean()), 3), round(float((oof[sem] >= t).mean()), 3), round(float((oof[hard] >= t).mean()), 3)) for t in (0.2, 0.3, 0.4, 0.5, 0.6, 0.7)])
        final = fit(X, y, w); final["t"] = best[0] if best else 0.5; models[name] = final
        print(name, "chosen t", final["t"], "coverage", None if best is None else round(best[1], 3), "inj", None if best is None else round(best[2], 3), "weights", [round(v, 2) for v in final["w"]])
    json.dump(models, open(MODEL_F, "w"))
else:
    models = json.load(open(MODEL_F)); T = [x for x in B if x["split"] == "prueba"]; X0 = feats(T); model = models["A"]
    PR = {n: np.array([gate.prob(models[n], r) for r in view(X0, n)]) for n in models}; pr = PR["A"]

    def run(fn):
        out = []
        for x in T:
            t0 = time.perf_counter(); r = fn(x); ms = (time.perf_counter() - t0) * 1e3
            rk = hit_rank(r, x["gold"]) if x["kind"] == "pos" else None
            out.append(dict(inj=len(r) > 0, hit4=rk is not None, rr=1 / (rk + 1) if rk is not None else 0.0, ms=ms))
        return out
    base = run(lambda x: docs.retrieve(M, x["query"], k=4, idx=idx))
    rank_plain = lambda x: docs.retrieve(M, x["query"], k=4, idx=idx, gate=False)
    rank_hyb = lambda x: hybrid.retrieve(M, x["query"], k=4, idx=idx, bm=bm, fusion="wsum", w=0.15, gate=False)
    i_of = {id(x): i for i, x in enumerate(T)}

    def gated(fn, name="A"):
        def g(x):
            f = view(gate.features(idx, bm, Q[x["query"]], x["query"])[None], name)[0]; p = gate.prob(models[name], f)
            return fn(x) if p >= models[name]["t"] else []
        return g
    cand = run(gated(rank_plain)); hyb = run(gated(rank_hyb)); candB = run(gated(rank_plain, "B"))

    def table(name, o):
        print(f"\n{name}")
        for grp in ("neg_word", "neg_tech", "neg_easy", "pos_named", "pos_sem", "pos_old"):
            ix = [i for i, x in enumerate(T) if x["group"] == grp]
            if grp.startswith("neg"): print(f"  {grp:10s} n={len(ix):3d} injection {np.mean([o[i]['inj'] for i in ix]):.3f}")
            else: print(f"  {grp:10s} n={len(ix):3d} recall@4 {np.mean([o[i]['hit4'] for i in ix]):.3f} MRR {np.mean([o[i]['rr'] for i in ix]):.3f}")
        ix = [i for i, x in enumerate(T) if x["group"] in HARD]; print(f"  HARD (word+tech) injection {np.mean([o[i]['inj'] for i in ix]):.3f}")
        print(f"  latency per query median {np.median([r['ms'] for r in o]):.1f} ms p95 {np.percentile([r['ms'] for r in o], 95):.1f} ms")
    table("baseline docs.retrieve", base); table("candidate gate + docs ranking", cand); table("candidate gate + hybrid ranking", hyb); table("ABLATION B: gate without usage-cue features + docs ranking", candB)
    rng = np.random.default_rng(0)

    def boot(a, b, sel, key):
        ix = [i for i, x in enumerate(T) if sel(x)]; ks = sorted({T[i]["key"] for i in ix}); by = {k: [i for i in ix if T[i]["key"] == k] for k in ks}
        d = {k: (sum(a[i][key] - b[i][key] for i in by[k]), len(by[k])) for k in ks}; est = sum(v[0] for v in d.values()) / len(ix); m = []
        for _ in range(5000):
            pick = rng.integers(0, len(ks), len(ks)); m.append(sum(d[ks[j]][0] for j in pick) / sum(d[ks[j]][1] for j in pick))
        return est, np.percentile(m, 2.5), np.percentile(m, 97.5)
    out = {}
    for lab, a_, b_, sel, key in (("G1 hard injection (cand - base)", cand, base, lambda x: x["group"] in HARD, "inj"), ("G2 easy injection", cand, base, lambda x: x["group"] == "neg_easy", "inj"),
                                 ("G3 recall@4 all positives", cand, base, lambda x: x["kind"] == "pos", "hit4"), ("B (no cues) hard injection", candB, base, lambda x: x["group"] in HARD, "inj"), ("B (no cues) recall@4 all positives", candB, base, lambda x: x["kind"] == "pos", "hit4"),
                                 ("A recall@4 only pos_sem+pos_old (no template cues)", cand, base, lambda x: x["group"] in ("pos_sem", "pos_old"), "hit4"), ("B recall@4 only pos_sem+pos_old", candB, base, lambda x: x["group"] in ("pos_sem", "pos_old"), "hit4"),
                                 ("hybrid-vs-docs recall@4 (both gated)", hyb, cand, lambda x: x["kind"] == "pos", "hit4"),
                                 ("hybrid-vs-docs injection (both gated)", hyb, cand, lambda x: x["kind"] == "neg", "inj")):
        e, lo, hi = boot(a_, b_, sel, key); out[lab] = (e, lo, hi); print(f"{lab}: {e:+.3f}  IC95 [{lo:+.3f}, {hi:+.3f}]  (bootstrap by key)")
    hard_ix = [i for i, x in enumerate(T) if x["group"] in HARD]
    print("operating curve on test (t, coverage of positives, injection on hard):", [(t, round(float((pr[[i for i, x in enumerate(T) if x['kind'] == 'pos']] >= t).mean()), 3), round(float((pr[hard_ix] >= t).mean()), 3)) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)])
    lat = np.median([r["ms"] for r in cand]) - np.median([r["ms"] for r in base]); print(f"G4 latency added (median) {lat:+.1f} ms")
    json.dump(dict(model_t=model["t"], results={k: list(v) for k, v in out.items()}, latency_added_ms=float(lat)), open(os.path.join(ROOT, "bench/gate_result.json"), "w"), indent=1)
