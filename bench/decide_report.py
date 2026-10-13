"""Metrics of the decision backends from bench/decide_results/*.jsonl (written by decide_eval.py). Prints markdown tables.
usage: .venv/bin/python bench/decide_report.py [file ...]      (default: every file in bench/decide_results)
Per task on the TEST split: accuracy (Wilson 95 % interval), Brier, ECE (10 bins), latency p50/p95. For `done` also AUC, and the threshold tau chosen on VALIDATION
(smallest value with no false accept among the validation negatives that pass the objective checks), with the false accepts and the recall it gives on test.
`done` is then repeated with padded evidence (pad column) for the context-length study."""
import glob, json, math, os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
files = sys.argv[1:] or sorted(glob.glob(os.path.join(ROOT, "bench", "decide_results", "*.jsonl")))


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def brier(P, y, opts):
    Y = np.eye(len(opts))[[opts.index(l) for l in y]]
    return float(((P - Y) ** 2).sum(1).mean())


def ece(P, y, opts, bins=10):
    conf, pred = P.max(1), P.argmax(1); ok = np.array([opts[i] == l for i, l in zip(pred, y)])
    e = 0.0
    for b in range(bins):
        m = (conf > b / bins) & (conf <= (b + 1) / bins)
        if m.any():
            e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)


def auc(score, pos):
    s, pos = np.asarray(score), np.asarray(pos, bool)
    if pos.all() or (~pos).all():
        return float("nan")
    r = s.argsort().argsort() + 1.0
    return float((r[pos].sum() - pos.sum() * (pos.sum() + 1) / 2) / (pos.sum() * (~pos).sum()))


def load(f):
    rows = [json.loads(l) for l in open(f)]
    meta = next((r["meta"] for r in rows if "meta" in r), {})
    return [r for r in rows if "probs" in r], meta


print("# Decision backends, test split (val used only to choose tau)\n")
for f in files:
    rows, meta = load(f)
    name = os.path.basename(f)[:-6]
    print(f"## {name}  (extra RAM {meta.get('extra_ram_mb', 0):.0f} MB)\n")
    print("| task | n | accuracy [95 % CI] | Brier | ECE | latency p50 / p95 (s) |\n|---|---:|---:|---:|---:|---:|")
    for t in ("done", "route", "mixer"):
        te = [r for r in rows if r["task"] == t and r["split"] == "test" and r["pad"] == 0]
        if not te:
            continue
        opts = list(te[0]["probs"]); P = np.array([[r["probs"][o] for o in opts] for r in te]); y = [r["label"] for r in te]
        k = sum(opts[i] == l for i, l in zip(P.argmax(1), y)); lo, hi = wilson(k, len(y)); lat = [r["seconds"] for r in te]
        print(f"| {t} | {len(y)} | {k / len(y):.3f} [{lo:.2f}, {hi:.2f}] | {brier(P, y, opts):.3f} | {ece(P, y, opts):.3f} | {np.percentile(lat, 50):.3f} / {np.percentile(lat, 95):.3f} |")
    d = [r for r in rows if r["task"] == "done"]
    if d:
        val = [r for r in d if r["split"] == "val" and r["pad"] == 0]
        neg = [r["probs"]["YES"] for r in val if r["label"] == "NO" and r["objective_ok"]]
        tau = (max(neg) + 0.001) if neg else 0.5
        print(f"\n`done` guard: tau = {tau:.3f} (from {len(neg)} validation negatives that pass the objective checks)\n")
        te0 = [r for r in d if r["split"] == "test" and r["pad"] == 0]
        neg_all, pos_all = [r for r in te0 if r["label"] == "NO"], [r for r in te0 if r["label"] == "YES"]
        acc_obj = lambda rs: sum(bool(r["objective_ok"]) for r in rs)
        acc_mod = lambda rs: sum(bool(r["objective_ok"]) and r["probs"]["YES"] >= tau for r in rs)
        print(f"End to end on the {len(te0)} test states (NO = the task is not done): DONE accepted by the objective checks alone: wrong {acc_obj(neg_all)}/{len(neg_all)}, right {acc_obj(pos_all)}/{len(pos_all)}; "
              f"objective checks + p(done) >= tau: wrong {acc_mod(neg_all)}/{len(neg_all)}, right {acc_mod(pos_all)}/{len(pos_all)}.\n")
        print("| tau | wrong DONE accepted (of 36) | right DONE accepted (of 39) |\n|---:|---:|---:|")
        for tt in (0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
            print(f"| {tt} | {sum(bool(r['objective_ok']) and r['probs']['YES'] >= tt for r in neg_all)} | {sum(bool(r['objective_ok']) and r['probs']['YES'] >= tt for r in pos_all)} |")
        print()
        print("| pad lines | n | AUC (all) | AUC (objective_ok) | accuracy @0.5 | false accepts @tau (objective_ok negatives) | recall @tau (objective_ok positives) | chars median |\n|---:|---:|---:|---:|---:|---:|---:|---:|")
        for pad in sorted({r["pad"] for r in d if r["split"] == "test"}):
            te = [r for r in d if r["split"] == "test" and r["pad"] == pad]
            s = [r["probs"]["YES"] for r in te]; pos = [r["label"] == "YES" for r in te]
            ok = [r for r in te if r["objective_ok"]]
            so = [r["probs"]["YES"] for r in ok]; po = [r["label"] == "YES" for r in ok]
            fa = sum(x >= tau for x, p in zip(so, po) if not p); nn = sum(not p for p in po)
            rc = sum(x >= tau for x, p in zip(so, po) if p); npos = sum(po)
            acc = np.mean([(x >= 0.5) == p for x, p in zip(s, pos)])
            lo, hi = wilson(fa, nn)
            print(f"| {pad} | {len(te)} | {auc(s, pos):.3f} | {auc(so, po):.3f} | {acc:.3f} | {fa}/{nn} [{lo:.2f}, {hi:.2f}] | {rc}/{npos} | {int(np.median([r['chars'] for r in te]))} |")
    print()
