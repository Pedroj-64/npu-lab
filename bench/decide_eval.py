"""Run one decision backend over the labelled data and store the probabilities per example (metrics are computed by bench/decide_report.py).
usage: .venv/bin/python bench/decide_eval.py --backend embed [--tasks done,route,mixer]                      (CPU only, ~1 min)
       .venv/bin/python bench/decide_eval.py --backend logprob --model models/qwen35_2b [--tasks ...]           (NPU+GPU: run alone)
Splits evaluated: val and test (logprob is never trained; embed is fit on train only). `done` is also run on the test split with its evidence padded with
--pad extra lines of irrelevant log output (default 0,25,80) to see how accuracy moves with the length of the context.
Output: bench/decide_results/<backend>[_<model>].jsonl, one line per decision: id, task, split, pad, label, probs, seconds, state chars."""
import argparse, json, os, random, resource, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import decide

ap = argparse.ArgumentParser()
ap.add_argument("--backend", required=True, choices=["logprob", "embed"])
ap.add_argument("--model", default="models/qwen35_2b")
ap.add_argument("--tasks", default="done,route,mixer")
ap.add_argument("--pad", default="0,25,80")
ap.add_argument("--vram-gb", type=float, default=4.5)
a = ap.parse_args()
DATA = {t: [json.loads(l) for l in open(f"bench/decide_data/{t}.jsonl")] for t in a.tasks.split(",")}
NOISE = ["INFO worker-{k} heartbeat ok seq={n}", "DEBUG cache hit ratio 0.{n}", "INFO scheduler tick {n} queue=0", "DEBUG gc pause {k}ms", "INFO request /health 200 {k}ms",
         "WARN slow disk read {n}us on sda", "INFO metrics flushed {k} points"]


def pad(state, n, seed):
    """Insert n lines of irrelevant log noise before the 'Compile:' line (the end of the evidence stays last)."""
    if n == 0:
        return state
    r = random.Random(seed)
    lines = [r.choice(NOISE).format(k=r.randint(1, 9), n=r.randint(100, 999)) for _ in range(n)]
    return state.replace("\nCompile:", "\nOther output:\n" + "\n".join(lines) + "\nCompile:", 1)


def jobs():
    for t, rows in DATA.items():
        for r in rows:
            if r["split"] in ("val", "test"):
                yield t, r, 0
        if t == "done":
            for n in map(int, a.pad.split(",")):
                if n:
                    for r in rows:
                        if r["split"] == "test":
                            yield t, r, n


if a.backend == "embed":
    from minilm import MiniLM
    rss = lambda: int(next(l for l in open("/proc/self/status") if l.startswith("VmRSS")).split()[1]) / 1024    # MB resident now
    m0 = rss()
    backend = decide.EmbedBackend(MiniLM(d="models/multi", max_len=256, fast_gelu=True))
    backend.emb.encode(["warm up"])
    for t, rows in DATA.items():
        tr = [r for r in rows if r["split"] == "train"]
        backend.fit(t, [r["state"] for r in tr], [r["label"] for r in tr])
    ram_mb = rss() - m0
    tag = "embed"
else:
    import qwen35_npu as Q35
    eng = Q35.load_engine(a.model, gpu=True, gpu_share=0.0, vram_gb=a.vram_gb, max_ctx=12000) if "qwen35" in a.model else __import__("qwen_npu").Qwen(a.model)
    backend = decide.LogprobBackend(eng, limit=12000)
    ram_mb = 0.0
    tag = "logprob_" + os.path.basename(a.model)
os.makedirs("bench/decide_results", exist_ok=True)
out = open(f"bench/decide_results/{tag}.jsonl", "w")
n = 0
for t, r, p in jobs():
    state = pad(r["state"], p, sum(map(ord, r["id"])))
    t0 = time.perf_counter()
    probs = backend.probs(state, {"task": t})
    dt = time.perf_counter() - t0
    out.write(json.dumps(dict(id=r["id"], task=t, split=r["split"], pad=p, label=r["label"], probs=probs, seconds=dt, chars=len(state), objective_ok=r.get("objective_ok"), how=r.get("how"))) + "\n")
    out.flush(); n += 1
    if n % 25 == 0:
        print(f"{n} decisions", flush=True)
out.write(json.dumps(dict(meta=dict(backend=a.backend, model=a.model if a.backend == "logprob" else "models/multi", extra_ram_mb=ram_mb, n=n))) + "\n")
print("done", n, "->", out.name)
