"""Replay the agent traces (bench/agent_traces/*.jsonl, written by agent_eval.py after EVERY task run) through the DONE guard, offline.
For each run that ended with an accepted DONE it asks: would the guard have refused it? and was the project really solved (the task's hidden check)?
  - objective guard: nothing changed / does not compile / configured tests not run after the last change / tests fail   (src/decide.py done_guard, same rules as src/agent.py)
  - model guard: the above AND p(done) >= tau, with p(done) from the loaded model (decide.LogprobBackend). The probabilities are computed once with --score (needs the
    NPU/GPU free: it loads the model) and cached in bench/agent_traces/guard_probs.jsonl; the report never loads a model.
The statistical unit is the TASK: confidence intervals are a bootstrap over tasks. n = 1 sample per task is exploration, not a conclusion.
usage: .venv/bin/python bench/guard_offline.py [--traces GLOB] [--calibrate]            report (CPU only; works with partial traces)
       .venv/bin/python bench/guard_offline.py --score --model models/qwen3          fill the probability cache for runs that have none (NPU+GPU: run alone; resumable)
--calibrate: tau chosen leave-one-task-out on these real traces (smallest tau with no false accept among the OTHER tasks' wrong DONEs) instead of the fixed tau of the synthetic data."""
import argparse, glob, json, os, re, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "bench"))
import decide, agent_tasks as T

ap = argparse.ArgumentParser()
ap.add_argument("--traces", default=os.path.join(ROOT, "bench", "agent_traces", "*_off_*.jsonl"), help="glob of trace files to read (default: the guard-off runs)")
ap.add_argument("--score", action="store_true")
ap.add_argument("--model", default="models/qwen3")
ap.add_argument("--calibrate", action="store_true")
ap.add_argument("--taus", default="0.30:0.90:0.05")
ap.add_argument("--boot", type=int, default=2000)
a = ap.parse_args()
CACHE = os.path.join(ROOT, "bench", "agent_traces", "guard_probs.jsonl")


def load_runs():
    runs = []
    for f in sorted(glob.glob(a.traces)):
        for i, l in enumerate(open(f)):
            if l.strip():
                r = json.loads(l); r["file"] = os.path.basename(f); r["key"] = f"{os.path.basename(f)}:{i}"; runs.append(r)
    return runs


def compile_err(files):
    for n, t in sorted(files.items()):
        if n.endswith(".py") and t is not None:
            try:
                compile(t, n, "exec")
            except (SyntaxError, ValueError) as e:
                return f"{n} does not compile: {e}"
    return ""


def replay(r):
    """-> dict(accepted DONE?, claim, tests (passed, output) | None, test_command configured?, before, after)."""
    t = T.TASKS[r["task"]]; before = T.FIXTURES[t["fixture"]]; after = r["final"]
    has_tests = bool(r.get("run")) and t["fixture"] == "shop"       # agent_eval gives RUN (and a test command) only on the shop fixture
    tests = None; done = None
    for rep in r["replies"]:
        if rep["kind"] in ("edit", "delete") and rep["ok"]:
            tests = None
        elif rep["kind"] == "run" and rep["ok"] and rep["shown"].startswith("exit code"):
            m = re.match(r"exit code (-?\d+)\n?(.*)", rep["shown"], re.S); tests = (m.group(1) == "0", m.group(2))
        elif rep["kind"] == "done" and rep["ok"]:
            done = re.sub(r"^\s*DONE\b\s*", "", rep["reply"].splitlines()[0] if rep["reply"] else "")
    return dict(accepted=done is not None, claim=done or "", tests=tests, has_tests=has_tests, before=before, after=after)


def objective(rp):
    """-> refusal reason or ''. Same rules as agent._guard + decide.done_guard."""
    if rp["has_tests"] and rp["tests"] is None and rp["after"] != rp["before"]:
        return "tests not run after the last change"
    if rp["before"] == rp["after"]:
        return "no file changed"
    if compile_err(rp["after"]):
        return "does not compile"
    if rp["tests"] is not None and rp["tests"][0] is False:
        return "tests fail"
    return ""


def state_of(r, rp):
    return decide.done_state(T.TASKS[r["task"]]["en"], rp["claim"], rp["before"], rp["after"], "", rp["tests"][1] if rp["tests"] else None)


def cache():
    return {json.loads(l)["key"]: json.loads(l) for l in open(CACHE)} if os.path.exists(CACHE) else {}


def score():
    import qwen_npu as Q35
    runs = [r for r in load_runs() if replay(r)["accepted"]]
    have = cache()
    todo = [r for r in runs if r["key"] not in have]
    print(f"{len(runs)} runs with an accepted DONE, {len(have)} already scored, {len(todo)} to do")
    if not todo:
        return
    eng = Q35.Qwen(a.model, gpu=True, vram_gb=4.5, max_ctx=12000)
    be = decide.LogprobBackend(eng, limit=12000)
    with open(CACHE, "a") as f:
        for r in todo:
            rp = replay(r); p = be.probs(state_of(r, rp), {"task": "done"})["YES"]
            f.write(json.dumps(dict(key=r["key"], task=r["task"], model=os.path.basename(a.model), p_yes=p)) + "\n"); f.flush()
            print(r["key"], f"p={p:.3f}", flush=True)


def boot_ci(per_task_pairs, n, rng):
    """per_task_pairs: {task: (k, m)} counts; bootstrap over tasks of sum(k)/sum(m)."""
    ts = list(per_task_pairs); out = []
    for _ in range(n):
        s = [per_task_pairs[ts[i]] for i in rng.integers(0, len(ts), len(ts))]
        m = sum(x[1] for x in s); out.append(sum(x[0] for x in s) / m if m else np.nan)
    return np.nanpercentile(out, [2.5, 97.5]) if len(out) else (np.nan, np.nan)


def rate(runs, pred_num, pred_den, rng):
    by = {}
    for r in runs:
        k, m = by.get(r["task"], (0, 0)); by[r["task"]] = (k + bool(pred_num(r)), m + bool(pred_den(r)))
    k, m = sum(v[0] for v in by.values()), sum(v[1] for v in by.values())
    lo, hi = boot_ci(by, a.boot, rng) if m else (np.nan, np.nan)
    return k, m, (k / m if m else float("nan")), lo, hi


def report():
    rng = np.random.default_rng(0)
    runs = load_runs()
    if not runs:
        sys.exit("no traces")
    for r in runs:
        r["rp"] = replay(r); r["obj"] = objective(r["rp"]) if r["rp"]["accepted"] else None
    n, solved = len(runs), sum(r["solved"] for r in runs)
    acc = [r for r in runs if r["rp"]["accepted"]]
    wrong = [r for r in acc if not r["solved"]]
    tasks = sorted({r["task"] for r in runs})
    print(f"# Guard replay on {n} runs of {len(tasks)} tasks (traces: {sorted({r['file'] for r in runs})})\n")
    print(f"- solved {solved}/{n}; ended with an accepted DONE {len(acc)}; ended without DONE {n - len(acc)}")
    k, m, p, lo, hi = rate(runs, lambda r: r["rp"]["accepted"] and not r["solved"], lambda r: True, rng)
    print(f"- **false DONE** (accepted DONE, hidden check fails): {k}/{m} runs = {p:.3f} [bootstrap by task 95 %: {lo:.3f}, {hi:.3f}]  [medido, n = {n // len(tasks) if tasks else 0} sample(s) per task: exploration]")
    k2, m2, p2, lo2, hi2 = rate(acc, lambda r: not r["solved"], lambda r: True, rng)
    print(f"- of the accepted DONEs, wrong: {k2}/{m2} = {p2:.3f} [{lo2:.3f}, {hi2:.3f}]")
    # objective guard
    caught = [r for r in wrong if r["obj"]]; lost = [r for r in acc if r["solved"] and r["obj"]]
    print(f"\n## Objective guard (no model)\n- wrong DONEs refused: {len(caught)}/{len(wrong)}; correct DONEs refused: {len(lost)}/{len(acc) - len(wrong)}")
    for r in caught + lost:
        print(f"  - task {r['task']} sample {r['sample']}: {'wrong' if not r['solved'] else 'CORRECT'} DONE refused: {r['obj']}")
    left = [r for r in wrong if not r["obj"]]
    print(f"- wrong DONEs that still pass the objective checks (what only a smarter signal could catch): {len(left)} -> tasks {[ (r['task'], r['sample']) for r in left ]}")
    # model guard
    pr = cache(); have = [r for r in acc if r["key"] in pr]
    print(f"\n## Model guard (p(done) cached for {len(have)}/{len(acc)} accepted DONEs)")
    if not have:
        print("- no probabilities yet: run with --score when the NPU/GPU are free")
    else:
        lo_t, hi_t, st = (float(x) for x in a.taus.split(":"))
        grid = np.arange(lo_t, hi_t + 1e-9, st)

        def cell(tau, subset):
            ok = [r for r in subset if not r["obj"] and pr[r["key"]]["p_yes"] >= tau]
            return sum(not r["solved"] for r in ok), sum(r["solved"] for r in ok)
        nw, nr = sum(not r["solved"] for r in have), sum(r["solved"] for r in have)
        print(f"\n| tau | wrong DONE accepted (of {nw}) | right DONE accepted (of {nr}) |\n|---:|---:|---:|")
        for tau in grid:
            w, rr = cell(tau, have); print(f"| {tau:.2f} | {w} | {rr} |")
        print(f"\n(objective guard alone, same runs: wrong accepted {sum(not r['solved'] and not r['obj'] for r in have)}, right accepted {sum(r['solved'] and not r['obj'] for r in have)}; "
              f"fixed tau from the synthetic data = 0.678: wrong {cell(0.678, have)[0]}, right {cell(0.678, have)[1]})")
        if a.calibrate:
            fa = ra = 0; taus = []
            for t in {r["task"] for r in have}:
                rest = [r for r in have if r["task"] != t and not r["obj"]]
                neg = [pr[r["key"]]["p_yes"] for r in rest if not r["solved"]]
                tau = (max(neg) + 0.001) if neg else 0.5
                taus.append((t, round(tau, 3), len(neg)))
                held = [r for r in have if r["task"] == t and not r["obj"] and pr[r["key"]]["p_yes"] >= tau]
                fa += sum(not r["solved"] for r in held); ra += sum(r["solved"] for r in held)
            print(f"\nleave-one-task-out tau (real traces): wrong accepted {fa}, right accepted {ra}; taus {taus}; negatives per fold {sorted({x[2] for x in taus})} "
                  f"{'<- too few negatives to calibrate' if min(x[2] for x in taus) < 2 else ''}")
    # data for a possible fine-tune (Phase 4b gate)
    good = [r for r in runs if r["solved"] and not r["false_done"]]
    print(f"\n## Traces usable for Phase 4b: {len(good)} good (check passes, no false DONE) of {n}; failed or false-DONE runs (the 'report incomplete' examples): {n - len(good)}  [target ~500 examples, gate >= 150]")


if __name__ == "__main__":
    score() if a.score else report()
