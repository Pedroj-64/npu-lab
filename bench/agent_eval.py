"""Can a small local model act as a terminal agent? It gets a task about a small project and answers with ONE action per turn in plain text (no native
tool calling): READ / LIST / DELETE / DONE, or SEARCH/REPLACE blocks to edit or create files. The harness executes each action on a temporary copy of the
project and feeds the result back, for at most --steps turns. Two numbers per run: how many replies were a valid action that ran ("valid calls"),
and how many tasks end with a project that passes a code/filesystem check ("solved").
usage: python bench/agent_eval.py --check              scripted perfect agent through the same harness, all 25 tasks of bench/agent_tasks.py (no model, ~15 s)
       python bench/agent_eval.py --model qwen3 [--tasks 1-25] [--samples 3] [--steps 8] [--tokens 350] [--deadline 7200] [--lang en|es]
                                  [--run] [--guard off|objective|model] [--tau 0.5]     (NPU+GPU: run alone; --run gives the agent RUN on the "shop" tasks;
                                  every task run is appended to bench/agent_traces/*.jsonl: replies, results and final files, for later analysis)"""
import http.client, json, os, re, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "bench"))
import edit_eval as E, project, repomap, agent

arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
LEGACY = '"""Old helpers nobody uses."""\n\n\ndef old():\n    return 0\n'
TASKS = {
    1: ("In util.py make clamp raise ValueError when lo is greater than hi.",
        "import util\nassert util.clamp(5, 0, 3) == 3\ntry:\n    util.clamp(1, 5, 0)\nexcept ValueError:\n    pass\nelse:\n    raise SystemExit('no ValueError')"),
    2: ("Create a new file stats.py with a function average(nums) that returns the mean of a list of numbers, and 0.0 for an empty list.",
        "import stats\nassert stats.average([1, 2, 3]) == 2 and stats.average([]) == 0.0"),
    3: ("Add a method total_value() to Store (sum of qty * price over all items) and make main in cli.py print a final line 'TOTAL: ' plus that value with two decimals.",
        "import io, contextlib, cli\nb = io.StringIO()\nwith contextlib.redirect_stdout(b):\n    cli.main([])\nassert 'TOTAL: 46.25' in b.getvalue(), b.getvalue()"),
    4: ("Delete the file legacy.py: nothing uses it.",
        "import os\nassert not os.path.exists('legacy.py')\nimport cli"),
    5: ("Rename the function format_line to format_item everywhere it appears (report.py and cli.py).",
        "src = open('cli.py').read() + open('report.py').read()\nassert 'format_line' not in src\nimport io, contextlib, cli\nb = io.StringIO()\nwith contextlib.redirect_stdout(b):\n    cli.main([])\nassert 'Pen' in b.getvalue()"),
}
B = lambda path, a, b: f"{path}\n<<<<<<< SEARCH\n{a}\n=======\n{b}\n>>>>>>> REPLACE"
SCRIPT = {          # the replies of a perfect agent, to verify the harness and the checks without a model
    1: ["READ util.py", B("util.py", "    return max(lo, min(hi, x))", "    if lo > hi:\n        raise ValueError('lo > hi')\n    return max(lo, min(hi, x))"), "DONE clamp validates"],
    2: ["LIST", B("stats.py", "", "def average(nums):\n    return sum(nums) / len(nums) if nums else 0.0"), "DONE created stats.py"],
    3: ["READ store.py", B("store.py", "        return list(self._items.values())", "        return list(self._items.values())\n\n    def total_value(self):\n        return sum(i.qty * i.price for i in self.items())"), "READ cli.py",
        B("cli.py", "        print(format_line(it.name, it.qty, it.price))", "        print(format_line(it.name, it.qty, it.price))\n    print(f\"TOTAL: {store.total_value():.2f}\")"), "DONE total added"],
    4: ["LIST", "DELETE legacy.py", "DONE deleted"],
    5: ["READ report.py", B("report.py", "def format_line(", "def format_item("), "READ cli.py", B("cli.py", "from report import format_line", "from report import format_item"),
        B("cli.py", "print(format_line(", "print(format_item("), "DONE renamed"],
}


def make_project(folder, fixture="base"):
    import agent_tasks as T
    T.write_project(folder, fixture)


def task_text(tid, lang="en"):
    import agent_tasks as T
    return T.TASKS[tid][lang]


def run_check(root, tid):
    import agent_tasks as T
    return T.run_check(root, tid)


def local_act(root, **kw):
    """The product's own agent code, in this process (used by --check and by runs that do not need the model-based guard)."""
    return lambda reply, last_ok, confirmed=True: agent.run(root, reply, confirmed=confirmed, last_ok=last_ok, **kw)


def run_task(tid, ask, steps, act=None, run_cmd="", lang="en", guard="off"):
    """-> (solved, valid calls, total calls, log, info). `act(root, reply, last_ok)` executes one action (default: agent.run in this process, every write auto-confirmed);
    info = dict(false_done, refused_done, steps, final: {path: text}, replies: [...])."""
    import agent_tasks as T
    t = T.TASKS[tid]
    root = tempfile.mkdtemp(dir=os.path.expanduser("~"))
    try:
        make_project(root, t["fixture"])
        cmd = test_command() if (run_cmd and t["fixture"] == "shop") else ""
        do = act(root, cmd) if act else (lambda reply, last_ok: agent.run(root, reply, confirmed=True, last_ok=last_ok, test_command=cmd, guard=guard, task=t[lang]))
        msg = f"{agent.PROTOCOL}{agent.PROTOCOL_RUN if cmd else ''}\n\nProject map:\n{repomap.build(root, 2500)}\n\nTask: {t[lang]}"
        valid = total = 0; log = []; last_ok = True; info = dict(false_done=0, refused_done=0, replies=[])
        for _ in range(steps):
            reply = ask(msg); total += 1
            r = do(reply, last_ok)
            ok, shown, done = r["ok"], r["shown"], r["done"]
            last_ok = ok
            valid += ok; log.append(("ok " if ok else "BAD") + " " + reply.split("</think>")[-1].strip().replace("\n", " | ")[:90])
            info["replies"].append(dict(reply=reply.split("</think>")[-1].strip()[:1500], kind=r["kind"], ok=ok, shown=shown[:300]))
            if r["kind"] == "done" and not done:
                info["refused_done"] += 1
            if done:
                break
            msg = ("Result: " if ok else "Error: ") + shown + "\nNext action?"
        info["steps"] = total
        solved, why = T.run_check(root, tid)
        info["false_done"] = int(bool(log) and r["kind"] == "done" and r["done"] and not solved)
        info["final"] = T.snapshot(root)
        return solved, valid, total, log + ([f"check: {why}"] if not solved else []), info
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_command():
    """The command a user would configure for RUN on the "shop" fixture (the interpreter of this process)."""
    import shlex, agent_tasks as T
    return " ".join(shlex.quote(sys.executable if c == "python" else c) for c in T.TEST_COMMAND)


def script_for(tid):
    """Replies of a perfect agent for any task of the bank: one edit block per changed file (whole-file SEARCH), or DELETE, then DONE."""
    import agent_tasks as T
    t = T.TASKS[tid]; base = T.FIXTURES[t["fixture"]]; out = ["LIST"]
    for n, new in t["ref"].items():
        if new is None:
            out.append(f"DELETE {n}")
        else:
            old = base.get(n)
            out.append(B(n, old.rstrip("\n") if old else "", new.rstrip("\n")))
    if t["fixture"] == "shop":
        out.append("RUN")
    return out + ["DONE finished"]


def check():
    import agent_tasks as T
    T.fill_original_refs()
    for tid in T.TASKS:
        script = iter(script_for(tid))
        solved, valid, total, log, info = run_task(tid, lambda m: next(script), 12, run_cmd=True)
        print(f"task {tid}: scripted agent solved={solved}  valid {valid}/{total}")
        assert solved and valid == total and not info["false_done"], log
        script = iter(script_for(tid))                                                             # the same agent with the objective guard on: still accepted
        assert run_task(tid, lambda m: next(script), 12, run_cmd=True, guard="objective")[0]
    nothing = run_task(1, lambda m: "DONE nothing", 3)       # an agent that does nothing must NOT pass any check
    assert not nothing[0] and nothing[4]["false_done"] == 1  # ... and its DONE is counted as a false DONE
    assert run_task(1, lambda m: "DONE nothing", 3, guard="objective")[4]["refused_done"] == 3        # with the guard on, every one of those is refused
    garbage = iter(["hello", "READ ../../etc/passwd", "DELETE /etc/passwd", "READ util.py", "DONE"])
    g = run_task(1, lambda m: next(garbage), 6); assert g[1] == 2 and g[2] == 5 and not g[0], g       # three refused replies; the READ and the final DONE count
    failed_done = iter(["DELETE nothere.py", "DONE all good", "LIST", "DONE ok"])
    f = run_task(1, lambda m: next(failed_done), 6); assert (f[1], f[2]) == (2, 4), f                   # a DONE right after a failed action is refused
    broken = iter([B("bad.py", "", "def (:"), "DONE x", "DELETE bad.py", "DONE y"])
    b = run_task(1, lambda m: next(broken), 6); assert (b[1], b[2]) == (3, 4), b                         # a file that no longer compiles blocks DONE
    miss = iter(["READ util.py", B("util.py", "return nothing", "x"), "LIST", "DONE"])
    shown = []
    run_task(1, lambda m: (shown.append(m), next(miss))[1], 5)
    assert any("currently contains" in m and "def clamp" in m for m in shown), shown                    # a mismatch shows the real file
    shop_bug = iter(["LIST", B("orders.py", "sum(price for _, qty, price in self.lines)", "sum(price for _, qty, price in self.lines) + 0"), "RUN", "DONE fixed it"])
    sb = run_task(23, lambda m: next(shop_bug), 6, run_cmd=True); assert not sb[0] and sb[4]["false_done"] == 1       # tests pass but the hidden check does not: a false DONE only a smarter guard can catch
    print("agent harness ok")


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 1.0
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** .5 / d
    return max(0.0, c - h), min(1.0, c + h)


def parse_ids(spec):
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-"); out += range(int(a), int(b or a) + 1)
    return out


def run():
    import statistics as st, agent_tasks as T
    T.fill_original_refs()
    model, steps, tokens, deadline = arg("--model", "qwen3"), arg("--steps", 8), arg("--tokens", 350), arg("--deadline", 7200)
    tids, samples, lang, guard, tau, use_run = parse_ids(arg("--tasks", "1-25")), arg("--samples", 3), arg("--lang", "en"), arg("--guard", "off"), arg("--tau", 0.5), "--run" in sys.argv
    tmp = tempfile.mkdtemp(); Tk = "g" * 43; t0 = time.time()
    p = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "src", "serve.py")], cwd=ROOT, stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "NPU_LAB_TOKEN": Tk, **{f"NPU_LAB_{k}": f"{tmp}/{v}" for k, v in dict(RULES="rules.json", DOCS="d", ROUTING="r.json", SETTINGS="s.json", CHATS="c", FACTS="f.json").items()}})
    port = json.loads(p.stdout.readline())["port"]

    def call(path, body, timeout=400):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        c.request("POST", path, json.dumps(body), {"Authorization": "Bearer " + Tk, "Host": f"127.0.0.1:{port}"}); r = c.getresponse(); return r.status, json.loads(r.read())

    hist = []

    def ask(prompt):
        if time.time() - t0 > deadline: raise SystemExit(f"deadline of {deadline}s reached")
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        c.request("POST", "/chat", json.dumps({"prompt": prompt, "tokens": tokens, "history": hist, "docs": "off", "web": "off", "mine": "off"}), {"Authorization": "Bearer " + Tk, "Host": f"127.0.0.1:{port}"})
        out = ""
        for line in c.getresponse():
            m = json.loads(line); out += m.get("token", "")
            if m.get("error"): raise SystemExit("engine error: " + str(m["error"]))
            if m.get("done") is True: break
        hist.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": out.split("</think>")[-1].strip()}])
        return out

    def act(root, cmd):                       # every action goes through the server's /agent/run (same code as the page and the terminal chat), writes auto-confirmed
        s, b = call("/rules", {"project_root": root, "agent_test_command": cmd, "agent_guard": guard, "agent_guard_tau": tau})
        assert s == 200, b
        def do(reply, last_ok):
            s, r = call("/agent/run", {"text": reply, "confirm": True, "last_ok": last_ok, "task": T.TASKS[cur[0]][lang]}); assert s == 200, r
            return r
        return do

    cur = [0]
    os.makedirs(os.path.join(ROOT, "bench", "agent_traces"), exist_ok=True)
    trace = open(os.path.join(ROOT, "bench", "agent_traces", f"{model}_{guard}_{int(t0)}.jsonl"), "w")
    try:
        call("/profile", {"profile": "assisted"}); call("/rules", {"context_tokens": 8192, "llm_vram_gb": 5.0, "gpu_share": 0.0, "compact_at": 0})
        s, b = call("/mode", {"mode": "llm", "model": model})
        if s != 200: raise SystemExit(f"load failed: {b}")
        res = []
        for smp in range(samples):
            for tid in tids:
                hist.clear(); cur[0] = tid; t = time.time()
                ok, v, n, log, info = run_task(tid, ask, steps, act=act, run_cmd=use_run, lang=lang, guard=guard)
                res.append(dict(sample=smp, task=tid, solved=ok, valid=v, total=n, **{k: info[k] for k in ("false_done", "refused_done", "steps")}, seconds=round(time.time() - t)))
                trace.write(json.dumps(dict(model=model, guard=guard, run=use_run, lang=lang, **res[-1], replies=info["replies"], final=info["final"]), ensure_ascii=False) + "\n"); trace.flush()
                print(f"sample {smp} task {tid}: {'SOLVED' if ok else 'FAILED'}  valid {v}/{n}  false DONE {info['false_done']}  refused DONE {info['refused_done']}  {res[-1]['seconds']}s", flush=True)
                for l in log: print("    " + l, flush=True)
    finally:
        try: call("/mode", {"mode": "none"}, 60)
        except Exception: pass
        p.terminate()
    per = [sum(r["solved"] for r in res if r["sample"] == k) for k in range(samples) if any(r["sample"] == k for r in res)]
    valid, total = sum(r["valid"] for r in res), sum(r["total"] for r in res)
    fd = sum(r["false_done"] for r in res); lo, hi = wilson(fd, len(res))
    print(f"{model} guard={guard} run={use_run}: solved per sample {per} (mean {st.mean(per):.1f} +- {st.stdev(per) if len(per) > 1 else 0:.1f} of {len(tids)}), "
          f"valid calls {valid}/{total} ({100 * valid // max(total, 1)}%), false DONE {fd}/{len(res)} task runs [{lo:.2f}, {hi:.2f}], refused DONE {sum(r['refused_done'] for r in res)}")


if __name__ == "__main__":
    check() if "--check" in sys.argv else run()
