"""Can a small local model act as a terminal agent? It gets a task about a small project and answers with ONE action per turn in plain text (no native
tool calling): READ / LIST / DELETE / DONE, or SEARCH/REPLACE blocks to edit or create files. The harness executes each action on a temporary copy of the
project and feeds the result back, for at most --steps turns. Two numbers per run: how many replies were a valid action that ran ("valid calls"),
and how many tasks end with a project that passes a code/filesystem check ("solved").
usage: python bench/agent_eval.py --check              scripted perfect agent through the same harness: verifies the tasks and checks (no model, ~2 s)
       python bench/agent_eval.py --model qwen35l [--tasks 1,2,3,4,5] [--steps 8] [--tokens 350] [--deadline 1500]   (NPU+GPU: run alone)"""
import http.client, json, os, re, shutil, subprocess, sys, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, os.path.join(ROOT, "bench"))
import edit_eval as E, project, repomap

arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
PROTOCOL = """You are an agent that changes the files of a small Python project, one action per reply. Reply with exactly ONE action and nothing else:
READ path                  show a file
LIST                       list the project files
DELETE path                delete a file
DONE one-line summary      you are finished
or to edit or create a file, a block (the path on the line before it):
path/file.py
<<<<<<< SEARCH
(lines copied exactly from the current file; leave empty to create a new file)
=======
(the new lines)
>>>>>>> REPLACE
You will see the result of each action. Read a file before editing it. Edit one place per reply."""
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


def make_project(folder):
    E.project(folder, 8)
    open(os.path.join(folder, "legacy.py"), "w").write(LEGACY)


def listing(root):
    return "\n".join(sorted(os.path.relpath(os.path.join(d, f), root) for d, subs, fs in os.walk(root) for f in fs if not f.endswith(".pyc") and "__pycache__" not in d))


def act(root, reply):
    """Run the single action in `reply` -> (valid, text shown to the model, finished)."""
    text = reply.split("</think>")[-1].strip()
    if "<<<<<<< SEARCH" in text:
        res = project.apply(root, text)
        bad = [r for r in res if r["action"] in ("failed", "none")]
        return (not bad, "; ".join(f"{r['action']} {r['file']}" + (f" ({r['error']})" if r["error"] else "") for r in res), False)
    m = re.match(r"\s*(READ|LIST|DELETE|DONE)\b[ \t]*([^\n]*)", text)
    if not m:
        return False, "not an action: reply with READ, LIST, DELETE, DONE or an edit block", False
    verb, rest = m.group(1), m.group(2).strip().strip("`")
    if verb == "DONE":
        return True, rest, True
    if verb == "LIST":
        return True, listing(root), False
    try:
        path = project._safe(root, rest, must_exist=True)
        if verb == "READ":
            return True, open(path, encoding="utf-8").read()[:4000], False
        os.remove(path)
        return True, f"deleted {rest}", False
    except (ValueError, OSError, UnicodeDecodeError) as e:
        return False, f"{verb} failed: {e}", False


def run_task(tid, ask, steps):
    """-> (solved, valid calls, total calls, log)."""
    root = tempfile.mkdtemp(dir=os.path.expanduser("~"))
    try:
        make_project(root)
        msg = f"{PROTOCOL}\n\nProject map:\n{repomap.build(root, 2500)}\n\nTask: {TASKS[tid][0]}"
        valid = total = 0; log = []
        for _ in range(steps):
            reply = ask(msg); total += 1
            ok, shown, done = act(root, reply)
            valid += ok; log.append(("ok " if ok else "BAD") + " " + reply.split("</think>")[-1].strip().replace("\n", " | ")[:90])
            if done:
                break
            msg = ("Result: " if ok else "Error: ") + shown + "\nNext action?"
        r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, '.')\n" + TASKS[tid][1]], cwd=root, capture_output=True, text=True, timeout=20)
        return r.returncode == 0, valid, total, log + ([f"check: {r.stderr.strip().splitlines()[-1][:100]}"] if r.returncode else [])
    finally:
        shutil.rmtree(root, ignore_errors=True)


def check():
    for tid in TASKS:
        script = iter(SCRIPT[tid])
        solved, valid, total, log = run_task(tid, lambda m: next(script), 10)
        print(f"task {tid}: scripted agent solved={solved}  valid {valid}/{total}")
        assert solved and valid == total, log
    nothing = run_task(1, lambda m: "DONE nothing", 3)       # an agent that does nothing must NOT pass any check
    assert not nothing[0]
    garbage = iter(["hello", "READ ../../etc/passwd", "DELETE /etc/passwd", "DONE"])
    g = run_task(1, lambda m: next(garbage), 5); assert g[1] == 1 and g[2] == 4 and not g[0], g       # three refused/invalid replies, only the final DONE counts
    print("agent harness ok")


def run():
    model, steps, tokens, deadline = arg("--model", "qwen35l"), arg("--steps", 8), arg("--tokens", 350), arg("--deadline", 1500)
    tids = [int(x) for x in arg("--tasks", "1,2,3,4,5").split(",")]
    tmp = tempfile.mkdtemp(); T = "g" * 43; t0 = time.time()
    p = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "src", "serve.py")], cwd=ROOT, stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "NPU_LAB_TOKEN": T, **{f"NPU_LAB_{k}": f"{tmp}/{v}" for k, v in dict(RULES="rules.json", DOCS="d", ROUTING="r.json", SETTINGS="s.json", CHATS="c", FACTS="f.json").items()}})
    port = json.loads(p.stdout.readline())["port"]

    def call(path, body, timeout=400):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
        c.request("POST", path, json.dumps(body), {"Authorization": "Bearer " + T, "Host": f"127.0.0.1:{port}"}); r = c.getresponse(); return r.status, json.loads(r.read())

    hist = []

    def ask(prompt):
        if time.time() - t0 > deadline: raise SystemExit(f"deadline of {deadline}s reached")
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
        c.request("POST", "/chat", json.dumps({"prompt": prompt, "tokens": tokens, "history": hist, "docs": "off", "web": "off", "mine": "off"}), {"Authorization": "Bearer " + T, "Host": f"127.0.0.1:{port}"})
        out = ""
        for line in c.getresponse():
            m = json.loads(line); out += m.get("token", "")
            if m.get("error"): raise SystemExit("engine error: " + str(m["error"]))
            if m.get("done") is True: break
        hist.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": out.split("</think>")[-1].strip()}])
        return out

    solved = valid = total = 0
    try:
        call("/profile", {"profile": "assisted"}); call("/rules", {"context_tokens": 8192, "llm_vram_gb": 5.0, "gpu_share": 0.0, "compact_at": 0})
        s, b = call("/mode", {"mode": "llm", "model": model})
        if s != 200: raise SystemExit(f"load failed: {b}")
        for tid in tids:
            hist.clear(); t = time.time()
            ok, v, n, log = run_task(tid, ask, steps)
            solved += ok; valid += v; total += n
            print(f"task {tid}: {'SOLVED' if ok else 'FAILED'}  valid {v}/{n}  {time.time() - t:.0f}s", flush=True)
            for l in log: print("    " + l, flush=True)
    finally:
        try: call("/mode", {"mode": "none"}, 60)
        except Exception: pass
        p.terminate()
    print(f"{model}: solved {solved}/{len(tids)}, valid calls {valid}/{total} ({100 * valid // max(total, 1)}%)")


if __name__ == "__main__":
    check() if "--check" in sys.argv else run()
