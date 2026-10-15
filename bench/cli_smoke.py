"""The terminal chat (src/chat_cli.py) against a REAL model, driven through pipes: scenario 1 solves task 7 of the agent bank answering 'y' to every
"Do it? [y/N]"; scenario 2 asks task 6 and answers 'n' at the first confirmation (nothing may be written). Settings live in a temp folder, the user's rules are untouched.
usage: .venv/bin/python bench/cli_smoke.py [--model qwen3] [--deadline 900]     (NPU+GPU: run alone)
Prints PASS/FAIL lines and the whole transcript to bench/.cli_smoke.txt (a temporary, not committed)."""
import os, select, shutil, subprocess, sys, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "bench")); sys.path.insert(0, os.path.join(ROOT, "src"))
import agent_tasks as T

arg = lambda k, d: sys.argv[sys.argv.index(k) + 1] if k in sys.argv else d
model, deadline = arg("--model", "qwen3"), float(arg("--deadline", 900))
T.fill_original_refs()
tmp = tempfile.mkdtemp(); proj = tempfile.mkdtemp(dir=os.path.expanduser("~")); T.write_project(proj, "base")
env = {**os.environ, "NPU_LAB_TRASH": tmp + "/trash", **{f"NPU_LAB_{k}": f"{tmp}/{v}" for k, v in dict(RULES="rules.json", DOCS="d", ROUTING="r.json", SETTINGS="s.json", CHATS="c", FACTS="f.json").items()}}
import json; os.environ["NPU_LAB_RULES"] = tmp + "/rules.json"; import search       # seed the temp rules with the limits the 4B needs (the defaults leave it no room for a context)
json.dump({**search.DEFAULT_RULES, "llm_vram_gb": 5.0, "context_tokens": 8192, "gpu_share": 0.0, "compact_at": 0}, open(tmp + "/rules.json", "w"))
p = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "src", "chat_cli.py"), "--model", model, "--project", proj, "--agent", "--tokens", "400"], cwd=ROOT,
                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
log, buf, t0 = [], "", time.time()
fails = 0


def check(cond, msg):
    global fails
    print(("PASS " if cond else "FAIL ") + msg, flush=True); fails += not cond


def send(s):
    p.stdin.write((s + "\n").encode()); p.stdin.flush(); log.append(f"\n<<< {s}\n")


def pump(wait=1.0):
    global buf
    r, _, _ = select.select([p.stdout], [], [], wait)
    if r:
        d = os.read(p.stdout.fileno(), 4096).decode(errors="replace")
        buf += d; log.append(d)
        return d
    return ""


def until(*needles, timeout):
    """Read until one of the needles shows up in the unread output; returns it (or None on timeout / process end)."""
    global buf
    end = time.time() + timeout
    while time.time() < end and time.time() - t0 < deadline:
        for n in needles:
            if n in buf:
                buf = buf[buf.index(n) + len(n):]; return n
        if p.poll() is not None and not pump(0.2):
            return None
        pump(1.0)
    return None


def read(f): return open(os.path.join(proj, f)).read()


try:
    check(until("npu-lab chat", timeout=300) is not None, "the chat starts")
    check(until("agent ▸ ", timeout=900) is not None, "the model loads and the agent prompt appears")
    # scenario 1: task 7, always yes
    send(T.TASKS[7]["en"]); confirms = 0
    while True:
        n = until("Do it? [y/N] ", "agent finished", "agent paused", "agent stopped", timeout=600)
        if n == "Do it? [y/N] ":
            confirms += 1; send("y")
        else:
            break
    ok, why = T.run_check(proj, 7)
    check(n == "agent finished", f"scenario 1: the agent finished ({confirms} confirmation(s) answered y)")
    check(ok, f"scenario 1: the task's own check passes ({why or 'ok'})")
    # scenario 2: task 6, answer n
    until("agent ▸ ", timeout=60)
    before = read("util.py"); send(T.TASKS[6]["en"])
    n = until("Do it? [y/N] ", "agent finished", timeout=600)
    check(n == "Do it? [y/N] ", "scenario 2: an edit waits for the answer before writing")
    check(read("util.py") == before, "scenario 2: nothing is written while the question is open")
    if n == "Do it? [y/N] ":
        send("n")
        check(until("nothing was changed", timeout=60) is not None, "scenario 2: answering n stops the agent")
        check(read("util.py") == before, "scenario 2: the file is untouched after n")
    send("/quit")
finally:
    try: p.stdin.close()
    except Exception: pass
    time.sleep(2); p.terminate()
    open(os.path.join(ROOT, "bench", ".cli_smoke.txt"), "w").write("".join(log))
    shutil.rmtree(proj, ignore_errors=True); shutil.rmtree(tmp, ignore_errors=True)
print("cli smoke ok" if not fails else f"{fails} FAILED")
sys.exit(1 if fails else 0)
