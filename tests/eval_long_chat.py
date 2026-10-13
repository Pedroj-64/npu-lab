"""Does a model hold up in a long conversation? Starts its own server (Assisted profile), loads one model and plays a scripted chat that
mixes general writing and code, with facts planted early and asked about late, a small context window so the conversation gets compacted,
and a code edit that refers to code written many turns earlier. Uses the NPU and GPU: run it alone. Bounded: --turns, a per-request
timeout and a global deadline, so it cannot hang.
usage: python tests/eval_long_chat.py [--model qwen35s|qwen35m|qwen35l] [--turns 24] [--tokens 90] [--window 2048] [--deadline 900]
Prints one line per turn (seconds, tokens read/reused, decode tok/s) and the checks; exit 1 when a check fails."""
import http.client, json, os, subprocess, sys, tempfile, time

arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
MODEL, TURNS, TOKENS, WINDOW, DEADLINE = arg("--model", "qwen35s"), arg("--turns", 24), arg("--tokens", 90), arg("--window", 2048), arg("--deadline", 900)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); T = "l" * 43
tmp = tempfile.mkdtemp()
p = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "src", "serve.py")], cwd=ROOT, stdout=subprocess.PIPE, text=True,
                     env={**os.environ, "NPU_LAB_TOKEN": T, **{f"NPU_LAB_{k}": f"{tmp}/{v}" for k, v in dict(RULES="rules.json", DOCS="d", ROUTING="r.json", SETTINGS="s.json", CHATS="c", FACTS="f.json").items()}})
port = json.loads(p.stdout.readline())["port"]
t0 = time.time()


def call(path, body=None, timeout=300):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST" if body is not None else "GET", path, json.dumps(body) if body is not None else None, {"Authorization": "Bearer " + T, "Host": f"127.0.0.1:{port}"})
    r = c.getresponse(); return r.status, json.loads(r.read())


HISTORY, rows = [], []


def say(prompt):
    """One turn like the web page does it: the answer is complete at `done`; notes from a compaction may arrive afterwards on the same stream."""
    if time.time() - t0 > DEADLINE:
        raise SystemExit(f"deadline of {DEADLINE}s reached at turn {len(rows) + 1}")
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=240)
    c.request("POST", "/chat", json.dumps({"prompt": prompt, "tokens": TOKENS, "history": HISTORY, "docs": "off", "web": "off", "mine": "off"}), {"Authorization": "Bearer " + T, "Host": f"127.0.0.1:{port}"})
    r = c.getresponse(); out, st, kind, t, answered, ans = "", {}, "", time.time(), None, ""
    if r.status != 200:
        raise SystemExit(f"/chat -> {r.status} {r.read()[:200]!r}")
    for line in r:
        m = json.loads(line)
        out += m.get("token", "")
        if m.get("done") is True:
            st, answered = m.get("stats", {}), time.time() - t
            ans = out.split("</think>")[-1].strip()
            HISTORY.extend([{"role": "user", "content": prompt}, {"role": "assistant", "content": ans}])
        if m.get("compacted"):
            s = m["compacted"]; kind = "before" if answered is None else "after"
            print("   notes after compaction:", repr(s["summary"][:400]), flush=True)
            HISTORY[:] = [{"role": "user", "content": "[Notes of the earlier conversation]\n" + s["summary"]}, {"role": "assistant", "content": "Understood, I will use those notes."}, *HISTORY[-s["kept"]:]]
        if m.get("warning"):
            print("   warning:", m["warning"][:200], flush=True)
        if m.get("error"):
            raise SystemExit("engine error: " + str(m["error"]))
    total = time.time() - t
    rows.append((answered, st.get("prefill_tokens", 0), st.get("reused_tokens", 0), st.get("decode_tokens", 0) / max(st.get("decode_s", 1e-9), 1e-9), kind, total - answered))
    print(f"turn {len(rows):2d}: {answered:5.1f}s  read {rows[-1][1]:5d} (reused {rows[-1][2]:5d})  {rows[-1][3]:4.1f} tok/s{f'  [compacted {kind}, tidy {total - answered:.0f}s]' if kind else ''}  | {ans[:70]!r}", flush=True)
    return ans


def runs(code, call, want):
    """Run the first python block of an answer (answers are the user's own model output, run in a child process with a 10 s limit) and compare call() with want."""
    import re
    m = re.search(r"```(?:python)?\n(.*?)```", code, re.S)
    src = m.group(1) if m else code
    try:
        r = subprocess.run([sys.executable, "-I", "-c", src + f"\nprint(repr({call}))"], capture_output=True, text=True, timeout=10, cwd=tempfile.mkdtemp())
        return r.stdout.strip().splitlines()[-1:] == [repr(want)]
    except (subprocess.TimeoutExpired, OSError):
        return False


FILLER = ["Write one sentence about why backups matter.", "Give me a one-line shell command to count lines in a file.", "Explain in two sentences what a mutex is.",
          "Write a short Python function is_even(n).", "Name three uses of the fish shell.", "What is the difference between a list and a tuple in Python? Be brief.",
          "Write one sentence about the sea.", "Give a one-line git command to undo the last commit but keep the changes.", "Explain what an index in a database is, in two sentences.",
          "Write a Python one-liner to reverse a string.", "Name two advantages of unit tests.", "Write one sentence about autumn."]
fails = []
try:
    assert call("/profile", {"profile": "assisted"})[0] == 200
    assert call("/rules", {"context_tokens": WINDOW, "llm_vram_gb": 5.0, "gpu_share": 0.0, "compact_at": 0.7})[0] == 200
    s, b = call("/mode", {"mode": "llm", "model": MODEL}, timeout=400)
    if s != 200:
        raise SystemExit(f"load failed: {b}")
    print(f"{MODEL} loaded in {time.time() - t0:.0f}s, window {call('/status')[1].get('context')} tokens", flush=True)
    say("My project is called Nebula, it listens on port 8431 and my cat is called Zeta. Reply only with OK.")
    first = say("Write a Python function slugify(text) that lowercases text and replaces spaces with '-'. Only the code.")
    for i in range(TURNS - 4):
        say(FILLER[i % len(FILLER)])
    edit = say("Change the slugify function you wrote at the start so it takes a second parameter sep='-' used instead of the hard-coded dash. Only the code.")
    recall = say("What is my project called, what port does it listen on and what is my cat called? One line.")
    for name, ok in (("remembers the project name", "nebula" in recall.lower()), ("remembers the port", "8431" in recall), ("remembers the cat", "zeta" in recall.lower()),
                     ("first slugify works", runs(first, "slugify('Hello World')", "hello-world")),
                     ("edited slugify works with the default dash", runs(edit, "slugify('Hello World')", "hello-world")),
                     ("edited slugify uses sep", runs(edit, "slugify('Hello World', sep='_')", "hello_world"))):
        print(("PASS " if ok else "FAIL ") + name); not ok and fails.append(name)
    a, b = [r[0] for r in rows[2:7]], [r[0] for r in rows[-6:-2]]
    slow = (sum(b) / len(b)) / max(sum(a) / len(a), 1e-9)
    print(("PASS" if slow < 2.0 else "FAIL") + f" late turns are not much slower than early ones ({slow:.2f}x, limit 2x)"); slow >= 2.0 and fails.append("slowdown")
    print(f"compactions: {sum(bool(r[4]) for r in rows)} ({sum(r[4] == 'after' for r in rows)} after the answer), longest wait before an answer {max(r[0] for r in rows):.1f}s vs median {sorted(r[0] for r in rows)[len(rows) // 2]:.1f}s, total {time.time() - t0:.0f}s")
finally:
    call("/mode", {"mode": "none"}, timeout=60) if p.poll() is None else None
    p.terminate()
sys.exit(1 if fails else 0)
