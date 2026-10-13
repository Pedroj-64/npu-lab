"""End to end project mode with a real model: set a project folder, ask for an edit in plain words, apply the answer through /project/apply and run the check.
Uses the NPU and GPU: run alone. Bounded (one task, ~2-3 min on the 4B).
usage: python tests/eval_project_mode.py [--model qwen35l] [--task 1] [--tokens 450]"""
import http.client, json, os, shutil, subprocess, sys, tempfile, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "bench"))
import edit_eval as E

arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
MODEL, TID, TOKENS = arg("--model", "qwen35l"), arg("--task", 1), arg("--tokens", 450)
T = "p" * 43; tmp = tempfile.mkdtemp(); proj = tempfile.mkdtemp(dir=os.path.expanduser("~"))       # the project must be inside the home
E.project(proj, 20)
p = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "src", "serve.py")], cwd=ROOT, stdout=subprocess.PIPE, text=True,
                     env={**os.environ, "NPU_LAB_TOKEN": T, **{f"NPU_LAB_{k}": f"{tmp}/{v}" for k, v in dict(RULES="rules.json", DOCS="d", ROUTING="r.json", SETTINGS="s.json", CHATS="c", FACTS="f.json").items()}})
port = json.loads(p.stdout.readline())["port"]


def call(path, body, timeout=400):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("POST", path, json.dumps(body), {"Authorization": "Bearer " + T, "Host": f"127.0.0.1:{port}"}); r = c.getresponse(); return r.status, json.loads(r.read())


ok = False
try:
    call("/profile", {"profile": "assisted"}); call("/rules", {"context_tokens": 8192, "llm_vram_gb": 5.0, "gpu_share": {"qwen35m": 0.0, "qwen35l": 0.0}, "compact_at": 0, "project_root": proj})
    s, b = call("/mode", {"mode": "llm", "model": MODEL})
    assert s == 200, b
    target, instr = E.TASKS[TID][0], E.TASKS[TID][1]
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600); t = time.time()
    c.request("POST", "/chat", json.dumps({"prompt": f"In {target}: {instr}", "tokens": TOKENS, "history": [], "docs": "off", "web": "off", "mine": "off"}), {"Authorization": "Bearer " + T, "Host": f"127.0.0.1:{port}"})
    out, st = "", {}
    for line in c.getresponse():
        m = json.loads(line); out += m.get("token", "")
        if m.get("done") is True: st = m.get("stats", {}); break
    ans = out.split("</think>")[-1]
    print(f"answered in {time.time() - t:.0f}s, prompt {st.get('prefill_tokens')} tokens, wrote {st.get('decode_tokens')} tokens")
    s, r = call("/project/apply", {"text": ans, "dry": True}); print("dry run:", r["results"])
    assert all(x["action"] in ("edit", "create") for x in r["results"]), "the answer is not an applicable edit"
    call("/project/apply", {"text": ans})
    r = subprocess.run([sys.executable, "-c", "import sys; sys.path.insert(0, '.')\n" + E.TASKS[TID][2]], cwd=proj, capture_output=True, text=True, timeout=20)
    ok = r.returncode == 0; print(("PASS" if ok else "FAIL") + f" task {TID} through project mode  {r.stderr.strip()[-120:]}")
finally:
    try: call("/mode", {"mode": "none"}, 60)
    except Exception: pass
    p.terminate(); shutil.rmtree(proj, ignore_errors=True)
sys.exit(0 if ok else 1)
