"""Offline test of the terminal agent's server side (no model): confirmation gating, trash instead of erase, refusals, DONE checks, and the HTTP endpoint.
usage: python tests/test_agent.py"""
import http.client, json, os, shutil, subprocess, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["NPU_LAB_TRASH"] = tempfile.mkdtemp(); sys.path.insert(0, os.path.join(ROOT, "src"))
import agent

proj = tempfile.mkdtemp(dir=os.path.expanduser("~"))
try:
    open(proj + "/a.py", "w").write("def f():\n    return 1\n"); open(proj + "/old.py", "w").write("x = 1\n")
    blk = "a.py\n<<<<<<< SEARCH\nreturn 1\n=======\nreturn 2\n>>>>>>> REPLACE"
    read = lambda n: open(os.path.join(proj, n)).read()
    r = agent.run(proj, "READ a.py"); assert r["ok"] and r["executed"] and not r["needs_confirm"] and "return 1" in r["shown"]
    assert agent.run(proj, "LIST")["shown"].split() == ["a.py", "old.py"]
    # writes wait for confirmation
    r = agent.run(proj, blk); assert r["ok"] and r["needs_confirm"] and not r["executed"] and "return 1" in read("a.py"), r
    r = agent.run(proj, blk, confirmed=True); assert r["ok"] and r["executed"] and "return 2" in read("a.py"), r
    r = agent.run(proj, "DELETE old.py"); assert r["needs_confirm"] and os.path.exists(proj + "/old.py"), r
    r = agent.run(proj, "DELETE old.py", confirmed=True); assert r["executed"] and not os.path.exists(proj + "/old.py")
    trashed = [os.path.join(d, f) for d, _, fs in os.walk(os.environ["NPU_LAB_TRASH"]) for f in fs]
    assert len(trashed) == 1 and open(trashed[0]).read() == "x = 1\n" and trashed[0].endswith("old.py")        # recoverable, outside the project
    # refusals: nothing happens and the reply says why
    for bad in ("READ ../../etc/passwd", "DELETE /etc/passwd", "READ nothere.py", "hello", "DELETE .git/config"):
        r = agent.run(proj, bad, confirmed=True); assert not r["ok"] and not r["executed"], (bad, r)
    r = agent.run(proj, "x.py\n<<<<<<< SEARCH\nzzz\n=======\nq\n>>>>>>> REPLACE", confirmed=True); assert not r["ok"]
    assert agent.run(proj, "a.py\n<<<<<<< SEARCH\nreturn 3\n=======\nq\n>>>>>>> REPLACE")["shown"].count("currently contains") == 1       # the real file text comes back on a mismatch
    # DONE is checked, not believed
    assert agent.run(proj, "DONE finished")["done"]
    assert not agent.run(proj, "DONE finished", last_ok=False)["done"]
    open(proj + "/bad.py", "w").write("def (:\n")
    r = agent.run(proj, "DONE finished"); assert not r["done"] and "does not compile" in r["shown"], r
    os.remove(proj + "/bad.py")
    # through HTTP
    T = "a" * 43; tmp = tempfile.mkdtemp()
    p = subprocess.Popen([sys.executable, "-u", os.path.join(ROOT, "src", "serve.py")], cwd=ROOT, stdout=subprocess.PIPE, text=True,
                         env={**os.environ, "NPU_LAB_TOKEN": T, **{f"NPU_LAB_{k}": f"{tmp}/{v}" for k, v in dict(RULES="rules.json", DOCS="d", ROUTING="r.json", SETTINGS="s.json", CHATS="c", FACTS="f.json").items()}})
    try:
        port = json.loads(p.stdout.readline())["port"]
        def call(path, body):
            c = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
            c.request("POST", path, json.dumps(body), {"Authorization": "Bearer " + T, "Host": f"127.0.0.1:{port}"}); r = c.getresponse(); return r.status, json.loads(r.read())
        assert call("/agent/run", {"text": "LIST"})[0] == 409                                         # no project set
        assert call("/rules", {"project_root": proj})[0] == 200
        s, r = call("/agent/run", {"text": "READ a.py"}); assert s == 200 and r["ok"] and "return 2" in r["shown"], r
        e = "a.py\n<<<<<<< SEARCH\nreturn 2\n=======\nreturn 9\n>>>>>>> REPLACE"
        s, r = call("/agent/run", {"text": e}); assert r["needs_confirm"] and "return 2" in read("a.py"), r
        s, r = call("/agent/run", {"text": e, "confirm": "yes"}); assert r["needs_confirm"] and "return 2" in read("a.py"), r        # only a real true confirms
        s, r = call("/agent/run", {"text": e, "confirm": True}); assert r["executed"] and "return 9" in read("a.py"), r
    finally:
        p.terminate()
    print("agent ok")
finally:
    shutil.rmtree(proj, ignore_errors=True); shutil.rmtree(os.environ["NPU_LAB_TRASH"], ignore_errors=True)
