"""Offline test of the agent's RUN action and the DONE guard (no model): RUN only runs the configured command after confirmation; with the guard on, DONE
needs a change, a compiling project, the tests run after the last change and passing, and (guard=model) a probability from the decision backend.
usage: python tests/test_agent_guard.py"""
import os, shutil, sys, tempfile
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["NPU_LAB_TRASH"] = tempfile.mkdtemp(); sys.path.insert(0, os.path.join(ROOT, "src"))
import agent

CMD = f"{sys.executable} -c \"import a; assert a.f() == 2\""
EDIT = "a.py\n<<<<<<< SEARCH\nreturn 1\n=======\nreturn 2\n>>>>>>> REPLACE"
BREAK = "a.py\n<<<<<<< SEARCH\nreturn 1\n=======\nreturn 3\n>>>>>>> REPLACE"


class Fake:                                           # stands in for a decision backend: fixed p(YES), counts the calls
    def __init__(self, p): self.p, self.calls = p, 0
    def probs(self, state, q): self.calls += 1; self.last = state; return {"YES": self.p, "NO": 1 - self.p}


def fresh():
    d = tempfile.mkdtemp(dir=os.path.expanduser("~")); open(d + "/a.py", "w").write("def f():\n    return 1\n"); return d


dirs = []
try:
    # RUN: needs a configured command, only that command, and confirmation
    p = fresh(); dirs.append(p)
    r = agent.run(p, "RUN"); assert not r["ok"] and "no test command" in r["shown"], r
    r = agent.run(p, "RUN rm -rf .", test_command=CMD, confirmed=True); assert not r["ok"] and not r["executed"] and os.path.exists(p + "/a.py"), r
    r = agent.run(p, "RUN", test_command=CMD); assert r["needs_confirm"] and not r["executed"], r
    r = agent.run(p, "RUN", test_command=CMD, confirmed=True); assert r["ok"] and r["executed"] and "exit code 1" in r["shown"] and "AssertionError" in r["shown"], r      # failing tests are a valid action with a visible output
    assert "RUN" in agent.system_block(p, CMD) and "RUN" not in agent.system_block(p)
    agent._STATE.clear()

    # guard off (the default): behaviour unchanged, DONE with no change is accepted
    p = fresh(); dirs.append(p)
    assert agent.run(p, "DONE x")["done"]

    # guard = objective
    p = fresh(); dirs.append(p)
    r = agent.run(p, "DONE x", guard="objective"); assert not r["done"] and "no file changed" in r["shown"], r
    agent.run(p, EDIT, confirmed=True, guard="objective", test_command=CMD)
    r = agent.run(p, "DONE x", guard="objective", test_command=CMD); assert not r["done"] and "run the project's tests" in r["shown"], r        # tests not run after the change
    agent.run(p, "RUN", confirmed=True, guard="objective", test_command=CMD)
    r = agent.run(p, "DONE x", guard="objective", test_command=CMD); assert r["done"], r
    assert p not in agent._STATE                                                                                                                    # next task: new baseline
    # tests that fail block DONE; a later edit makes the old result stale
    q = fresh(); dirs.append(q)
    agent.run(q, BREAK, confirmed=True, guard="objective", test_command=CMD)
    agent.run(q, "RUN", confirmed=True, guard="objective", test_command=CMD)
    r = agent.run(q, "DONE x", guard="objective", test_command=CMD); assert not r["done"] and "tests fail" in r["shown"], r
    agent.run(q, "a.py\n<<<<<<< SEARCH\nreturn 3\n=======\nreturn 2\n>>>>>>> REPLACE", confirmed=True, guard="objective", test_command=CMD)
    r = agent.run(q, "DONE x", guard="objective", test_command=CMD); assert not r["done"] and "run the project's tests" in r["shown"], r         # stale result
    agent.run(q, "RUN", confirmed=True, guard="objective", test_command=CMD)
    assert agent.run(q, "DONE x", guard="objective", test_command=CMD)["done"]
    # no tests configured: a change and a compiling project are enough for "objective"
    s = fresh(); dirs.append(s)
    agent.run(s, EDIT, confirmed=True, guard="objective")
    assert agent.run(s, "DONE x", guard="objective")["done"]

    # guard = model: objective checks first (backend not even called), then the probability against tau
    f = Fake(0.9); m = fresh(); dirs.append(m)
    r = agent.run(m, "DONE x", guard="model", backend=f); assert not r["done"] and f.calls == 0, (r, f.calls)
    agent.run(m, EDIT, confirmed=True, guard="model", backend=f)
    f.p = 0.2; r = agent.run(m, "DONE x", guard="model", backend=f, task="make f return 2", tau=0.5); assert not r["done"] and "p(done)=0.20" in r["shown"] and f.calls == 1, r
    assert "Task: make f return 2" in f.last and "return 2" in f.last and "unverified" in f.last, f.last                                              # short evidence: task, claim marked unverified, diff
    f.p = 0.8; r = agent.run(m, "DONE x", guard="model", backend=f, tau=0.5); assert r["done"] and abs(r["p_done"] - 0.8) < 1e-9, r
    print("agent guard ok")
finally:
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
    shutil.rmtree(os.environ["NPU_LAB_TRASH"], ignore_errors=True)
