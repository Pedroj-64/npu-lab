"""Offline test for src/devhelpers.py with a throw-away git repo. usage: python tests/test_devhelpers.py"""
import os, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import devhelpers as D

def sh(*a, cwd): subprocess.run(a, cwd=cwd, check=True, capture_output=True)
root = tempfile.mkdtemp(); repo = os.path.join(root, "proj"); os.makedirs(repo)
sh("git", "init", "-q", cwd=repo); open(os.path.join(repo, "a.py"), "w").write("x = 1\n")
sh("git", "add", ".", cwd=repo); sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", "x", cwd=repo)
fake = "hf_" + "A1b2C3d4E5" * 3
open(os.path.join(repo, "a.py"), "w").write(f'x = 1\nTOKEN = "{fake}"\nprint(x)\n')
open(os.path.join(repo, "new.env"), "w").write("api_key = '" + "abcdefgh" + "12345678" + "'\n")      # built at run time so this file does not look like a leak itself
assert D.repo_ok(repo, [root]) == os.path.realpath(repo)
for bad in ("/tmp", repo + "/..", "/etc"):
    try: D.repo_ok(bad, [root]); raise SystemExit(f"accepted {bad}")
    except D.HelperError: pass
files, diff = D.pending_diff(repo)
assert "a.py" in files and "+TOKEN" in diff, (files, diff)
f = D.secret_findings(diff)
assert f == [{"kind": "Hugging Face token", "line": 2}], f
assert fake not in str(f), "the secret value must never be returned"
u = D.scan_untracked(repo); assert u and u[0]["file"] == "new.env", u
assert D.secret_findings("+++ b/x\n@@ -1 +1 @@\n+hello world\n") == []
assert "Files: a.py" in D.commit_prompt(["a.py", "new.env"], diff) or "a.py, new.env" in D.commit_prompt(["a.py", "new.env"], diff)
try: D.unit_log("a b; rm -rf /"); raise SystemExit("bad unit accepted")
except D.HelperError: pass
print("devhelpers ok")
