"""Offline test for src/sentinel.py with throw-away git repos. usage: python tests/test_sentinel.py"""
import os, subprocess, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import sentinel

def sh(*a, cwd): subprocess.run(a, cwd=cwd, check=True, capture_output=True)
root = tempfile.mkdtemp()
clean, dirty, pushed = (os.path.join(root, n) for n in ("clean", "dirty", "ahead"))
for r in (clean, dirty):
    os.makedirs(r); sh("git", "init", "-q", cwd=r); sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "x", cwd=r)
open(os.path.join(dirty, "a.txt"), "w").write("x"); open(os.path.join(dirty, "b.txt"), "w").write("y")
up = os.path.join(root, "upstream.git"); sh("git", "init", "-q", "--bare", up, cwd=root)
sh("git", "clone", "-q", up, pushed, cwd=root)
sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "a", cwd=pushed); sh("git", "push", "-q", "origin", "HEAD", cwd=pushed)
sh("git", "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "b", cwd=pushed)
evs = sentinel.events([root]); names = {os.path.basename(e["repo"]): e for e in evs}
assert set(names) == {"dirty", "ahead"}, names                     # clean repo is silent
assert names["dirty"]["dirty"] == 2 and names["ahead"]["ahead"] == 1, names
assert sentinel.text(names["dirty"]) == "dirty: 2 uncommitted", sentinel.text(names["dirty"])
assert names["dirty"]["oldest_days"] == 0
old = dict(names["dirty"], oldest_days=5); assert sentinel.text(old).endswith("(oldest change 5 days old)")
assert "failed" in sentinel.text({"repo": "unit:x", "dirty": 0, "ahead": 0, "unit": "x.service"})
fresh, seen = sentinel.new_only(evs, {}); assert len(fresh) == 2
fresh2, seen = sentinel.new_only(evs, seen); assert fresh2 == [], "same state is not announced twice"
open(os.path.join(dirty, "c.txt"), "w").write("z")
fresh3, _ = sentinel.new_only(sentinel.events([root]), seen); assert [os.path.basename(e["repo"]) for e in fresh3] == ["dirty"], "a changed state is"
print("sentinel ok")
