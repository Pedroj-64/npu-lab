"""Offline test of project mode: block parsing, safe all-or-nothing apply, attach. usage: python tests/test_project.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import project

root = tempfile.mkdtemp(); out = tempfile.mkdtemp()
os.makedirs(root + "/pkg"); open(root + "/pkg/a.py", "w").write("def f():\n    return 1\n"); open(root + "/b.py", "w").write("x = 1\n")
open(out + "/secret.txt", "w").write("keep\n"); os.symlink(out, root + "/link")
blk = lambda p, a, b: f"{p}\n<<<<<<< SEARCH\n{a}\n=======\n{b}\n>>>>>>> REPLACE\n"
read = lambda p: open(os.path.join(root, p)).read()

assert project.parse("text\n" + blk("pkg/a.py", "return 1", "return 2") + "more") == [("pkg/a.py", "return 1", "return 2")]
assert project.parse("```\n" + blk("`pkg/a.py`", "x", "y") + "```")[0][0] == "pkg/a.py"
assert [r["error"] for r in project.apply(root, blk("pkg/a.py", "return 1", "return 2"), dry=True)] == [""] and "return 1" in read("pkg/a.py")          # dry run writes nothing
assert project.apply(root, blk("pkg/a.py", "return 1", "return 2"))[0]["action"] == "edit" and "return 2" in read("pkg/a.py")
assert project.apply(root, blk("new/c.py", "", "y = 2"))[0]["action"] == "create" and read("new/c.py") == "y = 2\n"
# every refusal leaves ALL files untouched, even when an earlier block was fine
bad = blk("b.py", "x = 1", "x = 9") + blk("pkg/a.py", "does not exist", "z")
r = project.apply(root, bad); assert r[-1]["action"] == "failed" and read("b.py") == "x = 1\n", r
for p in ("../outside.py", "/etc/passwd", "link/secret.txt", ".git/config", "pkg/../../x.py"):
    r = project.apply(root, blk(p, "keep" if "secret" in p else "", "pwned")); assert r[-1]["action"] == "failed", (p, r)
assert open(out + "/secret.txt").read() == "keep\n" and not os.path.exists(root + "/../outside.py")      # nothing escaped through ../ or the symlink
assert project.apply(root, blk("pkg/a.py", "e", "E"))[-1]["error"].startswith("SEARCH text found") and "return 2" in read("pkg/a.py")      # an ambiguous SEARCH is refused
assert project.apply(root, blk("pkg/a.py", "", "new"))[-1]["error"].startswith("file already exists")
assert project.apply(root, "just talk")[0]["action"] == "none"
assert "Files you asked about" in project.attach("what does pkg/a.py do", root) and "def f" in project.attach("explain a.py", root)
assert project.attach("hello there", root) == "hello there"
sb = project.system_block(root); assert "SEARCH" in sb and "def f()" in sb
print("project ok")
