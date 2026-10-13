"""Offline test of the project map. usage: python tests/test_repomap.py"""
import os, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import repomap

d = tempfile.mkdtemp()
os.makedirs(d + "/pkg"); os.makedirs(d + "/.git")
open(d + "/a.py", "w").write('class S:\n    """Store."""\n    def add(self, name, qty=1, *a, key=None, **kw):\n        """Add one."""\n    def _hidden(self): pass\n\ndef price(p, rate=0.21):\n    """With tax."""\n')
open(d + "/pkg/b.py", "w").write("async def go(x): pass\n")
open(d + "/.git/c.py", "w").write("def never(): pass\n")
open(d + "/bad.py", "w").write("def (:\n")
m = repomap.build(d)
assert "a.py:" in m and "  class S - Store." in m and "    def add(self, name, qty=1, *a, key=None, **kw) - Add one." in m, m
assert "def price(p, rate=0.21) - With tax." in m and "pkg/b.py:" in m and "def go(x)" in m, m
assert "_hidden" not in m and "never" not in m and "bad.py" not in m, m
short = repomap.build(d, max_chars=120)
assert len(short) <= 160 and " - " not in short.split("map cut")[0], short                # docstrings go first, then the tail
print("repomap ok")
