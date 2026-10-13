"""Offline test: the saved model state (derived from what you read) is private to the owner, and folders/files made by an older version are fixed on start. usage: python tests/test_statecache.py"""
import os, stat, sys, tempfile
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import statecache
try:
    import torch
except ImportError:                                               # the CI image has no torch: only the permission part runs
    torch = None

mode = lambda p: stat.S_IMODE(os.stat(p).st_mode)
root = tempfile.mkdtemp()
old = os.path.join(root, "qwen35_2b_1"); os.makedirs(old); os.chmod(old, 0o755)          # what the previous version left behind
open(os.path.join(old, "abc.pt"), "w").write("x"); os.chmod(os.path.join(old, "abc.pt"), 0o644)
open(os.path.join(old, "index.json"), "w").write("{}"); os.chmod(os.path.join(old, "index.json"), 0o644)
c = statecache.StateCache(root, "qwen35_2b_1", 1e9)
assert mode(root) == 0o700 and mode(old) == 0o700 and mode(os.path.join(old, "abc.pt")) == 0o600 and mode(os.path.join(old, "index.json")) == 0o600, "existing ones are fixed on start"
new = statecache.StateCache(root, "other_model", 1e9)
assert mode(new.dir) == 0o700, "a new model folder is private"


if torch is None:
    print("ok (no torch: only folder permissions and the fix-on-start were checked)"); sys.exit(0)


class Core:                                                       # the minimum save() reads
    K = torch.zeros(1, 1, 8192, 4); V = torch.zeros(1, 1, 8192, 4); snaps = {4096: ({0: torch.zeros(1)}, {0: torch.zeros(1)})}


ids = list(range(5000))
assert new.save(ids, 4096, Core()) is True
files = [f for f in os.listdir(new.dir)]
assert any(f.endswith(".pt") for f in files) and all(mode(os.path.join(new.dir, f)) == 0o600 for f in files), f"saved files are 0600: {[(f, oct(mode(os.path.join(new.dir, f)))) for f in files]}"
assert not any(f.endswith(".tmp") for f in files)
print("ok: state cache folders 0700, files 0600, old ones fixed")
