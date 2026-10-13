"""Offline test of remembered facts (no model, no NPU). usage: python tests/test_facts.py"""
import os, sys, tempfile
os.environ["NPU_LAB_FACTS"] = os.path.join(tempfile.mkdtemp(), "facts.json")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import facts

assert facts.load() == [] and facts.as_system("BASE") == "BASE"
assert facts.save(["  Uso   CachyOS ", "", "Prefiero Python"]) == ["Uso CachyOS", "Prefiero Python"]
s = facts.as_system("BASE")
assert s.startswith("BASE") and "- Uso CachyOS" in s and "- Prefiero Python" in s
for bad in ("x", [1], ["x" * 201], ["a"] * 31):
    try:
        facts.save(bad); raise SystemExit(f"accepted {bad!r:.30}")
    except ValueError:
        pass
assert facts.load() == ["Uso CachyOS", "Prefiero Python"]      # a refused save changes nothing
print("facts ok")
