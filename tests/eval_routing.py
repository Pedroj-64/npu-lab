"""Offline regression for the rule router and the web filter (the field-report rows). usage: python tests/eval_routing.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import router

# (question, expected route)
CASES = [
    ("how do I make a class in TypeScript", "none"),
    ("what happened on 5 March 2026 in Spain", "web"),
    ("qué pasó ayer en México", "web"),
    ("what did we talk about in this chat", "history"),
    ("de qué hablamos antes", "history"),
    ("and in Python?", "history"),
    ("summarise the above", "history"),
    ("install an AUR package with paru", "docs"),
    ("how do I list files by size in the terminal", "docs"),
    ("como instalo paquetes con pacman", "docs"),
    ("latest version of the linux kernel", "web"),
    ("how to make homemade soap, ingredients only", "docs"),    # not terminal-related, but docs are gated by relevance, so nothing is injected
]


def route(q):
    d = router.decide(q)
    return "history" if d["history"] else "web" if d["web"] else "docs" if d["docs"] else "none"


bad = [(q, want, route(q)) for q, want in CASES if route(q) != want]
for q, want, got in bad:
    print(f"FAIL {q!r}: want {want}, got {got}")

res = [{"url": "https://duckduckgo.com/?q=x", "snippet": "a", "title": ""}, {"url": "https://www.bbc.com/a", "snippet": "a", "title": ""},
       {"url": "https://bbc.com/b", "snippet": "b", "title": ""}, {"url": "https://example.org/", "snippet": "", "title": ""}]
assert [r["url"] for r in router.clean_web(res)] == ["https://www.bbc.com/a"], router.clean_web(res)
assert router.search_query("and in Python?", [{"role": "user", "content": "how do I read a file"}]).startswith("how do I read a file")
assert router.resolve("auto", True) and not router.resolve("off", True) and router.resolve("on", False) and router.resolve(True, False)
print(f"{len(CASES) - len(bad)}/{len(CASES)} routes ok")
sys.exit(1 if bad else 0)
