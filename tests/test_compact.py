"""Offline test of the incremental compaction with a fake model (no NPU, ~1 s). usage: python tests/test_compact.py"""
import os, sys
os.environ["NPU_LAB_FACTS"] = "/nonexistent/f.json"
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import serve


class Tok:                                                   # one token per character keeps the arithmetic obvious
    def encode(self, t, add_special_tokens=False):
        return type("E", (), {"ids": [ord(c) for c in t]})()
    def decode(self, ids):
        return "".join(map(chr, ids))


class Fake:
    tok, calls = Tok(), []
    def complete(self, prompt, system, n):
        self.calls.append((prompt, n)); return f"- notes v{len(self.calls)}"


serve.S.engine = Fake()
turn = lambda i: [{"role": "user", "content": f"question {i} " + "x" * 120}, {"role": "assistant", "content": f"answer {i} " + "y" * 120}]
H = [m for i in range(16) for m in turn(i)]
H[0]["content"] = "My project is called Nebula on port 8431"
h, info = serve.compact(H, 6000, 0.7, 3, lambda **k: None)
assert info and info["kept"] == 6 and len(Fake.calls) == 1 and Fake.calls[0][1] == serve.NOTES_TOKENS, (info, Fake.calls)
assert h[0]["content"].startswith(serve.NOTES_HEAD) and "port 8431" in h[0]["content"] and "- notes v1" in h[0]["content"]
# second compaction: the first notes are the starting point, only the newer turns are read, and the pinned line is not duplicated
h2 = h + [m for i in range(16, 28) for m in turn(i)]
h3, info2 = serve.compact(h2, 6000, 0.7, 3, lambda **k: None)
prompt2 = Fake.calls[-1][0]
assert "Previous notes:\n- notes v1" in prompt2 and "question 0" not in prompt2 and "question 16" in prompt2, prompt2[:300]
assert h3[0]["content"].count("port 8431") == 1 and info2["kept"] == 6
# nothing new to read (only notes + recent turns above the threshold): left alone, no model call
n = len(Fake.calls)
same, none = serve.compact(h3[:2] + h3[-6:], 1, 0.7, 3, lambda **k: None)
assert none is None and len(Fake.calls) == n
print("compact ok")
