"""A model load that dies half way (e.g. CUDA out of memory) must give its VRAM back. No NPU: the fake loader only allocates 400 MB on the GPU, then fails.
usage: .venv/bin/python tests/test_load_failure.py"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import torch, serve, qwen_npu
if not torch.cuda.is_available():
    sys.exit("no CUDA: nothing to check")
serve.S.profile = "assisted"
def fake(*a, **k):
    held = torch.empty(400 * 2**20, dtype=torch.uint8, device="cuda")       # lives in this frame, like the weights of a half-loaded engine
    raise MemoryError("CUDA out of memory (fake)")
qwen_npu.Qwen = fake
before = torch.cuda.memory_reserved()
try:
    serve.set_mode("llm", "qwen3s")
    raise SystemExit("load should have failed")
except MemoryError:
    pass
left = torch.cuda.memory_reserved() - before
assert left < 50 * 2**20, f"{left >> 20} MB still reserved after a failed load"
assert serve.S.mode == "none" and serve.S.engine is None
print("failed load frees the GPU ok")
