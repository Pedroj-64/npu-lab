"""Offline test of the per-model GPU share: defaults, old scalar ignored, VRAM margin refusal, and the degradation when NPU kernels for a share are not built. usage: python tests/test_shares.py"""
import os, sys, tempfile
tmp = tempfile.mkdtemp(); os.environ["NPU_LAB_RULES"] = tmp + "/rules.json"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, "src"))
import serve

# defaults and the old scalar
assert serve.share_of("qwen35m", {}) == 0.5 and serve.share_of("qwen35l", {}) == 0.0 and serve.share_of("qwen35s", {}) == 0.0
assert serve.share_of("qwen35l", {"gpu_share": 0.5}) == 0.0 and serve.share_of("qwen35m", {"gpu_share": 0.25}) == 0.5, "an old scalar is ignored, it used to apply to every model"
assert serve.share_of("qwen35m", {"gpu_share": {"qwen35m": 0.75}}) == 0.75 and serve.share_of("qwen35m", {"gpu_share": {"qwen35m": 0.0}}) == 0.0

# VRAM margin (free memory taken from nvidia-smi at load; here injected)
v = serve.vram_refusal
assert v("qwen35l", 0.0, 700, 6116) is None and v("qwen35l", 0.25, 700, 6116) is None, "4B at 0 % and 25 % leave >= 700 MiB"
m = v("qwen35l", 0.5, 700, 6116); assert m and "Pick a lower GPU share" in m and "389" not in m and "361" in m, m
assert v("qwen35l", 0.25, 1024, 6116), "a 1 GB margin refuses the 4B at 25 % (839 MiB left)"
assert v("qwen35l", 0.0, 700, 5000), "another program already holds the GPU memory"
assert v("qwen35l", 0.0, 700, None) is None and v("qwen35s", 0.0, 700, 100) is None, "no nvidia-smi, or no measured peak: only the check after loading applies"

# limits validation
for bad in (-1, 5000, "x", True):
    try: serve.set_rules({"vram_margin_mib": bad}); raise SystemExit(f"accepted {bad!r}")
    except serve.Bad as e: assert e.code == 400 if hasattr(e, "code") else True
assert serve.set_rules({"vram_margin_mib": 800})["vram_margin_mib"] == 800.0

# kernels not built: the page must be told, with the command
import importlib.util
if os.path.exists(os.path.join(ROOT, "models/qwen35_2b/config.json")) and importlib.util.find_spec("torch"):          # CI has neither the model files nor torch
    empty = tempfile.mkdtemp()
    need = serve.missing_kernels("qwen35m", 0.75, empty)
    assert len(need) > 5 and all(n.startswith("c4_") for n in need)
    for n in need: open(os.path.join(empty, n + ".xclbin"), "w").close()
    assert serve.missing_kernels("qwen35m", 0.75, empty) == []
    os.remove(os.path.join(empty, need[0] + ".xclbin")); assert serve.missing_kernels("qwen35m", 0.75, empty) == [need[0]]
    real = serve.missing_kernels
    serve.missing_kernels = lambda model, share, build_dir=None: ["c4_1x1"] * 3 if share == 0.75 else []          # as if the 75 % kernels were not built
    try:
        view = serve.rules_view()
    finally:
        serve.missing_kernels = real
    assert view["gpu_share_kernels_missing"]["qwen35m"]["0.75"] == 3 and view["gpu_share_kernels_missing"]["qwen35m"]["0.5"] == 0
    cmd = view["gpu_share_build_cmd"]["qwen35m"]; assert "build_matvec.py --shared4" in cmd and "--model models/qwen35_2b" in cmd and "{share}" in cmd
    assert view["gpu_share_effective"] == {"qwen35s": 0.0, "qwen35m": 0.5, "qwen35l": 0.0} and view["vram_margin_mib"] == 800.0
print("ok: per-model shares, old scalar ignored, VRAM margin, missing-kernel warning")
