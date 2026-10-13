"""How many tokens of context fit in the GPU memory budget for a Qwen3.5 model (GPU only, no NPU needed).
usage: .venv/bin/python bench/vram_ctx.py models/qwen35_4b 4.0 5.0 5.5 [--bf16]   (budgets in GB; int8 weights unless --bf16 forces bf16)"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import gc, json, torch
import qwen_npu as Q, qwen35_npu as N, gpu_qwen35 as G

d = sys.argv[1]
budgets = [float(a) for a in sys.argv[2:] if not a.startswith("--")]
c = N.text_config(json.load(open(f"{d}/config.json")))
w = Q.Weights(os.path.join(d, sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))[0]))
Q.QUANT_GPU = True
print("| budget GB | weights | max context (tokens) | weights in VRAM GB | KV bytes/token | VRAM allocated GB |\n|---:|---|---:|---:|---:|---:|", flush=True)
for b in budgets:
    try:
        g = G.GpuCore35(c, w, N.PREFIX, 262144, b, int8=False if "--bf16" in sys.argv else None)
        alloc = torch.cuda.memory_allocated() / 1e9
        kv = g.K.element_size() * g.K.shape[0] * g.K.shape[1] * g.K.shape[3] * 2
        print(f"| {b:g} | {'int8' if g.int8 else 'bf16'} | {g.max_ctx} | {alloc - g.K.numel() * 4 / 1e9:.2f} | {kv} | {alloc:.2f} |", flush=True)
    except MemoryError as e:
        print(f"| {b:g} | - | does not fit | | | {str(e)[:70]} |", flush=True)
    g = None; gc.collect(); torch.cuda.empty_cache()
