"""Real GPU memory of a Qwen3.5 GPU core at a given llm_vram_gb budget: what the driver reports (CUDA context included) after loading and after reading a long prompt
(the peak of the activations). GPU only, no NPU. The CUDA graphs of decode (~0.65 GB with the 4B, captured on the first tokens) come on top and are not measured here.
usage: .venv/bin/python bench/vram_peak.py models/qwen35_4b 4.0 4.5 [--prompt 4096]"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import gc, json, torch
import qwen_npu as Q, qwen35_npu as N, gpu_qwen35 as G

d = sys.argv[1]; budgets = [float(a) for i, a in enumerate(sys.argv[2:], 2) if a.replace(".", "").isdigit() and sys.argv[i - 1] != "--prompt"]
n = int(sys.argv[sys.argv.index("--prompt") + 1]) if "--prompt" in sys.argv else 4096
c = N.text_config(json.load(open(f"{d}/config.json")))
w = Q.Weights(os.path.join(d, sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))[0])); Q.QUANT_GPU = True
used = lambda: (torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 2**30
print(f"card {torch.cuda.mem_get_info()[1] / 2**30:.2f} GiB, in use before: {used():.2f} GiB\n| budget GB | weights | window | after load GiB | after {n}-token prompt GiB |\n|---:|---|---:|---:|---:|", flush=True)
for b in budgets:
    g = None
    try:
        g = G.GpuCore35(c, w, N.PREFIX, 262144, b); a = used()
        g.prefill(torch.randn(min(n, g.max_ctx), g.H, dtype=torch.bfloat16, device="cuda") * 0.1)
        torch.cuda.synchronize(); print(f"| {b:g} | {'int8' if g.int8 else 'bf16'} | {g.max_ctx} | {a:.2f} | {used():.2f} |", flush=True)
        GiB = 2**30; kv = (g.K.numel() + g.V.numel()) * g.K.element_size()
        print(f"    driver {used():.2f} = tensors alive {torch.cuda.memory_allocated() / GiB:.2f} (KV {kv / GiB:.2f}, weights+rest {(torch.cuda.memory_allocated() - kv) / GiB:.2f}) + allocator cache not in use {(torch.cuda.memory_reserved() - torch.cuda.memory_allocated()) / GiB:.2f}"
              f" + CUDA context/libraries {(used() * GiB - torch.cuda.memory_reserved()) / GiB:.2f};  peak tensors during the prompt {torch.cuda.max_memory_allocated() / GiB:.2f}", flush=True)
    except (MemoryError, torch.cuda.OutOfMemoryError) as e:
        print(f"| {b:g} | - | - | - | FAILED {str(e)[:60]} |", flush=True)
    g = None; gc.collect(); torch.cuda.empty_cache()
