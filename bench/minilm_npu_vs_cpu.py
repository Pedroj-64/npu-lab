"""Compare MiniLM embeddings: CPU float32 vs NPU int-quantized linears."""
import sys, time
sys.path.insert(0, "src")
import numpy as np
from minilm import MiniLM
from npu_linear import NpuLinear

texts = ["The cat sits on the mat", "A kitten is resting on a rug", "Quarterly tax filing deadline",
         "How to install a graphics driver on Linux", "Recipe for chocolate cake",
         "Neural networks learn from data", "The stock market fell sharply today",
         "She walked her dog in the park"] * 8
cpu, npu = MiniLM(), MiniLM(linear=NpuLinear())
npu.encode(texts[:2])  # warmup / load kernels
for name, m in (("cpu", cpu), ("npu", npu)):
    t = time.perf_counter(); e = m.encode(texts); dt = time.perf_counter() - t
    print(f"{name}: {dt*1e3:.0f} ms for {len(texts)} texts"); globals()[name + "_e"] = e
cos = (cpu_e * npu_e).sum(1)
print(f"cosine cpu vs npu: min {cos.min():.4f} mean {cos.mean():.4f}")
