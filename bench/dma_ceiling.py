"""DMA ceiling: host -> NPU -> host copy with the mlir-aie `memcpy` example (shim -> memtile -> shim, no compute tile).
Build first (CPU only, ~1 s each):
  for each cols in 1 2 4 and size MB in 8 32 128:
    python third_party/mlir-aie/programming_examples/basic/memcpy/memcpy.py -d npu -l <MB*262144> -co <cols> -ch 1 -b True \
      --xclbin-path=build/dma/mc_<MB>M_c<cols>.xclbin --insts-path=build/dma/mc_<MB>M_c<cols>.bin
usage: .venv/bin/python bench/dma_ceiling.py [iters]
GB/s = 10^9 bytes/s. "per direction" = buffer size / time (the NPU reads that much AND writes that much in each run);
"DDR traffic" = twice that. Decode only reads weights, so per-direction is the closest figure to its read bandwidth.
"""
import os, sys, time
import numpy as np
import pyxrt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
iters = int(sys.argv[1]) if len(sys.argv) > 1 else 10
TO, FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE
dev = pyxrt.device(0)
print("| cols | MB | ms (min) | ms (median) | GB/s per direction (median) | GB/s per direction (best) | DDR traffic GB/s (median) | check |\n|---:|---:|---:|---:|---:|---:|---:|---|")
for cols in (1, 2, 4):
    for mb in (8, 32, 128):
        name = os.path.join(ROOT, f"build/dma/mc_{mb}M_c{cols}")
        n = mb * 262144                                     # int32 elements
        try:
            xb = pyxrt.xclbin(name + ".xclbin")
            dev.register_xclbin(xb)
            k = pyxrt.kernel(pyxrt.hw_context(dev, xb.get_uuid()), xb.get_kernels()[0].get_name())
            ins = np.fromfile(name + ".bin", dtype=np.uint32)
            bi = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, k.group_id(1))
            bi.write(ins.tobytes(), 0); bi.sync(TO)
            ba = pyxrt.bo(dev, n * 4, pyxrt.bo.host_only, k.group_id(3))
            bb = pyxrt.bo(dev, n * 4, pyxrt.bo.host_only, k.group_id(4))
            a = np.arange(1, n + 1, dtype=np.int32)
            ba.write(a.tobytes(), 0); ba.sync(TO)
            ts = []
            for i in range(iters + 2):
                t = time.perf_counter(); k(3, bi, len(ins), ba, bb).wait(); ts.append(time.perf_counter() - t)
            bb.sync(FROM)
            ok = bool(np.array_equal(np.frombuffer(bb.read(n * 4, 0), dtype=np.int32), a))
            ts = sorted(ts[2:])                             # 2 warm-up runs dropped
            med, best, by = ts[len(ts) // 2], ts[0], n * 4
            print(f"| {cols} | {mb} | {best * 1e3:.2f} | {med * 1e3:.2f} | {by / med / 1e9:.2f} | {by / best / 1e9:.2f} | {2 * by / med / 1e9:.2f} | {'ok' if ok else 'MISMATCH'} |", flush=True)
            del ba, bb, bi, k
        except Exception as e:
            print(f"| {cols} | {mb} | - | - | - | - | - | {type(e).__name__}: {str(e)[:80]} |", flush=True)
