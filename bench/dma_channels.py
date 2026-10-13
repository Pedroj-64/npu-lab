"""Does a second DMA channel per column raise the ceiling? memcpy (shim -> memtile -> shim), 32 MB, 1 or 2 channels per column, 1/2/4 columns, 3 independent measurements of 30 calls.
Build first: see bench/DMA_READ.md (memcpy.py -co <cols> -ch <1|2> -l 8388608 -b True --xclbin-path build/dma/mcch_32M_c<cols>_ch<ch>.xclbin ...).  usage: .venv/bin/python bench/dma_channels.py"""
import os, sys, time
import numpy as np, pyxrt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
dev = pyxrt.device(0)
print("| cols | channels/col | MB | GB/s per direction (mean ± sd of 3 medians) |\n|---:|---:|---:|---:|")
for c in (1, 2, 4):
    for ch in (1, 2):
        name = os.path.join(ROOT, f"build/dma/mcch_32M_c{c}_ch{ch}")
        xb = pyxrt.xclbin(name + ".xclbin"); dev.register_xclbin(xb)
        k = pyxrt.kernel(pyxrt.hw_context(dev, xb.get_uuid()), xb.get_kernels()[0].get_name())
        ins = np.fromfile(name + ".bin", dtype=np.uint32); n = 32 * 262144
        bi = pyxrt.bo(dev, ins.nbytes, pyxrt.bo.cacheable, k.group_id(1)); bi.write(ins.tobytes(), 0); bi.sync(TO)
        ba = pyxrt.bo(dev, n * 4, pyxrt.bo.host_only, k.group_id(3)); bb = pyxrt.bo(dev, n * 4, pyxrt.bo.host_only, k.group_id(4))
        ba.write(np.arange(1, n + 1, dtype=np.int32).tobytes(), 0); ba.sync(TO)
        meds = []
        for rep in range(3):
            ts = []
            for i in range(35):
                t = time.perf_counter(); k(3, bi, len(ins), ba, bb).wait(); ts.append(time.perf_counter() - t)
            meds.append(n * 4 / np.median(ts[5:]) / 1e9)
        print(f"| {c} | {ch} | 34 | {np.mean(meds):.2f} ± {np.std(meds, ddof=1):.2f} |", flush=True)
