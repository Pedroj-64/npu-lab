"""`linear(x, w, b)` backend that runs the matmul on the XDNA1 NPU (pyxrt).

Kernels: mlir-aie whole_array, i16 in / i32 out, 512 rows per call.
Values are quantized to +-QMAX (10 bit) so the int32 accumulator can't overflow
for K <= 1536:  511*511*1536 ~ 4.0e8 < 2^31.
"""
import numpy as np
import pyxrt

BUILD = ("third_party/mlir-aie/programming_examples/basic/matrix_multiplication/"
         "whole_array/build")
ROWS, QMAX = 512, 511
SYNC_TO = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE
SYNC_FROM = pyxrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE


def _q(a):
    s = np.abs(a).max() / QMAX or 1.0
    return np.round(a / s).astype(np.int16), np.float32(s)


class _Kernel:
    def __init__(self, dev, k, n):
        name = f"{ROWS}x{k}x{n}_32x32x32_4c"
        xb = pyxrt.xclbin(f"{BUILD}/final_{name}.xclbin")
        dev.register_xclbin(xb)
        self.k = pyxrt.kernel(pyxrt.hw_context(dev, xb.get_uuid()), xb.get_kernels()[0].get_name())
        self.insts = np.fromfile(f"{BUILD}/insts_{name}.bin", dtype=np.uint32)
        self.bi = pyxrt.bo(dev, self.insts.nbytes, pyxrt.bo.cacheable, self.k.group_id(1))
        self.bi.write(self.insts.tobytes(), 0)
        self.bi.sync(SYNC_TO)
        self.ba = pyxrt.bo(dev, ROWS * k * 2, pyxrt.bo.host_only, self.k.group_id(3))
        self.bc = pyxrt.bo(dev, ROWS * n * 4, pyxrt.bo.host_only, self.k.group_id(5))
        self.dev, self.kk, self.n = dev, k, n
        self.weights = {}  # id(w) -> (bo_b, scale)

    def weight(self, w):
        if id(w) not in self.weights:
            q, s = _q(np.ascontiguousarray(w.T))
            bb = pyxrt.bo(self.dev, q.nbytes, pyxrt.bo.host_only, self.k.group_id(4))
            bb.write(q.tobytes(), 0)
            bb.sync(SYNC_TO)
            self.weights[id(w)] = (bb, s, q)  # keep q alive
        return self.weights[id(w)][:2]

    def run(self, x, w):
        bb, sw = self.weight(w)
        xq, sx = _q(x)
        self.ba.write(xq.tobytes(), 0)
        self.ba.sync(SYNC_TO)
        self.k(3, self.bi, len(self.insts), self.ba, bb, self.bc).wait()
        self.bc.sync(SYNC_FROM)
        c = np.frombuffer(self.bc.read(ROWS * self.n * 4, 0), dtype=np.int32).reshape(ROWS, self.n)
        return c.astype(np.float32) * (sx * sw)


class NpuLinear:
    def __init__(self):
        self.dev = pyxrt.device(0)
        self.kernels = {}

    def __call__(self, x, w, b):
        n, k = w.shape
        if (k, n) not in self.kernels:
            self.kernels[(k, n)] = _Kernel(self.dev, k, n)
        ker = self.kernels[(k, n)]
        out = np.empty((x.shape[0], n), np.float32)
        for i in range(0, x.shape[0], ROWS):
            chunk = x[i:i + ROWS]
            pad = np.zeros((ROWS, k), np.float32)
            pad[:len(chunk)] = chunk
            out[i:i + len(chunk)] = ker.run(pad, w)[:len(chunk)]
        return out + b
