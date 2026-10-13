"""Batched prompt processing ("prefill") on the NPU: bf16 x bf16 -> f32 matmuls, 128 tokens per call, 4 columns.

Weights are used exactly as stored (Qwen ships bf16), so this path is as accurate as the model itself;
the int8 matvec path in qwen_npu.py stays for one-token-at-a-time decoding.
Kernels: src/kernels/matmul_whole_array.py (4 columns), built by scripts/build_prefill.py into build/mm/.
"""
import numpy as np
from ml_dtypes import bfloat16

CH = 128            # tokens per NPU call (rows of the matmul)
COLS = 4            # 4 columns like the decode context (a 4-column context cannot coexist with 2-column ones, but two 4-column ones can)
BUILD = "build/mm"
MAX_TILES = 40      # the DMA repeat count limits N per call to ~40 tiles of 32*COLS columns


MAX_CONTEXTS = 6    # the driver refuses to create more hardware contexts than this (decode ones included)


def split_n(N, chunk=None):
    """Split N output columns into equal chunks of at most MAX_TILES tiles (a hardware DMA limit).
    With `chunk` (see common_chunk) every matrix of the model uses that chunk width, so fewer distinct kernels are needed."""
    tile = 32 * COLS
    assert N % tile == 0, f"N={N} must be a multiple of {tile}"
    if chunk and N % chunk == 0:
        return [chunk] * (N // chunk)
    t = N // tile
    d = next(d for d in range(1, t + 1) if t % d == 0 and t // d <= MAX_TILES)
    return [N // d] * d


def _widths(c):
    H, FF = c["hidden_size"], c["intermediate_size"]
    QD = c["num_attention_heads"] * c.get("head_dim", H // c["num_attention_heads"])
    kv = c["num_key_value_heads"] * c.get("head_dim", H // c["num_attention_heads"])
    return H, FF, QD, kv


def common_chunk(c):
    """Widest chunk that divides every output width of the model (None if it would exceed the DMA limit)."""
    import math
    H, FF, QD, kv = _widths(c)
    g = math.gcd(math.gcd(QD + 2 * kv, H), FF)
    return g if g % (32 * COLS) == 0 and g // (32 * COLS) <= MAX_TILES else None


def shapes(c):
    """(K, N) of every prefill matmul kernel a model needs."""
    H, FF = c["hidden_size"], c["intermediate_size"]
    kv = c["num_key_value_heads"] * c.get("head_dim", H // c["num_attention_heads"])
    QD = c["num_attention_heads"] * c.get("head_dim", H // c["num_attention_heads"])
    return sorted({(K, n) for K, N in [(H, QD + 2 * kv), (QD, H), (H, FF), (FF, H)] for n in split_n(N, common_chunk(c))})


def xclbin_name(K, N):
    return f"{BUILD}/bf{COLS}_{CH}x{K}x{N}"


class PrefillNpu:
    """One 2-column context per (K, N) chunk shape, shared by every layer."""
    def __init__(self, npu, chunk=None):
        self.npu, self.k, self.chunk = npu, {}, chunk

    def kernel(self, K, N):
        if (K, N) not in self.k:
            px, dev, name = self.npu.px, self.npu.dev, xclbin_name(K, N)
            xb = px.xclbin(name + ".xclbin")
            dev.register_xclbin(xb)
            k = px.kernel(px.hw_context(dev, xb.get_uuid()), xb.get_kernels()[0].get_name())
            ins = np.fromfile(name + ".bin", dtype=np.uint32)
            bi = px.bo(dev, ins.nbytes, px.bo.cacheable, k.group_id(1))
            bi.write(ins.tobytes(), 0)
            bi.sync(px.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            ba = px.bo(dev, CH * K * 2, px.bo.host_only, k.group_id(3))
            bc = px.bo(dev, CH * N * 4, px.bo.host_only, k.group_id(5))
            self.k[(K, N)] = (k, bi, len(ins), ba, bc)
        return self.k[(K, N)]


class PrefillLin:
    """y = x @ W.T (+ bias) for a whole (T, K) block of tokens. `W_raw` is the bf16 weight as uint16, shape (N, K)."""
    def __init__(self, pf, W_raw, bias=None):
        self.pf, self.bias, self.parts = pf, bias, []
        px, dev = pf.npu.px, pf.npu.dev
        N, K = W_raw.shape
        Bt = np.ascontiguousarray(W_raw.T)                       # (K, N), what the kernel multiplies by
        c0 = 0
        for n in split_n(N, pf.chunk):
            k, *_ = pf.kernel(K, n)
            blk = np.ascontiguousarray(Bt[:, c0:c0 + n])
            bo = px.bo(dev, blk.nbytes, px.bo.host_only, k.group_id(4))
            bo.write(blk.tobytes(), 0)
            bo.sync(px.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self.parts.append((K, n, bo))
            c0 += n
        self.N, self.K = N, K

    def __call__(self, X):
        px, T = self.pf.npu.px, X.shape[0]
        Xb = X.astype(bfloat16)
        out = np.empty((T, self.N), np.float32)
        for r in range(0, T, CH):
            rows = min(CH, T - r)
            blk = np.zeros((CH, self.K), bfloat16)
            blk[:rows] = Xb[r:r + rows]
            c0 = 0
            for K, n, bo in self.parts:
                k, bi, ni, ba, bc = self.pf.kernel(K, n)
                ba.write(blk.tobytes(), 0)
                ba.sync(px.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
                k(3, bi, ni, ba, bo, bc).wait()
                bc.sync(px.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
                out[r:r + rows, c0:c0 + n] = np.frombuffer(bc.read(CH * n * 4, 0), dtype=np.float32).reshape(CH, n)[:rows]
                c0 += n
        return out if self.bias is None else out + self.bias
