"""Qwen2.5-0.5B-Instruct decoding with the dense matvecs on the XDNA1 NPU.

usage: python src/qwen_npu.py [--cpu] [--tokens N] "prompt"

Weights are int16 (+-511, per-row scale), activations int16 (+-511, per-vector scale),
int32 accumulate. Attention, norms, RoPE, SiLU and sampling (greedy) run in numpy.
--cpu runs the same model in float32 numpy, to compare outputs.
"""
import argparse, json, os, struct, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):  # keep the CPU light: the NPU does the work
    os.environ.setdefault(v, "1")
import numpy as np
from tokenizers import Tokenizer

D = "models/qwen"
H, NL, NH, NKV, HD, FF, EPS, THETA = 896, 24, 14, 2, 64, 4864, 1e-6, 1e6
QMAX, EOS = 511, 151645
I8 = os.environ.get("NPU_INT8", "1") == "1"   # int8 weights (default) or int16 weights
WMAX, XMAX = (127, 2047) if I8 else (511, 511)
LM_CHUNKS = [9728] * 15 + [6016]


def load_safetensors(path):
    """Minimal reader (numpy can't read bf16); yields (name, float32 array)."""
    f = open(path, "rb")
    n = struct.unpack("<Q", f.read(8))[0]
    hdr = json.loads(f.read(n))
    base = 8 + n
    hdr.pop("__metadata__", None)
    for name, m in hdr.items():
        a, b = m["data_offsets"]
        f.seek(base + a)
        raw = np.frombuffer(f.read(b - a), dtype=np.uint16)
        assert m["dtype"] == "BF16", m["dtype"]
        yield name, (raw.astype(np.uint32) << 16).view(np.float32).reshape(m["shape"])


def qmat(W):
    s = np.abs(W).max(1) / WMAX
    s[s == 0] = 1
    return np.round(W / s[:, None]).astype(np.int8 if I8 else np.int16), s.astype(np.float32)


class CpuLin:
    def __init__(self, W, b=None):
        self.W, self.b = W, b

    def __call__(self, x):
        y = self.W @ x
        return y if self.b is None else y + self.b


class Npu:
    """Owns the device and one kernel (context) per matrix shape."""
    def __init__(self):
        import pyxrt
        self.px, self.dev, self.k = pyxrt, pyxrt.device(0), {}
        self.t = 0.0

    def kernel(self, M, K):
        if (M, K) not in self.k:
            px = self.px
            name = f"build/mv8/mv8_{M}x{K}_c4" if I8 else f"build/mv/mv_{M}x{K}_c4"
            xb = px.xclbin(name + ".xclbin")
            self.dev.register_xclbin(xb)
            k = px.kernel(px.hw_context(self.dev, xb.get_uuid()), xb.get_kernels()[0].get_name())
            ins = np.fromfile(name + ".bin", dtype=np.uint32)
            bi = px.bo(self.dev, ins.nbytes, px.bo.cacheable, k.group_id(1))
            bi.write(ins.tobytes(), 0)
            bi.sync(px.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            bx = px.bo(self.dev, K * 2, px.bo.host_only, k.group_id(4))
            bc = px.bo(self.dev, M * 4, px.bo.host_only, k.group_id(5))
            self.k[(M, K)] = (k, bi, len(ins), bx, bc)
        return self.k[(M, K)]


class NpuLin:
    def __init__(self, npu, W, b=None):
        self.npu, self.b = npu, b
        self.parts = []
        px = npu.px
        chunks = LM_CHUNKS if W.shape[0] == 151936 else [W.shape[0]]
        r = 0
        for m in chunks:
            q, s = qmat(W[r:r + m])
            k, *_ = npu.kernel(m, W.shape[1])
            bo = px.bo(npu.dev, q.nbytes, px.bo.host_only, k.group_id(3))
            bo.write(q.tobytes(), 0)
            bo.sync(px.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            self.parts.append((m, W.shape[1], bo, s))
            r += m

    def __call__(self, x):
        npu, px = self.npu, self.npu.px
        sx = np.abs(x).max() / XMAX or 1.0
        xq = np.round(x / sx).astype(np.int16)
        out = []
        for m, K, bo, s in self.parts:
            k, bi, n, bx, bc = npu.kernel(m, K)
            t = time.perf_counter()
            bx.write(xq.tobytes(), 0)
            bx.sync(px.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            k(3, bi, n, bo, bx, bc).wait()
            bc.sync(px.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
            acc = np.frombuffer(bc.read(m * 4, 0), dtype=np.int32)
            npu.t += time.perf_counter() - t
            out.append(acc * s * sx)
        y = np.concatenate(out) if len(out) > 1 else out[0]
        return y if self.b is None else y + self.b


def rms(x, w):
    return x / np.sqrt((x * x).mean() + EPS) * w


def rope_tables(pos):
    inv = 1.0 / THETA ** (np.arange(0, HD, 2) / HD)
    a = pos * inv
    return np.concatenate([np.cos(a)] * 2).astype(np.float32), np.concatenate([np.sin(a)] * 2).astype(np.float32)


def rope(x, c, s):  # x: (heads, HD)
    h = HD // 2
    return x * c + np.concatenate([-x[:, h:], x[:, :h]], 1) * s


class Qwen:
    def __init__(self, cpu=False):
        self.npu = None if cpu else Npu()
        mk = (lambda W, b=None: CpuLin(W, b)) if cpu else (lambda W, b=None: NpuLin(self.npu, W, b))
        self.tok = Tokenizer.from_file(f"{D}/tokenizer.json")
        raw = dict(load_safetensors(f"{D}/model.safetensors"))
        self.embed = raw["model.embed_tokens.weight"]
        self.norm = raw["model.norm.weight"]
        self.layers = []
        for i in range(NL):
            p = f"model.layers.{i}."
            g = lambda n: raw.pop(p + n)
            qw, kw, vw = (raw.pop(p + f"self_attn.{n}_proj.weight") for n in "qkv")
            qb, kb, vb = (raw.pop(p + f"self_attn.{n}_proj.bias") for n in "qkv")
            self.layers.append(dict(
                ln1=g("input_layernorm.weight"), ln2=g("post_attention_layernorm.weight"),
                qkv=mk(np.concatenate([qw, kw, vw]), np.concatenate([qb, kb, vb])),
                o=mk(g("self_attn.o_proj.weight")),
                gu=mk(np.concatenate([g("mlp.gate_proj.weight"), g("mlp.up_proj.weight")])),
                down=mk(g("mlp.down_proj.weight"))))
        self.lm = mk(self.embed)
        self.kc = np.zeros((NL, 4096, NKV, HD), np.float32)
        self.vc = np.zeros_like(self.kc)

    def step(self, tok, pos):
        x = self.embed[tok].copy()
        c, s = rope_tables(pos)
        for i, L in enumerate(self.layers):
            qkv = L["qkv"](rms(x, L["ln1"]))
            q = rope(qkv[:H].reshape(NH, HD), c, s)
            self.kc[i, pos] = rope(qkv[H:H + NKV * HD].reshape(NKV, HD), c, s)
            self.vc[i, pos] = qkv[H + NKV * HD:].reshape(NKV, HD)
            K, V = self.kc[i, :pos + 1], self.vc[i, :pos + 1]       # (T, NKV, HD)
            rep = NH // NKV
            sc = np.einsum("hd,thd->ht", q, np.repeat(K, rep, 1)) / np.sqrt(HD)
            sc = np.exp(sc - sc.max(1, keepdims=True)); sc /= sc.sum(1, keepdims=True)
            att = np.einsum("ht,thd->hd", sc, np.repeat(V, rep, 1)).reshape(-1)
            x = x + L["o"](att)
            gu = L["gu"](rms(x, L["ln2"]))
            g_, u_ = gu[:FF], gu[FF:]
            x = x + L["down"]((g_ / (1 + np.exp(-g_))) * u_)
        return self.lm(rms(x, self.norm))

    def generate(self, prompt, n):
        text = ("<|im_start|>system\nYou are a helpful local assistant running on the user's own NPU. Always answer in Spanish unless the user writes in another language or asks for one.<|im_end|>\n"
                f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        ids = self.tok.encode(text, add_special_tokens=False).ids
        t0 = time.perf_counter(); out = []
        for pos, t in enumerate(ids):                     # prefill, one token at a time
            logits = self.step(t, pos)
        t_pre = time.perf_counter() - t0
        t1 = time.perf_counter()
        for j in range(n):
            nxt = int(np.argmax(logits))
            if nxt == EOS:
                break
            out.append(nxt)
            logits = self.step(nxt, len(ids) + j)
        dt = time.perf_counter() - t1
        print(f"[prefill {len(ids)} tok {t_pre:.1f}s | decode {len(out)} tok {dt:.1f}s = "
              f"{len(out)/dt:.2f} tok/s" + (f" | NPU calls {self.npu.t:.1f}s total]" if self.npu else "]"),
              file=sys.stderr)
        return self.tok.decode(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="What is the capital of France? Answer in one sentence.")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--tokens", type=int, default=32)
    a = ap.parse_args()
    t = time.perf_counter()
    m = Qwen(cpu=a.cpu)
    print(f"[loaded in {time.perf_counter()-t:.0f}s]", file=sys.stderr)
    print(m.generate(a.prompt, a.tokens))
