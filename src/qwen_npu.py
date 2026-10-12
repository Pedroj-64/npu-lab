"""Qwen2.5-Instruct decoding with every dense matvec on the XDNA1 NPU.

  python src/qwen_npu.py [--model models/qwen] [--tokens N] "prompt"
  python src/qwen_npu.py --model models/qwen15 --shapes     # kernels to build for that model

Weights int8 (per-row scale), activations int16 (per-vector scale), int32 accumulate.
Attention, norms, RoPE, SiLU and (greedy) sampling run in numpy. Weights are streamed layer by
layer so a 1.5B model never needs its float32 copy in RAM. --cpu runs a float32 numpy reference
(only sensible for the 0.5B model).
"""
import argparse, json, os, struct, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):  # keep the CPU light: the NPU does the work
    os.environ.setdefault(v, "1")
import numpy as np
from tokenizers import Tokenizer

EOS, WMAX, XMAX = 151645, 127, 2047
SYSTEM = ("You are a helpful local assistant running on the user's own NPU. "
          "Always answer in Spanish unless the user writes in another language or asks for one.")
LM_ROWS = 9728                    # lm_head is split in chunks of this many rows (+ remainder)
BUILD = "build/c2"                # 2-column int8 kernels, see scripts/build_matvec.py --shared


def f32(raw_bf16):
    return (raw_bf16.astype(np.uint32) << 16).view(np.float32)


class Weights:
    """Lazy bf16 safetensors reader: get(name) -> float32 array."""
    def __init__(self, path):
        self.f = open(path, "rb")
        n = struct.unpack("<Q", self.f.read(8))[0]
        self.h = json.loads(self.f.read(n))
        self.h.pop("__metadata__", None)
        self.base = 8 + n

    def raw(self, name):
        """The bf16 tensor exactly as stored, as uint16."""
        m = self.h[name]
        a, b = m["data_offsets"]
        assert m["dtype"] == "BF16", m["dtype"]
        self.f.seek(self.base + a)
        return np.frombuffer(self.f.read(b - a), dtype=np.uint16).reshape(m["shape"])

    def get(self, name):
        return f32(self.raw(name))


def qmat(W):
    s = np.abs(W).max(1) / WMAX
    s[s == 0] = 1
    return np.round(W / s[:, None]).astype(np.int8), s.astype(np.float32)


def lm_chunks(vocab):
    return [LM_ROWS] * (vocab // LM_ROWS) + ([vocab % LM_ROWS] if vocab % LM_ROWS else [])


def shapes(c):
    """(M, K) of every matvec this model needs."""
    H, FF = c["hidden_size"], c["intermediate_size"]
    kv = c["num_key_value_heads"] * (H // c["num_attention_heads"])
    return sorted({(H + 2 * kv, H), (H, H), (2 * FF, H), (H, FF)} | {(m, H) for m in lm_chunks(c["vocab_size"])})


class CpuLin:
    def __init__(self, W, b=None):
        self.W, self.b = W, b

    def __call__(self, x):
        y = self.W @ x
        return y if self.b is None else y + self.b


class Npu:
    """Owns the device. One 2-column context per K, kept resident: the xclbin does not depend on M
    (only the instruction stream does), so the NPU never reconfigures between layers."""
    def __init__(self):
        import pyxrt
        self.px, self.dev, self.k, self.ctx = pyxrt, pyxrt.device(0), {}, {}
        self.t = 0.0

    def kernel(self, M, K):
        if (M, K) not in self.k:
            px, name = self.px, f"{BUILD}/c2_{M}x{K}"
            if K not in self.ctx:
                xb = px.xclbin(name + ".xclbin")
                self.dev.register_xclbin(xb)
                self.ctx[K] = px.kernel(px.hw_context(self.dev, xb.get_uuid()), xb.get_kernels()[0].get_name())
            k = self.ctx[K]
            ins = np.fromfile(name + ".bin", dtype=np.uint32)
            bi = px.bo(self.dev, ins.nbytes, px.bo.cacheable, k.group_id(1))
            bi.write(ins.tobytes(), 0)
            bi.sync(px.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
            bx = px.bo(self.dev, K * 2, px.bo.host_only, k.group_id(4))
            bc = px.bo(self.dev, M * 4, px.bo.host_only, k.group_id(5))
            self.k[(M, K)] = (k, bi, len(ins), bx, bc)
        return self.k[(M, K)]


class NpuLin:
    def __init__(self, npu, W, b=None, chunks=None):
        self.npu, self.b, self.parts = npu, b, []
        px, r = npu.px, 0
        for m in chunks or [W.shape[0]]:
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


def rms(x, w, eps):
    return x / np.sqrt((x * x).mean() + eps) * w


def rope_tables(pos, hd, theta):
    a = pos * (1.0 / theta ** (np.arange(0, hd, 2) / hd))
    return np.concatenate([np.cos(a)] * 2).astype(np.float32), np.concatenate([np.sin(a)] * 2).astype(np.float32)


def rope(x, c, s):  # x: (heads, hd)
    h = x.shape[1] // 2
    return x * c + np.concatenate([-x[:, h:], x[:, :h]], 1) * s


class Qwen:
    def __init__(self, d="models/qwen", cpu=False, max_ctx=4096):
        c = json.load(open(f"{d}/config.json"))
        self.H, self.NL, self.NH, self.NKV = c["hidden_size"], c["num_hidden_layers"], c["num_attention_heads"], c["num_key_value_heads"]
        self.HD, self.FF, self.eps, self.theta = self.H // self.NH, c["intermediate_size"], c["rms_norm_eps"], c["rope_theta"]
        self.npu = None if cpu else Npu()
        mk = (lambda W, b=None, chunks=None: CpuLin(W, b)) if cpu else (lambda W, b=None, chunks=None: NpuLin(self.npu, W, b, chunks))
        self.tok = Tokenizer.from_file(f"{d}/tokenizer.json")
        w = Weights(f"{d}/model.safetensors")
        self.embed, self.norm = w.get("model.embed_tokens.weight"), w.get("model.norm.weight")
        self.pf = None
        if not cpu and os.environ.get("NPU_PREFILL", "1") == "1":
            try:
                import prefill_npu as P
                miss = [x for x in P.shapes(c) if not os.path.exists(P.xclbin_name(*x) + ".xclbin")]
                if miss:
                    print(f"[prefill kernels missing {miss}; run scripts/build_prefill.py {d}. Using slow token-by-token prefill]", file=sys.stderr)
                else:
                    self.pf = P.PrefillNpu(self.npu)
            except Exception as e:
                print(f"[batched prefill disabled: {type(e).__name__}: {e}]", file=sys.stderr)
        self.layers = []
        for i in range(self.NL):
            p = f"model.layers.{i}."
            r = lambda n: w.raw(p + n)
            qkv_w = np.concatenate([r(f"self_attn.{n}_proj.weight") for n in "qkv"])
            qkv_b = np.concatenate([w.get(p + f"self_attn.{n}_proj.bias") for n in "qkv"])
            o_w, g_w, u_w, d_w = r("self_attn.o_proj.weight"), r("mlp.gate_proj.weight"), r("mlp.up_proj.weight"), r("mlp.down_proj.weight")
            L = dict(ln1=w.get(p + "input_layernorm.weight"), ln2=w.get(p + "post_attention_layernorm.weight"),
                     qkv=mk(f32(qkv_w), qkv_b), o=mk(f32(o_w)), gu=mk(f32(np.concatenate([g_w, u_w]))), down=mk(f32(d_w)))
            if self.pf:
                try:
                    from prefill_npu import PrefillLin
                    L["p"] = dict(qkv=PrefillLin(self.pf, qkv_w, qkv_b), o=PrefillLin(self.pf, o_w),
                                  gate=PrefillLin(self.pf, g_w), up=PrefillLin(self.pf, u_w), down=PrefillLin(self.pf, d_w))
                except Exception as e:      # e.g. the NPU has no free columns for the extra contexts
                    print(f"[batched prefill disabled: {type(e).__name__}: {e}]", file=sys.stderr)
                    self.pf = None
                    for prev in self.layers:
                        prev.pop("p", None)
            self.layers.append(L)
        self.lm = mk(self.embed, chunks=None if cpu else lm_chunks(c["vocab_size"]))  # tied embeddings
        self.kc = np.zeros((self.NL, max_ctx, self.NKV, self.HD), np.float32)
        self.vc = np.zeros_like(self.kc)

    def step(self, tok, pos):
        H, NH, NKV, HD, FF = self.H, self.NH, self.NKV, self.HD, self.FF
        x = self.embed[tok].copy()
        c, s = rope_tables(pos, HD, self.theta)
        rep = NH // NKV
        for i, L in enumerate(self.layers):
            qkv = L["qkv"](rms(x, L["ln1"], self.eps))
            q = rope(qkv[:H].reshape(NH, HD), c, s)
            self.kc[i, pos] = rope(qkv[H:H + NKV * HD].reshape(NKV, HD), c, s)
            self.vc[i, pos] = qkv[H + NKV * HD:].reshape(NKV, HD)
            K, V = self.kc[i, :pos + 1], self.vc[i, :pos + 1]       # (T, NKV, HD)
            sc = np.einsum("hd,thd->ht", q, np.repeat(K, rep, 1)) / np.sqrt(HD)
            sc = np.exp(sc - sc.max(1, keepdims=True)); sc /= sc.sum(1, keepdims=True)
            x = x + L["o"](np.einsum("ht,thd->hd", sc, np.repeat(V, rep, 1)).reshape(-1))
            gu = L["gu"](rms(x, L["ln2"], self.eps))
            g_, u_ = gu[:FF], gu[FF:]
            x = x + L["down"]((g_ / (1 + np.exp(-np.clip(g_, -60, 60)))) * u_)
        return self.lm(rms(x, self.norm, self.eps))

    def prefill(self, ids):
        """Process the whole prompt at once (matmuls on the NPU, attention in numpy); fills the KV cache,
        returns the logits of the last token."""
        H, NH, NKV, HD, eps = self.H, self.NH, self.NKV, self.HD, self.eps
        T, rep = len(ids), self.NH // self.NKV
        X = self.embed[ids].astype(np.float32)
        ang = np.arange(T)[:, None] * (1.0 / self.theta ** (np.arange(0, HD, 2) / HD))[None, :]
        cos = np.concatenate([np.cos(ang)] * 2, 1).astype(np.float32)[:, None, :]
        sin = np.concatenate([np.sin(ang)] * 2, 1).astype(np.float32)[:, None, :]
        rot = lambda x: np.concatenate([-x[..., HD // 2:], x[..., :HD // 2]], -1)
        mask = np.triu(np.full((T, T), -1e30, np.float32), 1)
        rmsb = lambda x, w: x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w
        for i, L in enumerate(self.layers):
            P = L["p"]
            qkv = P["qkv"](rmsb(X, L["ln1"]))
            q = qkv[:, :H].reshape(T, NH, HD)
            k = qkv[:, H:H + NKV * HD].reshape(T, NKV, HD)
            v = qkv[:, H + NKV * HD:].reshape(T, NKV, HD)
            q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
            self.kc[i, :T], self.vc[i, :T] = k, v
            kh, vh = np.repeat(k, rep, 1).transpose(1, 0, 2), np.repeat(v, rep, 1).transpose(1, 0, 2)
            sc = q.transpose(1, 0, 2) @ kh.transpose(0, 2, 1) / np.sqrt(HD) + mask      # (NH, T, T), causal
            sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True)
            X = X + P["o"]((sc @ vh).transpose(1, 0, 2).reshape(T, H))
            h2 = rmsb(X, L["ln2"])
            g, u = P["gate"](h2), P["up"](h2)
            X = X + P["down"]((g / (1 + np.exp(-np.clip(g, -60, 60)))) * u)
        return self.lm(rms(X[-1], self.norm, eps))

    def stream(self, prompt, n):
        """Yield the answer piece by piece (UTF-8 safe). Timing ends up in self.stats."""
        text = (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        ids = self.tok.encode(text, add_special_tokens=False).ids
        if len(ids) + n >= self.kc.shape[1]:
            raise ValueError(f"prompt ({len(ids)} tokens) + answer exceeds the {self.kc.shape[1]}-token context")
        t0 = time.perf_counter(); out = []
        if getattr(self, "pf", None):                     # whole prompt at once
            logits = self.prefill(ids)
        else:                                             # fallback: one token at a time
            for pos, t in enumerate(ids):
                logits = self.step(t, pos)
        t_pre = time.perf_counter() - t0
        t1, sent = time.perf_counter(), ""
        self.stats = {"prefill_tokens": len(ids), "prefill_s": t_pre, "decode_tokens": 0, "decode_s": 0.0}
        for j in range(n):
            nxt = int(np.argmax(logits))
            if nxt == EOS:
                break
            out.append(nxt)
            self.stats.update(decode_tokens=len(out), decode_s=time.perf_counter() - t1)
            full = self.tok.decode(out)
            if not full.endswith("\ufffd"):               # wait for the rest of a multi-byte character
                yield full[len(sent):]
                sent = full
            logits = self.step(nxt, len(ids) + j)
        self.stats["decode_s"] = time.perf_counter() - t1
        full = self.tok.decode(out)
        if len(full) > len(sent):
            yield full[len(sent):]

    def generate(self, prompt, n):
        text = "".join(self.stream(prompt, n))
        st = self.stats
        print(f"[prefill {st['prefill_tokens']} tok {st['prefill_s']:.1f}s | decode {st['decode_tokens']} tok "
              f"{st['decode_s']:.1f}s = {st['decode_tokens']/max(st['decode_s'], 1e-9):.2f} tok/s"
              + (f" | NPU calls {self.npu.t:.1f}s total]" if self.npu else "]"), file=sys.stderr)
        return text


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="What is the capital of France? Answer in one sentence.")
    ap.add_argument("--model", default="models/qwen")
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--shapes", action="store_true", help="print the MxK kernels to build and exit")
    ap.add_argument("--tokens", type=int, default=32)
    a = ap.parse_args()
    if a.shapes:
        print(" ".join(f"{m}x{k}" for m, k in shapes(json.load(open(f"{a.model}/config.json")))))
        sys.exit()
    t = time.perf_counter()
    m = Qwen(a.model, cpu=a.cpu)
    print(f"[loaded in {time.perf_counter()-t:.0f}s]", file=sys.stderr)
    print(m.generate(a.prompt, a.tokens))
