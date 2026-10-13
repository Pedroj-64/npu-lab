"""Qwen3.5 (hybrid Gated DeltaNet + gated attention) decoding: every dense matvec on the XDNA1 NPU, everything else on the GPU.

  python src/qwen35_npu.py [--model models/qwen35_2b] [--tokens N] "prompt"

Per layer and token the NPU runs the projections (int8 weights, shared 4-column kernel); the GPU (src/gpu_qwen35.py) runs the DeltaNet
conv/recurrence or the attention with its KV cache and returns a vector for the next NPU matvec. Needs the GPU (Assisted profile):
the DeltaNet recurrence has no CPU path here. Subclass of the Qwen3 engine: streaming, prefix reuse and the router classifier are shared.
"""
import argparse, json, math, os, sys, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
import numpy as np
import torch
from tokenizers import Tokenizer
import qwen_npu as Q
import statecache

PREFIX = "model.language_model."
CALL_S, STREAM_BPS = 120e-6, 21e9      # fixed cost of one NPU call and the weight-streaming rate measured on this machine (see ROADMAP)


def text_config(c):
    return c.get("text_config", c)


def dims(c):
    """All the matrix shapes of the model. c is the text config."""
    H, FF, HD = c["hidden_size"], c["intermediate_size"], c["head_dim"]
    NH, NKV = c["num_attention_heads"], c["num_key_value_heads"]
    kd = c["linear_num_key_heads"] * c["linear_key_head_dim"]
    vd = c["linear_num_value_heads"] * c["linear_value_head_dim"]
    inp = 2 * kd + vd + vd + 2 * c["linear_num_value_heads"]            # qkv | z | b | a
    pad = lambda m: -(-m // 128) * 128                                    # the 4-column kernel needs M % 128 == 0
    return dict(H=H, FF=FF, HD=HD, NH=NH, NKV=NKV, kd=kd, vd=vd, inp=inp, inp_pad=pad(inp), qkvg=NH * HD * 2 + 2 * NKV * HD, qd=NH * HD)


def k0(c):
    """The K of the single decode kernel. Every matrix is cut into pieces of K = k0 (zero-padded at the end), one NPU call each; a small k0
    means many calls (fixed cost each), a large one means padding (wasted bytes). Picks the k0 that minimises the estimated time per token."""
    d = dims(c)
    types = c["layer_types"]
    nl, nf = types.count("linear_attention"), types.count("full_attention")
    chunks = len(Q.lm_chunks(c["vocab_size"]))
    mats = [(d["inp_pad"], d["H"], nl), (d["H"], d["vd"], nl), (d["qkvg"], d["H"], nf), (d["H"], d["qd"], nf),
            (2 * d["FF"], d["H"], nl + nf), (d["H"], d["FF"], nl + nf)]

    def cost(K0):
        t = 0.0
        for M, K, n in mats:
            pieces = -(-K // K0)
            t += n * (pieces * CALL_S + M * pieces * K0 / STREAM_BPS)
        pieces = -(-d["H"] // K0)
        return t + chunks * pieces * CALL_S + c["vocab_size"] * pieces * K0 / STREAM_BPS
    return min(range(512, 4097, 128), key=cost)


def split_rows(M, share):
    """(rows for the NPU, rows for the GPU) of a matrix with M rows: the NPU part is a multiple of 128 rows (kernel tiling)."""
    mn = max(128, int(M * (1 - share)) // 128 * 128)
    return (mn, M - mn) if M - mn > 0 else (M, 0)


def shapes(c, share=0.0):
    """(M, K) of every matvec instruction stream this model needs (all with the same K). With `share` > 0 the GPU takes that fraction
    of the rows of every layer matrix, so the NPU kernels have fewer rows."""
    d, K = dims(c), k0(c)
    if share > 0:
        ms = {split_rows(m, share)[0] for m in (d["inp"], d["qkvg"], d["H"], 2 * d["FF"])}
    else:
        ms = {d["inp_pad"], d["qkvg"], d["H"], 2 * d["FF"]}
    return sorted((m, K) for m in ms | set(Q.lm_chunks(split_rows(c["vocab_size"], share)[0] if share > 0 else c["vocab_size"])))


class Bf16Rows:
    """Rows of a bf16 matrix (stored as uint16) converted to float32 only when sliced, so a 248k-row embedding never exists as float32."""
    def __init__(self, raw):
        self.raw, self.shape = raw, raw.shape

    def __getitem__(self, s):
        return Q.f32(self.raw[s])


def zrms_np(x, w, eps):
    return x / np.sqrt((x * x).mean() + eps) * (1.0 + w)


class Qwen35(Q.Qwen):
    def __init__(self, d="models/qwen35_2b", max_ctx=8192, vram_gb=4.0, gpu=True, gpu_share=0.0, cache_gb=0.0, fast_head=None):
        import gpu_qwen35
        if not (gpu and gpu_qwen35.available()):
            raise RuntimeError("Qwen3.5 needs the GPU: switch to the Assisted profile (or use a Qwen3 model for NPU only)")
        cfg = json.load(open(f"{d}/config.json"))
        c = text_config(cfg)
        self.c, self.d, self.share = c, dims(c), float(gpu_share)
        miss = [f"c4_{m}x{k}" for m, k in shapes(c, self.share) if not os.path.exists(f"{Q.BUILD}/c4_{m}x{k}.xclbin")]
        if miss:
            raise RuntimeError(f"{len(miss)} NPU kernels are missing for this model at a {self.share:.0%} GPU share; build them with: "
                               f"python scripts/build_matvec.py --shared4 $(python src/qwen35_npu.py --model {d} --shapes --share {self.share})")
        self.H, self.NL, self.eps = c["hidden_size"], c["num_hidden_layers"], c["rms_norm_eps"]
        self.qwen3, self.vsub, self.vpos, self.pf = True, None, None, None
        self.tok = Tokenizer.from_file(f"{d}/tokenizer.json")
        w = Q.Weights(f"{d}/model.safetensors" if os.path.exists(f"{d}/model.safetensors") else
                      os.path.join(d, sorted(f for f in os.listdir(d) if f.endswith(".safetensors"))[0]))
        t0 = time.perf_counter()
        lap = lambda what: print(f"[load {what}: {time.perf_counter() - t0:.1f}s]", file=sys.stderr) if os.environ.get("NPU_LOAD_TIMING") else None
        Q.QUANT_GPU = True                                  # quantize the weights on the GPU while loading
        self.npu = Q.Npu(k0(c))
        mk = lambda W, b=None, chunks=None, tail=None: Q.NpuLin(self.npu, W, b, chunks, tail)
        self.g = gpu_qwen35.GpuCore35(c, w, PREFIX, max_ctx, vram_gb)
        self.max_ctx, self.ctx_limited = self.g.max_ctx, self.g.limited
        lap("GPU weights")
        self.embed_raw = w.memmap(PREFIX + "embed_tokens.weight")                    # (vocab, H) bf16 bits, memory-mapped: no RAM of its own
        self.norm = Q.f32(w.raw(PREFIX + "norm.weight"))
        self.layers = []
        d_ = self.d
        for i in range(self.NL):
            p = f"{PREFIX}layers.{i}."
            L = dict(type=c["layer_types"][i], ln1=Q.f32(w.raw(p + "input_layernorm.weight")), ln2=Q.f32(w.raw(p + "post_attention_layernorm.weight")))
            def mkm(raw, key, pad_to=None):
                """The NPU matvec of one layer matrix; with a GPU share the last rows are left to the GPU (computed from the resident weights)."""
                mn, mg = split_rows(raw.shape[0], self.share) if self.share > 0 else (raw.shape[0], 0)
                if mg == 0:
                    if pad_to:
                        raw = np.concatenate([raw, np.zeros((pad_to - raw.shape[0], raw.shape[1]), np.uint16)])   # bf16 zero = 0 bits
                    return mk(raw)
                return mk(raw[:mn], tail=gpu_qwen35.GpuTail(self.g, self.g.layers[i][key], mn, raw.shape[0]))
            g_, u_, dn = (w.raw(p + f"mlp.{n}_proj.weight") for n in ("gate", "up", "down"))       # bf16 bits; quantised on the GPU
            L["gu"], L["down"] = mkm(np.concatenate([g_, u_]), "gu"), mkm(dn, "down")
            if L["type"] == "linear_attention":
                a = p + "linear_attn."
                W = np.concatenate([w.raw(a + n + ".weight") for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")])
                L["inp"], L["out"] = mkm(W, "inp", pad_to=d_["inp_pad"]), mkm(w.raw(a + "out_proj.weight"), "out")
            else:
                a = p + "self_attn."
                L["qkv"] = mkm(np.concatenate([w.raw(a + n + "_proj.weight") for n in "qkv"]), "qkv")
                L["o"] = mkm(w.raw(a + "o_proj.weight"), "o")
            self.layers.append(L)
        lap("layers on the NPU")
        V = c["vocab_size"]
        mn, mg = split_rows(V, self.share) if self.share > 0 else (V, 0)
        if fast_head is None:
            fast_head = os.environ.get("NPU_FAST_VOCAB") == "1"          # off by default: the output head over a subset of the vocabulary (see bench/FASE2.md, batch B)
        if fast_head and self.share > 0:
            print("[NPU_FAST_VOCAB ignored: it does not combine with a GPU share]", file=sys.stderr)
            fast_head = False
        if fast_head:
            # the cache is per tokenizer: Qwen3 and Qwen3.5 share only 281 token ids, and fast_vocab's own check is just the length
            self.vsub = Q.fast_vocab(self.tok, V, cache=os.path.expanduser(f"~/.local/share/npu-lab/vocab_fast_qwen35_{V}.npy"))
            self.vpos = np.full(V, -1, np.int64); self.vpos[self.vsub] = np.arange(len(self.vsub))
            self.lm = mk(np.ascontiguousarray(self.embed_raw[self.vsub]), chunks=[Q.LM_ROWS] * (len(self.vsub) // Q.LM_ROWS))
        else:
            tail = gpu_qwen35.GpuTail(self.g, self.g.lm_tail(self.embed_raw, mn, V), 0, mg) if mg else None
            self.lm = mk(self.embed_raw[:mn], chunks=Q.lm_chunks(mn), tail=tail)           # tied embeddings; with a GPU share the GPU keeps the last rows
        lap("lm_head")
        gen = os.path.join(d, "generation_config.json")
        self.eos = set(np.atleast_1d(json.load(open(gen)).get("eos_token_id", []))) if os.path.exists(gen) else set()
        self.eos |= {self.tok.token_to_id("<|im_end|>"), self.tok.token_to_id("<|endoftext|>")}
        self.cache_ids = []
        self.cache = statecache.StateCache(os.environ.get("NPU_LAB_STATE") or os.path.expanduser("~/.cache/npu-lab/state"), os.path.basename(os.path.normpath(d)) + "_" + str(os.path.getsize(f"{d}/config.json")),
                                           cache_gb * 1e9) if cache_gb > 0 else None
        self.tail_len = len(self.tok.encode("<think>\n\n</think>\n\n", add_special_tokens=False).ids)
        self.stats = {}

    def _embed(self, ids):
        rows = np.ascontiguousarray(self.embed_raw[np.asarray(ids, dtype=np.int64)]).view(np.int16)   # gather the rows from the mapped file
        return torch.from_numpy(rows).view(torch.bfloat16).to("cuda")

    def step(self, tok, pos):
        d, eps = self.d, self.eps
        x = Q.f32(self.embed_raw[tok]).copy()
        for i, L in enumerate(self.layers):
            h = zrms_np(x, L["ln1"], eps)
            if L["type"] == "linear_attention":
                core = self.g.linear_step(i, L["inp"](h)[:d["inp"]])
                x = x + L["out"](core)
            else:
                qkvg = L["qkv"](h)
                n, nk = d["NH"] * d["HD"] * 2, d["NKV"] * d["HD"]
                x = x + L["o"](self.g.attn_step(i, qkvg[:n], qkvg[n:n + nk], qkvg[n + nk:], pos))
            gu = L["gu"](zrms_np(x, L["ln2"], eps))
            g_, u_ = gu[:d["FF"]], gu[d["FF"]:]
            x = x + L["down"]((g_ / (1 + np.exp(-np.clip(g_, -60, 60)))) * u_)
        return self.lm(zrms_np(x, self.norm, eps))

    def prefill(self, ids, start=0, marks=()):
        if start == 0:
            self.g.reset_states()
        hid = self.g.prefill(self._embed(ids), start, marks)
        return self.lm(zrms_np(hid, self.norm, self.eps))

    def prepare_prefix(self, L):
        return self.g.rewind(L)                          # DeltaNet state cannot be cut back: resume at the newest snapshot <= L

    def restore_prefix(self, ids, L):
        if self.cache is None:
            return L
        n = self.cache.restore(ids, self.g, at_least=L)  # a document read earlier (even in another session)
        return n or L

    def after_prefill(self, ids):
        if self.cache is None:
            return
        pos = max((p for p in self.g.snaps if p <= len(ids) - self.tail_len), default=0)      # newest snapshot before the question/answer tail
        if pos >= statecache.MIN_TOKENS:
            self.cache.save(ids, pos, self.g)

    def classify(self, prompt, system, labels):
        self.g.reset_states()
        return super().classify(prompt, system, labels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt", nargs="?", default="¿Cuál es la capital de Colombia? Responde en una frase.")
    ap.add_argument("--model", default="models/qwen35_2b")
    ap.add_argument("--tokens", type=int, default=60)
    ap.add_argument("--shapes", action="store_true", help="print the MxK kernels to build for that model and exit")
    ap.add_argument("--share", type=float, default=0.0, help="fraction of every layer matrix computed on the GPU (0 = NPU only)")
    a = ap.parse_args()
    if a.shapes:
        cfg = json.load(open(f"{a.model}/config.json"))
        print(" ".join(f"{m}x{k}" for m, k in (shapes(text_config(cfg), a.share) if cfg.get("model_type") == "qwen3_5" else Q.shapes(cfg))))
        return
    t = time.time()
    m = Qwen35(a.model, gpu_share=a.share)
    print(f"[loaded in {time.time() - t:.0f}s, context {m.max_ctx}]", file=sys.stderr)
    print(m.generate(a.prompt, a.tokens))



def load_engine(d, **kw):
    """The right engine for a model folder: Qwen3.5 (hybrid, needs the GPU) or Qwen3 (dense)."""
    if json.load(open(f"{d}/config.json")).get("model_type") == "qwen3_5":
        return Qwen35(d, **{k: v for k, v in kw.items() if k in ("max_ctx", "vram_gb", "gpu", "gpu_share", "cache_gb", "fast_head")})
    return Q.Qwen(d, **{k: v for k, v in kw.items() if k in ("max_ctx", "vram_gb", "gpu", "fast_head")})


if __name__ == "__main__":
    main()
