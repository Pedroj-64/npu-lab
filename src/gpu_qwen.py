"""GPU side of the Qwen3 engine: prompt reading (prefill), KV cache and attention live in VRAM (the embedding table stays out of it).
The NPU keeps the decode matvecs (the weight stream); per token and layer only q/k/v go to the GPU and the attention output comes back
(a ~20 us round trip), see ROADMAP "Plan: NPU decodes, the RTX 3050 reads and remembers".

  g = GpuCore(config_dict, weights, max_ctx)    weights.raw(name) -> bf16 tensor as uint16 numpy
  g.prefill(ids, start)  -> last hidden state (numpy float32, before the final norm); fills the KV cache from `start`
  g.attn(layer, q, k, v, pos) -> float32 attention output (numpy, NH*HD) for one token; stores k, v at `pos`
"""
import numpy as np
import torch
import torch.nn.functional as F
from gpu_qwen35 import Lin

DEV = "cuda"
CHUNK = 2048          # prompt tokens per forward pass (bounds activation memory)
RESERVE = 0.5e9       # VRAM kept for activations and the attention scratch space
MIN_CTX = 2048


def available():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _bf16(w, name):
    import warnings
    with warnings.catch_warnings():                              # read-only map of the file: it is only read
        warnings.simplefilter("ignore")
        return torch.from_numpy(w.raw(name).view(np.int16)).to(DEV).view(torch.bfloat16)    # uint16 bits -> bf16 on the GPU


class GpuCore:
    def __init__(self, c, w, max_ctx, vram_gb=4.0, int8=None):
        self.NH, self.NKV, self.HD = c["num_attention_heads"], c["num_key_value_heads"], c.get("head_dim", c["hidden_size"] // c["num_attention_heads"])
        self.NL, self.eps, self.theta, self.max_ctx = c["num_hidden_layers"], c["rms_norm_eps"], c["rope_theta"], max_ctx
        self.qk_norm = c.get("model_type") == "qwen3"
        self.rep = self.NH // self.NKV
        self.embed_raw = w.memmap("model.embed_tokens.weight")                   # rows are gathered on the CPU, the table never enters VRAM
        if int8 is None:                                                          # int8 only when the wanted window would not fit next to bf16 weights
            nbytes = sum(m["data_offsets"][1] - m["data_offsets"][0] for k, m in w.h.items() if k.startswith("model.layers."))
            int8 = (vram_gb * 1e9 - RESERVE - nbytes) // (self.NL * self.NKV * self.HD * 4 + 2 * self.HD * 4) < max_ctx
        self.int8 = int8
        W = lambda t: Lin(t, int8)
        self.layers = []
        for i in range(self.NL):
            p = f"model.layers.{i}."
            L = {n: _bf16(w, p + k) for n, k in (("ln1", "input_layernorm.weight"), ("ln2", "post_attention_layernorm.weight"))}
            for n, k in (("o", "self_attn.o_proj.weight"), ("gate", "mlp.gate_proj.weight"), ("up", "mlp.up_proj.weight"), ("down", "mlp.down_proj.weight")):
                L[n] = W(_bf16(w, p + k))
            L["qkv"] = W(torch.cat([_bf16(w, p + f"self_attn.{n}_proj.weight") for n in "qkv"]))     # one matmul for q, k, v
            if self.qk_norm:
                L["qn"], L["kn"] = _bf16(w, p + "self_attn.q_norm.weight"), _bf16(w, p + "self_attn.k_norm.weight")
            self.layers.append(L)
        per_token = self.NL * self.NKV * self.HD * 2 * 2                                  # K and V, bf16
        room = vram_gb * 1e9 - torch.cuda.memory_allocated() - RESERVE
        fit = int(room // (per_token + 2 * self.HD * 4))                                   # + the rope tables
        if fit < MIN_CTX:
            raise MemoryError(f"only {max(fit, 0)} tokens of context fit in a {vram_gb:g} GB budget after the weights "
                              f"({torch.cuda.memory_allocated() / 1e9:.1f} GB); raise the GPU memory limit")
        self.limited = fit < max_ctx                                                       # the wanted window does not fit
        max_ctx = self.max_ctx = min(max_ctx, fit)
        self.K = torch.zeros(self.NL, self.NKV, max_ctx, self.HD, dtype=torch.bfloat16, device=DEV)
        self.V = torch.zeros_like(self.K)
        inv = 1.0 / self.theta ** (torch.arange(0, self.HD, 2, dtype=torch.float32, device=DEV) / self.HD)
        ang = torch.arange(max_ctx, dtype=torch.float32, device=DEV)[:, None] * inv[None, :]
        self.cos, self.sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)      # (max_ctx, HD) float32
        self.pin = torch.empty((self.NH + 2 * self.NKV) * self.HD, dtype=torch.float32).pin_memory()
        self.pout = torch.empty(self.NH * self.HD, dtype=torch.float32).pin_memory()

    # ---- pieces
    def _rms(self, x, w):
        xf = x.float()
        return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)).to(x.dtype) * w

    def _rope(self, x, pos):                       # x: (..., T, HD) bf16, pos: (T,) positions
        c, s = self.cos[pos], self.sin[pos]
        h = self.HD // 2
        xf = x.float()
        return (xf * c + torch.cat([-xf[..., h:], xf[..., :h]], -1) * s).to(x.dtype)

    # ---- prompt reading
    @torch.no_grad()
    def prefill(self, ids, start=0):
        ids = np.asarray(ids, dtype=np.int64)
        for a in range(0, len(ids), CHUNK):
            rows = np.ascontiguousarray(self.embed_raw[ids[a:a + CHUNK]]).view(np.int16)             # gather the embedding rows from the mapped file
            last = self._forward(torch.from_numpy(rows).to(DEV).view(torch.bfloat16), start + a, a + CHUNK >= len(ids))
        return last

    def _forward(self, x, start, want_logits):
        T, NH, NKV, HD = x.shape[0], self.NH, self.NKV, self.HD
        S = start + T
        pos = torch.arange(start, S, device=DEV)
        mask = torch.arange(S, device=DEV)[None, :] <= pos[:, None]                          # (T, S) causal over the cached prefix too
        for i, L in enumerate(self.layers):
            qkv = L["qkv"](self._rms(x, L["ln1"]))
            q = qkv[:, :NH * HD].view(T, NH, HD)
            k = qkv[:, NH * HD:(NH + NKV) * HD].view(T, NKV, HD)
            v = qkv[:, (NH + NKV) * HD:].view(T, NKV, HD)
            if self.qk_norm:
                q, k = self._rms(q, L["qn"]), self._rms(k, L["kn"])
            q = self._rope(q.transpose(0, 1), pos)                                           # (NH, T, HD)
            self.K[i, :, start:S] = self._rope(k.transpose(0, 1), pos)
            self.V[i, :, start:S] = v.transpose(0, 1)
            att = F.scaled_dot_product_attention(q[None], self.K[i, None, :, :S], self.V[i, None, :, :S], attn_mask=mask, enable_gqa=True)
            x = x + L["o"](att[0].transpose(0, 1).reshape(T, NH * HD))
            h = self._rms(x, L["ln2"])
            x = x + L["down"](F.silu(L["gate"](h)) * L["up"](h))
        return x[-1].float().cpu().numpy() if want_logits else None

    # ---- one decoded token, attention only (the matvecs run on the NPU)
    @torch.no_grad()
    def attn(self, i, q, k, v, pos):
        NH, NKV, HD = self.NH, self.NKV, self.HD
        self.pin.numpy()[:] = np.concatenate([q.ravel(), k.ravel(), v.ravel()])
        g = self.pin.to(DEV, non_blocking=True)
        qg = g[:NH * HD].view(1, NH, 1, HD).to(torch.bfloat16)
        self.K[i, :, pos] = g[NH * HD:(NH + NKV) * HD].view(NKV, HD).to(torch.bfloat16)
        self.V[i, :, pos] = g[(NH + NKV) * HD:].view(NKV, HD).to(torch.bfloat16)
        o = F.scaled_dot_product_attention(qg, self.K[i, None, :, :pos + 1], self.V[i, None, :, :pos + 1], enable_gqa=True)
        self.pout.copy_(o.reshape(-1).float(), non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return self.pout.numpy().copy()
