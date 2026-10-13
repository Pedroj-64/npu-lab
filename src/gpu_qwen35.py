"""GPU side of the Qwen3.5 engine (hybrid: Gated DeltaNet linear-attention layers + gated full-attention layers).
Prompt reading (prefill), the DeltaNet conv/recurrent state, the KV cache of the full-attention layers and all attention maths live in VRAM;
the NPU runs the matvecs of every layer during decode (see src/qwen35_npu.py). The embedding table and lm_head stay out of VRAM.

The DeltaNet chunked algorithm below is adapted from Hugging Face transformers (modeling_qwen3_5.py, Apache-2.0, Copyright 2025 The Qwen team
and The HuggingFace Inc. team); everything else follows the same reference so the results can be checked against it.

  g = GpuCore35(text_config, weights, prefix, max_ctx, vram_gb)
  g.prefill(embeds, start, marks)  -> last hidden state (float32 numpy, before the final norm); snapshots states at `marks`
  g.linear_step(i, inp) / g.attn_step(i, qg, k, v, pos)   one token of one layer (inputs come from the NPU matvecs)
  g.forward_one(embed_vec, pos)    one token fully on the GPU (reference / tests)
"""
import numpy as np
import torch
import torch.nn.functional as F

DEV = "cuda"
CHUNK = 1024          # prompt tokens per GPU pass: same speed as 2048 (5.3 s vs 5.2 s per 4096 tokens, 4B) with half the activation memory (355 vs 708 MiB)
RESERVE = 0.7e9
OVERHEAD = 1.05e9      # what a card needs beyond llm_vram_gb: CUDA context + prefill activations (~0.6 GiB measured, bench/vram_peak.py) + decode CUDA graphs (~0.65 GiB with the 4B)
QBLOCK = 512
MIN_CTX = 2048
MAX_SNAPSHOTS = 48


def available():
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


def _bf16(w, name):
    import warnings
    with warnings.catch_warnings():                              # read-only map of the file: it is only read
        warnings.simplefilter("ignore")
        return torch.from_numpy(w.raw(name).view(np.int16)).to(DEV).view(torch.bfloat16)


def _f32(w, name):
    r = w.h[name]
    if r["dtype"] == "F32":
        f = w.f[w.of[name]]; a, b = r["data_offsets"]; f.seek(w.base[w.of[name]] + a)
        return torch.from_numpy(np.frombuffer(f.read(b - a), dtype=np.float32).copy()).reshape(r["shape"]).to(DEV)
    return _bf16(w, name).float()


class Lin:
    """y = x @ W.T with W resident in VRAM either as bf16 or as int8 + a per-row scale (the same quantisation the NPU uses, so a model
    that does not fit in bf16 still reads prompts with the weights it decodes with)."""
    def __init__(self, W, int8):
        if int8:
            self.q = torch.empty(W.shape, dtype=torch.int8, device=W.device)
            self.s = torch.empty(W.shape[0], dtype=torch.bfloat16, device=W.device)
            for r in range(0, W.shape[0], 4096):                  # blocks of rows: no float32 copy of the whole matrix (1 GB for the output head)
                blk = W[r:r + 4096].float()
                sc = blk.abs().amax(1) / 127
                sc[sc == 0] = 1
                self.q[r:r + 4096] = torch.round(blk / sc[:, None]).to(torch.int8)
                self.s[r:r + 4096] = sc.to(torch.bfloat16)
        else:
            self.q, self.s = W, None

    def __call__(self, x):
        if self.s is None:
            return x @ self.q.T
        return (x @ self.q.to(x.dtype).T) * self.s

    def rows(self, a, b, x):
        """y = x @ W[a:b].T : a slice of the output rows (the GPU's share of a matvec that the NPU also works on)."""
        if self.s is None:
            return x @ self.q[a:b].T
        return (x @ self.q[a:b].to(x.dtype).T) * self.s[a:b]


def zrms(x, w, eps):
    """Zero-centred RMSNorm of Qwen3.5: x / rms(x) * (1 + w), computed in float32."""
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w.float())).to(x.dtype)


def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def chunk_gated_delta_rule(query, key, value, g, beta, initial_state, chunk_size=64):
    """Gated delta rule over a block of tokens. query/key: (1,T,H,Dk) value: (1,T,H,Dv) g,beta: (1,T,H). Returns (out (1,T,H,Dv), final state (1,H,Dk,Dv))."""
    dt = query.dtype
    B, T, _, Dk = key.shape
    H, Dv = value.shape[-2:]
    query, key, value, beta, decay = [x.transpose(1, 2).to(torch.float32).contiguous() for x in (query, key, value, beta, g)]
    query, key = _l2norm(query), _l2norm(key)
    query = query * Dk ** -0.5
    pad = (chunk_size - T % chunk_size) % chunk_size
    query, key, value = (F.pad(x, (0, 0, 0, pad)) for x in (query, key, value))
    beta, decay = (F.pad(x, (0, pad)) for x in (beta, decay))
    n = (T + pad) // chunk_size
    v_beta, k_beta = value * beta.unsqueeze(-1), key * beta.unsqueeze(-1)
    query, key, k_beta, v_beta = [x.reshape(x.shape[0], x.shape[1], -1, chunk_size, x.shape[-1]) for x in (query, key, k_beta, v_beta)]
    decay = decay.reshape(decay.shape[0], decay.shape[1], -1, chunk_size)
    upper = torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=query.device).triu(1)
    cum = decay.cumsum(dim=3)
    pair = (cum.unsqueeze(4) - cum.unsqueeze(3)).masked_fill(upper, float("-inf")).exp()
    ut = (k_beta @ key.transpose(-1, -2)) * pair
    intra = (query @ key.transpose(-1, -2)) * pair
    new_values = torch.linalg.solve_triangular(ut, v_beta, upper=False, unitriangular=True)
    k_cumdecay = torch.linalg.solve_triangular(ut, k_beta * cum.exp().unsqueeze(-1), upper=False, unitriangular=True)
    S = initial_state.to(new_values)
    out = torch.zeros_like(new_values)
    query = query * cum.exp().unsqueeze(-1)
    key = key * (cum[..., -1:] - cum).exp().unsqueeze(-1)
    chunk_decay = cum[..., -1].exp()[..., None, None]
    for i in range(n):
        v_new = new_values[:, :, i] - k_cumdecay[:, :, i] @ S
        out[:, :, i] = query[:, :, i] @ S + intra[:, :, i].masked_fill(upper, 0) @ v_new
        S = S * chunk_decay[:, :, i] + key[:, :, i].transpose(-1, -2) @ v_new
    out = out.reshape(B, H, -1, Dv)[:, :, :T].transpose(1, 2).to(dt)
    return out, S


class GpuCore35:
    def __init__(self, c, w, prefix, max_ctx, vram_gb=4.0, int8=None):
        self.c = c
        cap = (torch.cuda.mem_get_info()[1] - OVERHEAD) / 1e9          # a budget above this runs out of memory when the first decode graphs are captured
        self.vram_asked, vram_gb = vram_gb, min(vram_gb, cap)
        self.H, self.NL, self.eps = c["hidden_size"], c["num_hidden_layers"], c["rms_norm_eps"]
        self.NH, self.NKV, self.HD = c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        self.Nk, self.Nv, self.Dk, self.Dv = c["linear_num_key_heads"], c["linear_num_value_heads"], c["linear_key_head_dim"], c["linear_value_head_dim"]
        self.kd, self.vd = self.Nk * self.Dk, self.Nv * self.Dv
        self.conv_dim, self.ck = 2 * self.kd + self.vd, c["linear_conv_kernel_dim"]
        rp = c["rope_parameters"]
        self.rot = int(self.HD * rp.get("partial_rotary_factor", 1.0))
        self.types = c["layer_types"]
        self.full_ids = [i for i, t in enumerate(self.types) if t == "full_attention"]
        self.slot = {i: j for j, i in enumerate(self.full_ids)}
        self.norm = _bf16(w, prefix + "norm.weight")
        if int8 is None:                                            # int8 only when the wanted window would not fit next to bf16 weights
            nbytes = sum(m["data_offsets"][1] - m["data_offsets"][0] for k, m in w.h.items() if k.startswith(prefix + "layers."))
            per_tok = len(self.full_ids) * self.NKV * self.HD * 4 + 2 * self.rot * 4 + 6 * QBLOCK
            int8 = (vram_gb * 1e9 - RESERVE - nbytes) // per_tok < max_ctx
        self.int8 = int8
        W = lambda t: Lin(t, int8)
        self.layers = []
        for i in range(self.NL):
            p = f"{prefix}layers.{i}."
            L = {"type": self.types[i], "ln1": _bf16(w, p + "input_layernorm.weight"), "ln2": _bf16(w, p + "post_attention_layernorm.weight"),
                 "gu": W(torch.cat([_bf16(w, p + "mlp.gate_proj.weight"), _bf16(w, p + "mlp.up_proj.weight")])), "down": W(_bf16(w, p + "mlp.down_proj.weight"))}
            if self.types[i] == "linear_attention":
                a = p + "linear_attn."
                L["inp"] = W(torch.cat([_bf16(w, a + n + ".weight") for n in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a")]))
                L["conv"] = _bf16(w, a + "conv1d.weight").squeeze(1).float()            # (conv_dim, 4)
                L["A"], L["dtb"], L["nw"] = _f32(w, a + "A_log"), _f32(w, a + "dt_bias"), _f32(w, a + "norm.weight")
                L["out"] = W(_bf16(w, a + "out_proj.weight"))
            else:
                a = p + "self_attn."
                L["qkv"] = W(torch.cat([_bf16(w, a + n + "_proj.weight") for n in "qkv"]))  # q rows are [q(HD), gate(HD)] per head
                L["o"], L["qn"], L["kn"] = W(_bf16(w, a + "o_proj.weight")), _bf16(w, a + "q_norm.weight"), _bf16(w, a + "k_norm.weight")
            self.layers.append(L)
        per_token = len(self.full_ids) * self.NKV * self.HD * 2 * 2
        room = vram_gb * 1e9 - torch.cuda.memory_allocated() - RESERVE
        fit = int(room // (per_token + 2 * self.rot * 4 + 6 * QBLOCK))        # + rope tables + the attention mask/softmax scratch, which grows with the window
        if fit < MIN_CTX:
            raise MemoryError(f"only {max(fit, 0)} tokens of context fit in a {vram_gb:g} GB budget after the weights "
                              f"({torch.cuda.memory_allocated() / 1e9:.1f} GB); raise the GPU memory limit")
        self.limited = fit < max_ctx
        self.max_ctx = min(max_ctx, fit)
        self.K = torch.zeros(len(self.full_ids), self.NKV, self.max_ctx, self.HD, dtype=torch.bfloat16, device=DEV)
        self.V = torch.zeros_like(self.K)
        inv = 1.0 / rp["rope_theta"] ** (torch.arange(0, self.rot, 2, dtype=torch.float32, device=DEV) / self.rot)
        ang = torch.arange(self.max_ctx, dtype=torch.float32, device=DEV)[:, None] * inv[None, :]
        self.cos, self.sin = torch.cat([ang.cos()] * 2, -1), torch.cat([ang.sin()] * 2, -1)    # (max_ctx, rot)
        self.snaps = {}
        self.graphs = {}                                              # per-layer CUDA graphs of the DeltaNet decode step
        self.pool = torch.cuda.graph_pool_handle()                    # one memory pool for every graph: they never run at the same time
        self.reset_states()
        self.side = torch.cuda.Stream()                                # the GPU's share of NPU matvecs runs here (see GpuTail)
        self.tin = torch.empty(max(self.H, self.c["intermediate_size"], self.vd, self.NH * self.HD), dtype=torch.float32).pin_memory()
        self.tout = torch.empty(max(2 * self.c["intermediate_size"], self.c["vocab_size"]), dtype=torch.float32).pin_memory()
        self.pin = torch.empty(max(self.conv_dim + self.vd + 2 * self.Nv, (self.NH * 2 + 2 * self.NKV) * self.HD), dtype=torch.float32).pin_memory()
        self.pout = torch.empty(max(self.vd, self.NH * self.HD), dtype=torch.float32).pin_memory()

    def lm_tail(self, embed_raw, a, b):
        """The GPU's rows a..b of the (tied) output head, kept in VRAM like the layer weights (bf16, or int8 when the layers are)."""
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = torch.from_numpy(np.ascontiguousarray(embed_raw[a:b]).view(np.int16)).to(DEV).view(torch.bfloat16)
        return Lin(t, self.int8)

    # ---- recurrent state
    def reset_states(self):
        """Zero the DeltaNet states. The tensors are allocated once and only ever modified in place: the CUDA graphs keep their addresses."""
        if not hasattr(self, "conv"):
            self.conv = {i: torch.zeros(self.conv_dim, self.ck - 1, device=DEV) for i, t in enumerate(self.types) if t == "linear_attention"}
            self.rec = {i: torch.zeros(1, self.Nv, self.Dk, self.Dv, device=DEV) for i in self.conv}
        for t in (*self.conv.values(), *self.rec.values()):
            t.zero_()
        self.snaps = {}

    def snapshot(self, pos):
        self.snaps[pos] = ({i: t.cpu() for i, t in self.conv.items()}, {i: t.cpu() for i, t in self.rec.items()})
        while len(self.snaps) > MAX_SNAPSHOTS:                       # drop the oldest
            del self.snaps[min(self.snaps)]

    def rewind(self, L):
        """Restore the newest snapshot at a position <= L (the DeltaNet state cannot be cut back). Returns that position (0 = from scratch)."""
        ok = [p for p in self.snaps if p <= L]
        if not ok:
            self.reset_states()
            return 0
        p = max(ok)
        conv, rec = self.snaps[p]
        for i, t in conv.items():
            self.conv[i].copy_(t)
        for i, t in rec.items():
            self.rec[i].copy_(t)
        self.snaps = {q: s for q, s in self.snaps.items() if q <= p}
        return p

    # ---- shared pieces
    def _rope(self, x, pos):                                       # x: (..., T, HD); only the first `rot` dims rotate
        c, s = self.cos[pos], self.sin[pos]
        h = self.rot // 2
        xr, xp = x[..., :self.rot].float(), x[..., self.rot:]
        r = xr * c + torch.cat([-xr[..., h:], xr[..., :h]], -1) * s
        return torch.cat([r.to(x.dtype), xp], -1)

    def _mlp(self, x, L):
        h = zrms(x, L["ln2"], self.eps)
        g, u = L["gu"](h).chunk(2, -1)
        return x + L["down"](F.silu(g) * u)

    def _gated_norm(self, core, z, nw):                            # core, z: (T, Nv, Dv)
        cf = core.float()
        return (cf * torch.rsqrt(cf.pow(2).mean(-1, keepdim=True) + self.eps) * nw * F.silu(z.float())).to(core.dtype)

    def _delta_inputs(self, mixed, b, a, L):
        """After the conv: split q/k/v, build beta and the log-decay g."""
        q, k, v = mixed.split([self.kd, self.kd, self.vd], dim=-1)
        T = mixed.shape[0]
        q, k = q.reshape(1, T, self.Nk, self.Dk), k.reshape(1, T, self.Nk, self.Dk)
        v = v.reshape(1, T, self.Nv, self.Dv)
        beta = b.sigmoid().reshape(1, T, self.Nv)
        g = (-L["A"].exp() * F.softplus(a.float() + L["dtb"])).reshape(1, T, self.Nv)
        if self.Nv > self.Nk:
            q, k = q.repeat_interleave(self.Nv // self.Nk, dim=2), k.repeat_interleave(self.Nv // self.Nk, dim=2)
        return q, k, v, beta, g

    # ---- prompt reading
    @torch.no_grad()
    def prefill(self, embeds, start=0, marks=()):
        """embeds: (T, H) bf16 tensor on the GPU holding the embeddings of the tokens at positions start..start+T. `marks`: absolute positions
        where the recurrent state is snapshotted (so a later prompt can resume there). Returns the last hidden state (numpy float32, pre-norm)."""
        T = embeds.shape[0]
        cuts = sorted({m - start for m in marks if 0 < m - start < T} | {a for a in range(CHUNK, T, CHUNK)} | {T})
        a, last = 0, None
        for e in cuts:
            last = self._forward(embeds[a:e], start + a)
            if start + e in set(marks) or e % CHUNK == 0 or e == T:
                self.snapshot(start + e)
            a = e
        return last.float().cpu().numpy()

    def _forward(self, x, start):
        T = x.shape[0]
        S = start + T
        pos = torch.arange(start, S, device=DEV)
        mask = torch.arange(S, device=DEV)[None, :] <= pos[:, None]
        for i, L in enumerate(self.layers):
            h = zrms(x, L["ln1"], self.eps)
            if L["type"] == "linear_attention":
                proj = L["inp"](h)
                qkv, z, b, a = proj.split([self.conv_dim, self.vd, self.Nv, self.Nv], dim=-1)
                z = z.reshape(T, self.Nv, self.Dv)
                seq = torch.cat([self.conv[i].T.to(qkv.dtype), qkv], 0)                       # (3+T, conv_dim)
                self.conv[i].copy_(seq[-(self.ck - 1):].T.float())
                mixed = F.silu(F.conv1d(seq.T.float()[None], L["conv"][:, None, :], groups=self.conv_dim))[0].T   # (T, conv_dim)
                q, k, v, beta, g = self._delta_inputs(mixed.to(x.dtype), b, a, L)
                core, new_state = chunk_gated_delta_rule(q, k, v, g, beta, self.rec[i])
                self.rec[i].copy_(new_state)
                core = self._gated_norm(core[0], z, L["nw"]).reshape(T, self.vd)
                x = x + L["out"](core)
            else:
                qkvg = L["qkv"](h)
                q, kk, vv = qkvg.split([self.NH * self.HD * 2, self.NKV * self.HD, self.NKV * self.HD], dim=-1)
                q = q.view(T, self.NH, 2 * self.HD)
                q, gate = q[..., :self.HD], q[..., self.HD:].reshape(T, -1)
                q = self._rope(zrms(q, L["qn"], self.eps).transpose(0, 1), pos)                 # (NH, T, HD)
                kk = self._rope(zrms(kk.view(T, self.NKV, self.HD), L["kn"], self.eps).transpose(0, 1), pos)
                j = self.slot[i]
                self.K[j, :, start:S], self.V[j, :, start:S] = kk, vv.view(T, self.NKV, self.HD).transpose(0, 1)
                att = torch.empty(self.NH, T, self.HD, dtype=q.dtype, device=DEV)
                for a in range(0, T, QBLOCK):            # blocks of queries: the mask and softmax scratch stay small at long context
                    e = min(a + QBLOCK, T)
                    Sb = start + e                       # keys beyond the block's last query are masked anyway
                    att[:, a:e] = F.scaled_dot_product_attention(q[None, :, a:e], self.K[j, None, :, :Sb], self.V[j, None, :, :Sb],
                                                                 attn_mask=mask[a:e, :Sb], enable_gqa=True)[0]
                x = x + L["o"](att.transpose(0, 1).reshape(T, -1) * torch.sigmoid(gate))
            x = self._mlp(x, L)
        return x[-1]

    # ---- one decoded token, one layer (matvecs run on the NPU)
    def _linear_work(self, i, n):
        """One DeltaNet token of layer i from the NPU's projection vector (in self.pin[:n]) to the normalised output (in self.pout)."""
        L = self.layers[i]
        g_ = self.pin[:n].to(DEV, non_blocking=True)
        qkv, z, b, a = g_.split([self.conv_dim, self.vd, self.Nv, self.Nv])
        win = torch.cat([self.conv[i], qkv[:, None]], 1)                                  # (conv_dim, 4)
        self.conv[i].copy_(win[:, 1:])
        mixed = F.silu((win * L["conv"]).sum(1))[None]                                     # (1, conv_dim)
        q, k, v, beta, g = self._delta_inputs(mixed, b[None], a[None], L)
        core, S = self._recurrent(q, k, v, g, beta, i)
        self.rec[i].copy_(S)
        out = self._gated_norm(core, z.reshape(1, self.Nv, self.Dv), L["nw"]).reshape(-1)
        self.pout[:self.vd].copy_(out.float(), non_blocking=True)

    @torch.no_grad()
    def linear_step(self, i, inp):
        n = self.conv_dim + self.vd + 2 * self.Nv
        self.pin[:len(inp)].numpy()[:] = inp
        gr = self.graphs.get(i)
        if gr is None:                                           # first use: this token runs eagerly, then a graph is captured for the next ones
            self._linear_work(i, n)
            torch.cuda.current_stream().synchronize()
            try:
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr, pool=self.pool):       # capturing records the kernels without running them: the state is untouched
                    self._linear_work(i, n)
                self.graphs[i] = gr
            except Exception:
                self.graphs[i] = False                           # keep launching kernels one by one
            return self._finish_linear()
        if gr is False:
            self._linear_work(i, n)
        else:
            gr.replay()
        return self._finish_linear()

    def _finish_linear(self):
        torch.cuda.current_stream().synchronize()
        return self.pout[:self.vd].numpy().copy()

    def _recurrent(self, q, k, v, g, beta, i):
        """One-token update of the delta-rule state (same maths as torch_recurrent_gated_delta_rule)."""
        q, k = _l2norm(q.float()[0, 0]) * self.Dk ** -0.5, _l2norm(k.float()[0, 0])         # (Nv, Dk)
        v, beta, S = v.float()[0, 0], beta.float()[0, 0], self.rec[i]
        S = S * g.float()[0, 0].exp()[None, :, None, None]
        kv = (S[0] * k[:, :, None]).sum(1)                                                  # (Nv, Dv)
        S = S + (k[:, :, None] * ((v - kv) * beta[:, None])[:, None, :])[None]
        core = (S[0] * q[:, :, None]).sum(1)                                                # (Nv, Dv)
        return core[None].to(torch.bfloat16), S

    @torch.no_grad()
    def attn_step(self, i, qg, k, v, pos):
        L, NH, NKV, HD = self.layers[i], self.NH, self.NKV, self.HD
        n = len(qg) + len(k) + len(v)
        self.pin[:n].numpy()[:] = np.concatenate([qg, k, v])
        g_ = self.pin[:n].to(DEV, non_blocking=True).to(torch.bfloat16)
        q = g_[:NH * 2 * HD].view(NH, 2 * HD)
        q, gate = q[:, :HD], q[:, HD:].reshape(-1)
        p = torch.tensor([pos], device=DEV)
        q = self._rope(zrms(q, L["qn"], self.eps)[:, None], p)                              # (NH, 1, HD)
        kk = self._rope(zrms(g_[NH * 2 * HD:(NH * 2 + NKV) * HD].view(NKV, HD), L["kn"], self.eps)[:, None], p)
        j = self.slot[i]
        self.K[j, :, pos], self.V[j, :, pos] = kk[:, 0], g_[(NH * 2 + NKV) * HD:].view(NKV, HD)
        att = F.scaled_dot_product_attention(q[None], self.K[j, None, :, :pos + 1], self.V[j, None, :, :pos + 1], enable_gqa=True)   # (1, NH, 1, HD)
        out = att.reshape(-1) * torch.sigmoid(gate)
        self.pout[:NH * HD].copy_(out.float(), non_blocking=True)
        torch.cuda.current_stream().synchronize()
        return self.pout[:NH * HD].numpy().copy()

    # ---- one token fully on the GPU (reference path used by the tests; the engine runs these matvecs on the NPU instead)
    @torch.no_grad()
    def forward_one(self, x, pos):
        """x: (H,) bf16 embedding on the GPU. Returns the hidden state (pre-norm) as a float32 numpy vector."""
        for i, L in enumerate(self.layers):
            h = zrms(x[None], L["ln1"], self.eps)[0]
            if L["type"] == "linear_attention":
                inp = L["inp"](h[None])[0].float().cpu().numpy()
                x = x + L["out"](torch.from_numpy(self.linear_step(i, inp)).to(DEV).to(x.dtype)[None])[0]
            else:
                qkvg = L["qkv"](h[None])[0].float().cpu().numpy()
                n = self.NH * self.HD * 2
                nk = self.NKV * self.HD
                x = x + L["o"](torch.from_numpy(self.attn_step(i, qkvg[:n], qkvg[n:n + nk], qkvg[n + nk:], pos)).to(DEV).to(x.dtype)[None])[0]
            x = self._mlp(x[None], L)[0]
        return x.float().cpu().numpy()


class GpuTail:
    """The GPU's share of one NPU matvec: rows a..b of a resident matrix, computed on a side stream while the NPU works on the other rows.
    start(x) returns at once; finish() waits for the result (float32 numpy). One tail is in flight at a time (the matvecs of a token run in
    order). The copy-in, matvec and copy-out are captured once as a CUDA graph, so starting a tail costs one launch instead of ~6 from Python."""
    def __init__(self, core, lin, a, b):
        self.core, self.lin, self.a, self.b = core, lin, a, b        # lin: a resident matrix (layer weight, or the output-head rows the GPU keeps)
        self.K = self.lin.q.shape[1]
        self.ev = torch.cuda.Event()
        self.graph = None
        try:
            self._capture()
        except Exception as e:                                   # fall back to launching the kernels one by one
            self.graph = None
            self._why = repr(e)

    def _work(self):
        c = self.core
        xt = c.tin[:self.K].to(DEV, non_blocking=True).to(torch.bfloat16)[None]
        y = self.lin.rows(self.a, self.b, xt)[0]
        c.tout[:self.b - self.a].copy_(y.float(), non_blocking=True)

    def _capture(self):
        c = self.core
        with torch.cuda.stream(c.side):
            self._work()                                         # warm-up (allocations, cuBLAS handles) before capturing
        c.side.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=c.pool, stream=c.side):
            self._work()
        self.graph = g

    def start(self, x):
        c = self.core
        c.tin[:len(x)].numpy()[:] = x
        with torch.cuda.stream(c.side):
            if self.graph is not None:
                self.graph.replay()
            else:
                self._work()
            self.ev.record(c.side)

    def finish(self):
        self.ev.synchronize()
        return self.core.tout[:self.b - self.a].numpy().copy()
