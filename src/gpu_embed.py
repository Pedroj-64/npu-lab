"""Optional GPU backend for the same embedding model (torch, fp16, hard VRAM cap).

Same interface as MiniLM.encode, so vectors are interchangeable with the NPU/CPU ones.
Needs `torch` with CUDA; nothing else in the project does.
"""
import json
import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open
from tokenizers import Tokenizer


class GpuEmbedder:
    def __init__(self, d="models/multi", max_len=128, vram_gb=1.5, batch=64):
        total = torch.cuda.get_device_properties(0).total_memory / 2**30
        torch.cuda.set_per_process_memory_fraction(min(1.0, vram_gb / total))
        f = safe_open(f"{d}/model.safetensors", "np")
        self.w = {k: torch.from_numpy(f.get_tensor(k).astype(np.float32)).half().cuda()
                  for k in f.keys() if "position_ids" not in k}
        cfg = json.load(open(f"{d}/config.json"))
        self.L, self.H, self.NH = cfg["num_hidden_layers"], cfg["hidden_size"], cfg["num_attention_heads"]
        self.tok = Tokenizer.from_file(f"{d}/tokenizer.json")
        self.tok.enable_truncation(max_len)
        self.tok.enable_padding()
        self.batch = batch

    @torch.no_grad()
    def _enc(self, texts):
        w, H, NH = self.w, self.H, self.NH
        enc = self.tok.encode_batch(texts)
        ids = torch.tensor([e.ids for e in enc]).cuda()
        mask = torch.tensor([e.attention_mask for e in enc]).cuda()
        B, T = ids.shape
        ln = lambda x, p: F.layer_norm(x, (H,), w[p + ".weight"], w[p + ".bias"], 1e-12)
        x = (w["embeddings.word_embeddings.weight"][ids] + w["embeddings.position_embeddings.weight"][:T]
             + w["embeddings.token_type_embeddings.weight"][0])
        x = ln(x, "embeddings.LayerNorm")
        am = (mask[:, None, None, :] == 1)
        for i in range(self.L):
            p = f"encoder.layer.{i}."
            lin = lambda x, n: F.linear(x, w[p + n + ".weight"], w[p + n + ".bias"])
            sp = lambda t: t.view(B, T, NH, H // NH).transpose(1, 2)
            q, k, v = (sp(lin(x, f"attention.self.{n}")) for n in ("query", "key", "value"))
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=am).transpose(1, 2).reshape(B, T, H)
            x = ln(lin(a, "attention.output.dense") + x, p + "attention.output.LayerNorm")
            x = ln(lin(F.gelu(lin(x, "intermediate.dense")), "output.dense") + x, p + "output.LayerNorm")
        m = mask[:, :, None].half()
        e = ((x * m).sum(1) / m.sum(1)).float()
        return F.normalize(e, dim=1).cpu().numpy()

    def encode(self, texts):
        return np.concatenate([self._enc(texts[i:i + self.batch]) for i in range(0, len(texts), self.batch)])
