"""all-MiniLM-L6-v2 sentence embeddings in plain numpy.

Every dense layer goes through `linear`, so a different backend (the NPU)
can be swapped in by passing `linear=` to `MiniLM`.
"""
import numpy as np
from safetensors import safe_open
from scipy.special import erf
from tokenizers import Tokenizer

D = "models/minilm"
H, NH, L = 384, 12, 6


def cpu_linear(x, w, b):
    return x @ w.T + b


def gelu(x):
    return 0.5 * x * (1 + erf(x * np.float32(0.7071067811865476)))


def layer_norm(x, w, b, eps=1e-12):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / np.sqrt(v + eps) * w + b


class MiniLM:
    def __init__(self, linear=cpu_linear, d=D):
        f = safe_open(f"{d}/model.safetensors", "np")
        self.w = {k: f.get_tensor(k).astype(np.float32) for k in f.keys()}
        self.tok = Tokenizer.from_file(f"{d}/tokenizer.json")
        self.tok.enable_truncation(256)
        self.tok.enable_padding()
        self.linear = linear

    def encode(self, texts):
        w, lin = self.w, self.linear
        enc = self.tok.encode_batch(texts)
        ids = np.array([e.ids for e in enc])
        mask = np.array([e.attention_mask for e in enc], dtype=np.float32)
        B, T = ids.shape
        x = (w["embeddings.word_embeddings.weight"][ids]
             + w["embeddings.position_embeddings.weight"][:T]
             + w["embeddings.token_type_embeddings.weight"][0])
        x = layer_norm(x, w["embeddings.LayerNorm.weight"], w["embeddings.LayerNorm.bias"])
        bias = (1 - mask)[:, None, None, :] * -1e9
        for i in range(L):
            p = f"encoder.layer.{i}."
            flat = x.reshape(B * T, H)

            def proj(n):
                a = lin(flat, w[p + f"attention.self.{n}.weight"], w[p + f"attention.self.{n}.bias"])
                return a.reshape(B, T, NH, H // NH).transpose(0, 2, 1, 3)

            q, k, v = proj("query"), proj("key"), proj("value")
            s = q @ k.transpose(0, 1, 3, 2) / np.sqrt(H // NH) + bias
            s = np.exp(s - s.max(-1, keepdims=True))
            s /= s.sum(-1, keepdims=True)
            ctx = (s @ v).transpose(0, 2, 1, 3).reshape(B * T, H)
            a = lin(ctx, w[p + "attention.output.dense.weight"], w[p + "attention.output.dense.bias"])
            x = layer_norm(a.reshape(B, T, H) + x, w[p + "attention.output.LayerNorm.weight"],
                           w[p + "attention.output.LayerNorm.bias"])
            h = gelu(lin(x.reshape(B * T, H), w[p + "intermediate.dense.weight"], w[p + "intermediate.dense.bias"]))
            o = lin(h, w[p + "output.dense.weight"], w[p + "output.dense.bias"]).reshape(B, T, H)
            x = layer_norm(o + x, w[p + "output.LayerNorm.weight"], w[p + "output.LayerNorm.bias"])
        e = (x * mask[:, :, None]).sum(1) / mask.sum(1, keepdims=True)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


if __name__ == "__main__":
    m = MiniLM()
    t = ["The cat sits on the mat", "A kitten is resting on a rug",
         "Quarterly tax filing deadline", "El gato duerme en la alfombra"]
    e = m.encode(t)
    print(np.round(e @ e.T, 3))
    assert e[0] @ e[1] > e[0] @ e[2], "cat/kitten should beat cat/tax"
    print("ok")
