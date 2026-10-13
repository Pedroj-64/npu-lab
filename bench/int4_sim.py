"""int4 weights simulated on the CPU (Phase 2 step 3; criterion fixed in bench/INT4_SIM.md BEFORE measuring). No NPU, no GPU: torch + transformers (installed), weights from models/.
Every nn.Linear of the layers and the output head is quantised and de-quantised ("fake quantisation") to bf16 from the ORIGINAL weights in the safetensors files, then the model is evaluated on a fixed
text (two segments of 384 tokens: README.md and bench/PERF.md, as in this commit): mean cosine of the logits against the unquantised bf16 model, perplexity, and (stage gen) greedy tokens on the 5 fixed prompts.
usage: .venv/bin/python bench/int4_sim.py sweep --model models/qwen3 [--only int8,A_g64_sym]     -> bench/int4_sim_<model>.jsonl (resumable: finished configurations are skipped)
       .venv/bin/python bench/int4_sim.py gen   --model models/qwen3 [--cfgs int8,B_g64_asym]
Configurations: bf16 | int8 (per-row, like qmat) | {A,B}_g{32,64,128}_{sym,asym}.  A = all int4; B = output head int8 per row, the rest int4. Scales are rounded to fp16."""
import argparse, glob, json, os, sys, time
import numpy as np, torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); os.chdir(ROOT)
ap = argparse.ArgumentParser()
ap.add_argument("stage", choices=["sweep", "gen"])
ap.add_argument("--model", required=True)
ap.add_argument("--only", default="")
ap.add_argument("--cfgs", default="")
ap.add_argument("--new", type=int, default=32)
a = ap.parse_args()
OUT = f"bench/int4_sim_{os.path.basename(a.model)}.jsonl"
CFGS = ["int8"] + [f"{v}_g{g}_{m}" for v in "AB" for g in (32, 64, 128) for m in ("sym", "asym")]
COMBOS = [f"C_{c}_g{g}_asym" for c in ("GU", "DOWN", "ATT", "HEAD", "GU+DOWN", "GU+ATT", "DOWN+ATT") for g in (32, 64, 128)]
SEG, NSEG = 384, 2
PROMPTS = ["What is the capital of France? Answer in one sentence.", "Escribe una función en Python que invierta una cadena y explica cómo funciona.",
           "Lista tres comandos de pacman para instalar, quitar y buscar paquetes.", "Explain in two sentences why the sky is blue.",
           "Read this and summarise it in one sentence: " + " ".join(f"Section {i}: the NPU streams int8 weights while the GPU keeps the cache of key and value vectors for attention." for i in range(30))]
torch.set_grad_enabled(False)


def q8(W):
    s = W.abs().amax(1, keepdim=True) / 127; s[s == 0] = 1
    return torch.round(W / s).clamp(-127, 127) * s


def q4(W, g, asym):
    o, k = W.shape; pad = (-k) % g
    Wp = torch.nn.functional.pad(W, (0, pad)) if pad else W
    G = Wp.view(o, -1, g)
    if not asym:
        s = (G.abs().amax(2, keepdim=True) / 7).half().float(); s[s == 0] = 1
        D = torch.round(G / s).clamp(-8, 7) * s
    else:
        mn, mx = G.amin(2, keepdim=True), G.amax(2, keepdim=True)
        s = ((mx - mn) / 15).half().float(); s[s == 0] = 1
        zp = torch.round(-mn / s).clamp(0, 15)
        D = (torch.round(G / s + zp).clamp(0, 15) - zp) * s
    return D.view(o, -1)[:, :k]


def cls_of(name):
    return "HEAD" if name == "lm_head" else "GU" if ("gate_proj" in name or "up_proj" in name) else "DOWN" if "down_proj" in name else "ATT"


def parse(cfg):
    """-> {class: (group, asym)} of the classes that go to int4; every other class is int8 per row."""
    if cfg == "int8":
        return {}
    if cfg.startswith("C_"):
        _, cl, g, m = cfg.split("_")
        return {c: (int(g[1:]), m == "asym") for c in cl.split("+")}
    v, g, m = cfg.split("_")
    return {c: (int(g[1:]), m == "asym") for c in (("ATT", "GU", "DOWN", "HEAD") if v == "A" else ("ATT", "GU", "DOWN"))}


def bytes_cfg(shapes, cfg, only=None):
    """MB streamed per token, scales included (restricted to class `only` if given). shapes: [(out, in, class)]."""
    q, tot = parse(cfg), 0.0
    for o, k, c in shapes:
        if only and c != only:
            continue
        if c in q:
            g, asym = q[c]; ng = -(-k // g)
            tot += o * k * 0.5 + o * ng * (2 + (0.5 if asym else 0))
        else:
            tot += o * k + 4 * o
    return tot / 1e6


tok = AutoTokenizer.from_pretrained(a.model)
model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
if model.config.get_text_config().tie_word_embeddings or getattr(model.config, "tie_word_embeddings", False):
    model.lm_head.weight = torch.nn.Parameter(model.lm_head.weight.detach().clone())          # untie: the input table stays bf16 (it is gathered by rows, not streamed)
orig_head = model.get_input_embeddings().weight.detach() if model.lm_head.weight.shape == model.get_input_embeddings().weight.shape else None
files = sorted(glob.glob(f"{a.model}/*.safetensors")); src = {}
for f in files:
    with safe_open(f, "pt") as h:
        for k in h.keys(): src[k] = f
def original(name):
    key = next((k for k in (name, name.replace("model.", "model.language_model.", 1)) if k in src), None)
    assert key, f"no original for {name}"
    with safe_open(src[key], "pt") as h: return h.get_tensor(key).float()
lins = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
shapes = [(m.weight.shape[0], m.weight.shape[1], cls_of(n)) for n, m in lins]
body = [(n, m) for n, m in lins if n != "lm_head"]; head = model.lm_head


def apply(cfg):
    t = time.perf_counter(); q = parse(cfg)
    for n, m in body:
        W = original(n + ".weight"); c = cls_of(n)
        m.weight.copy_((q4(W, *q[c]) if c in q else q8(W)).to(torch.bfloat16))
    Wh = (orig_head if orig_head is not None else original("lm_head.weight")).float()
    head.weight.copy_((q4(Wh, *q["HEAD"]) if "HEAD" in q else q8(Wh)).to(torch.bfloat16))
    return time.perf_counter() - t


def restore_bf16():
    for n, m in body: m.weight.copy_(original(n + ".weight").to(torch.bfloat16))
    head.weight.copy_((orig_head if orig_head is not None else original("lm_head.weight")).to(torch.bfloat16))


txt = [open("README.md").read(), open("bench/PERF.md").read()]
segs = [tok(t, return_tensors="pt").input_ids[:, :SEG] for t in txt]
assert all(s.shape[1] == SEG for s in segs)


def evaluate(ref=None):
    nll, cos, logits = 0.0, [], []
    for s in segs:
        lg = model(s, use_cache=False).logits[0]                                  # (SEG, V) bf16
        logits.append(lg)
        lp = torch.log_softmax(lg[:-1].float(), -1)
        nll += -lp.gather(1, s[0, 1:, None]).sum().item()
    ppl = float(np.exp(nll / (NSEG * (SEG - 1))))
    if ref is not None:
        cos = [torch.nn.functional.cosine_similarity(l.float(), r.float(), dim=-1) for l, r in zip(logits, ref)]
        c = torch.cat(cos); return ppl, float(c.mean()), float(c.min()), logits
    return ppl, 1.0, 1.0, logits


done = {json.loads(l)["cfg"]: json.loads(l) for l in open(OUT)} if os.path.exists(OUT) else {}
if a.stage == "sweep":
    want = a.only.split(",") if a.only else ["bf16"] + CFGS
    if a.only == "combos": want = ["int8"] + COMBOS
    ref_logits = None
    if "bf16" not in done or any(c not in done for c in want if c != "bf16"):
        t = time.perf_counter(); ppl0, _, _, ref_logits = evaluate(); tref = time.perf_counter() - t
        print(f"bf16 reference: ppl {ppl0:.3f} ({tref:.0f} s per evaluation)", flush=True)
        if "bf16" not in done:
            r = dict(cfg="bf16", ppl=ppl0, cos_mean=1.0, cos_min=1.0, mb_token=sum(o * k * 2 for o, k, h in shapes) / 1e6); open(OUT, "a").write(json.dumps(r) + "\n"); done["bf16"] = r
    for cfg in want:
        if cfg in done: continue
        tq = apply(cfg); t = time.perf_counter(); ppl, cm, cmin, _ = evaluate(ref_logits)
        r = dict(cfg=cfg, ppl=ppl, cos_mean=cm, cos_min=cmin, mb_token=bytes_cfg(shapes, cfg), mb_by_class={c: bytes_cfg(shapes, cfg, c) for c in ('ATT', 'GU', 'DOWN', 'HEAD')}, quant_s=tq, eval_s=time.perf_counter() - t)
        open(OUT, "a").write(json.dumps(r) + "\n"); done[cfg] = r
        print(f"{cfg:12} ppl {ppl:.3f}  cos mean {cm:.5f} min {cmin:.4f}  {r['mb_token']:.0f} MB/token  (quantise {tq:.0f} s, eval {r['eval_s']:.0f} s)", flush=True)
else:
    cfgs = a.cfgs.split(",")
    for cfg in cfgs:
        key = f"gen_{cfg}"
        if key in done: continue
        if cfg == "bf16": restore_bf16()
        else: apply(cfg)
        outs = []
        for p in PROMPTS:
            ids = tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, return_tensors="pt", enable_thinking=False, return_dict=True)["input_ids"]
            o = model.generate(ids, max_new_tokens=a.new, do_sample=False)[0, ids.shape[1]:].tolist(); outs.append(o)
        r = dict(cfg=key, ids=outs); open(OUT, "a").write(json.dumps(r) + "\n"); done[key] = r
        print(cfg, "generated", [len(o) for o in outs], flush=True)
