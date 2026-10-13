"""Quality of a GPU share against share 0 (Qwen3.5, decode path): (a) cosine of the logits after each of the 5 fixed prompts (bench/token_equiv.py prompts), (b) perplexity on a fixed text
teacher-forced through the decode step, (c) first token where the greedy text diverges from share 0 (informative).
usage: .venv/bin/python bench/share_quality.py run SHARE [--model models/qwen35_2b]     -> bench/.share_quality_<model>_<share>.json
       .venv/bin/python bench/share_quality.py cmp BASE SHARE...                                          (NPU + GPU: run alone)"""
import json, os, sys
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
import numpy as np
arg = lambda k, d: type(d)(sys.argv[sys.argv.index(k) + 1]) if k in sys.argv else d
model = arg("--model", "models/qwen35_2b")
f = lambda s: f"bench/.share_quality_{os.path.basename(model)}_{s}.json"
doc = " ".join(f"Section {i}: the NPU streams int8 weights while the GPU keeps the cache of key and value vectors for attention." for i in range(30))
PROMPTS = ["What is the capital of France? Answer in one sentence.", "Escribe una función en Python que invierta una cadena y explica cómo funciona.",
           "Lista tres comandos de pacman para instalar, quitar y buscar paquetes.", "Explain in two sentences why the sky is blue.", f"Read this and summarise it in one sentence: {doc}"]
TEXT = ("The Roman Empire reached its greatest extent under Trajan, stretching from Britain to Mesopotamia. Its roads, aqueducts and legal ideas outlived the state that built them. "
        "La fotosíntesis es el proceso por el cual las plantas convierten la luz del sol, el agua y el dióxido de carbono en glucosa y oxígeno. Sin ella, casi toda la vida en la Tierra desaparecería. "
        "In computing, a cache keeps recently used data close to the processor so that the next request can be answered without the slow trip to main memory. "
        "Para instalar un paquete en Arch Linux se usa pacman, y para buscar uno se usa pacman -Ss seguido del nombre.")
if sys.argv[1] == "run":
    share = sys.argv[2]
    import qwen35_npu as Q35
    m = Q35.load_engine(model, gpu=True, gpu_share=float(share), vram_gb=4.5, max_ctx=4096)
    out = dict(prompts=[])
    for p in PROMPTS:
        text = "".join(m.stream(p, 32)); ids = m.tok.encode(text, add_special_tokens=False).ids[:32]
        ids_p = m.tok.encode(f"<|im_start|>system\n{Q35.Q.SYSTEM}<|im_end|>\n<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n", add_special_tokens=False).ids
        m.cache_ids = []; lg = np.array(m.prefill(ids_p, 0), dtype=np.float64)
        out["prompts"].append(dict(tokens=ids, logits=lg.astype(np.float32).tolist()))
    ids = m.tok.encode(TEXT, add_special_tokens=False).ids
    m.cache_ids = []; lg = np.array(m.prefill(ids[:1], 0), dtype=np.float64); nll = []
    for i in range(1, len(ids)):
        z = lg - lg.max(); nll.append(float(np.log(np.exp(z).sum()) - z[ids[i]]))
        lg = np.array(m.step(ids[i], i), dtype=np.float64)
    out["nll"] = nll; out["n"] = len(nll)
    json.dump(out, open(f(share), "w"))
    print(f"share {share}: perplexity {np.exp(np.mean(nll)):.4f} over {len(nll)} tokens")
else:
    base = json.load(open(f(sys.argv[2])))
    for s in sys.argv[3:]:
        o = json.load(open(f(s)))
        cos = [float(np.dot(a["logits"], b["logits"]) / (np.linalg.norm(a["logits"]) * np.linalg.norm(b["logits"]))) for a, b in zip(base["prompts"], o["prompts"])]
        div = [next((i for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])) if x != y), None) for a, b in zip(base["prompts"], o["prompts"])]
        pb, ps = float(np.exp(np.mean(base["nll"]))), float(np.exp(np.mean(o["nll"])))
        print(f"share {s} vs {sys.argv[2]}: min cosine {min(cos):.6f} (per prompt {[round(c, 6) for c in cos]}); perplexity {ps:.4f} vs {pb:.4f} ({(ps / pb - 1) * 100:+.2f} %); first diverging token per prompt {div}")
