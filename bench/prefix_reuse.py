"""Phase 3, batch 1: how much of a stable prefix is really reused between two different questions about the same project (bench/PROPUESTA_GPU_SANA.md, section D).
Scenarios for question 2 of each pair: E1 cold; E2 warm in the same process (in-process KV/snapshot reuse); E3 "new process" = in-process state wiped, only the disk StateCache survives
(wiping cache_ids and snapshots is the same code path as a fresh process, without reloading the model); E4 like E2 but with ~900 tokens of excerpts put before the question (compared to its own cold run).
Sizes: S = the real project-mode system prompt (system + FORMAT + repo map); L = the same plus source files up to ~6000 tokens (so the stable part passes the StateCache minimum of 4096).
usage: .venv/bin/python bench/prefix_reuse.py --model models/qwen35_2b      (NPU + GPU: run alone)"""
import argparse, json, os, shutil, sys, tempfile, time
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, os.path.join(ROOT, "src"))
ap = argparse.ArgumentParser()
ap.add_argument("--model", default="models/qwen35_2b")
ap.add_argument("--vram-gb", type=float, default=5.0)
ap.add_argument("--tokens", type=int, default=32)
ap.add_argument("--json", default="bench/prefix_reuse.jsonl")
a = ap.parse_args()
state_dir = tempfile.mkdtemp(prefix="npu_state_", dir=os.environ.get("TMPDIR", "/tmp"))
os.environ["NPU_LAB_STATE"] = state_dir
import numpy as np
import qwen_npu as Q, qwen35_npu as Q35, project, statecache

m = Q35.load_engine(a.model, gpu=True, gpu_share=0.0, vram_gb=a.vram_gb, max_ctx=8192, cache_gb=20.0)
disk = m.cache
orig_prefill, last = m.prefill, {}


def wrapped(*x, **k):
    r = orig_prefill(*x, **k); last["logits"] = np.array(r, dtype=np.float32); return r


m.prefill = wrapped
PAIRS = [("¿Qué hace el archivo src/statecache.py?", "¿Para qué sirve el archivo src/hybrid.py?"),
         ("Resume la estructura del proyecto en tres líneas.", "¿Qué módulos se encargan de la búsqueda de documentos?"),
         ("¿Dónde se calcula el reparto de la GPU?", "¿Cómo se compacta una conversación larga?")]
base_sys = Q.SYSTEM + project.system_block(ROOT)
code = "".join(f"\n### {p}\n" + open(os.path.join(ROOT, "src", p), encoding="utf-8").read() for p in ("serve.py", "qwen_npu.py", "qwen35_npu.py", "gpu_qwen35.py"))
code_ids = m.tok.encode(code, add_special_tokens=False).ids[:6000]
SYS = {"S": base_sys, "L": base_sys + "\n\nSource files of the project:\n" + m.tok.decode(code_ids)}
text = open(os.path.join(ROOT, "bench/ENERGY.md"), encoding="utf-8").read()
EXC = [{"src": f"file:ENERGY.md#{i}", "text": text[i * 1400:(i + 1) * 1400]} for i in range(3)]       # 3 x ~1400 characters of real project text


def wipe():
    m.cache_ids = []; m.g.snaps = {}; m.g.reset_states()


def ask(q, system, context=None):
    t = time.perf_counter(); out = "".join(m.stream(q, a.tokens, system=system, context=context)); s = dict(m.stats)
    return dict(out=out, logits=last["logits"], reread=s["prefill_tokens"] - s["reused_tokens"], total=s["prefill_tokens"], prefill_s=s["prefill_s"])


def cmp(r, ref):
    n = min(len(r["out"]), len(ref["out"]))
    cos = float(r["logits"] @ ref["logits"] / (np.linalg.norm(r["logits"]) * np.linalg.norm(ref["logits"])))
    return dict(same_text=r["out"] == ref["out"], same_first=r["out"][:1] == ref["out"][:1], cos=cos)


rows = []
for size, system in SYS.items():
    for pi, (q1, q2) in enumerate(PAIRS):
        shutil.rmtree(state_dir, ignore_errors=True); os.makedirs(state_dir)
        m.cache = None; wipe(); e1 = ask(q2, system)                                   # E1 cold
        wipe(); ask(q1, system); e2 = ask(q2, system)                                  # E2 warm, same process
        m.cache = disk; wipe(); ask(q1, system); wipe(); e3 = ask(q2, system)          # E3: only the disk cache survives
        m.cache = None; wipe(); e4c = ask(q2, system, EXC)                             # E4 cold, with excerpts
        wipe(); ask(q1, system); e4 = ask(q2, system, EXC)                             # E4 warm, with excerpts
        for name, r, ref in (("E1", e1, e1), ("E2", e2, e1), ("E3", e3, e1), ("E4cold", e4c, e4c), ("E4", e4, e4c)):
            row = dict(model=os.path.basename(a.model), size=size, pair=pi, scenario=name, reread=r["reread"], total=r["total"], prefill_s=r["prefill_s"], **cmp(r, ref))
            rows.append(row); print(json.dumps(row), flush=True)
            open(a.json, "a").write(json.dumps(dict(row, gpu_state="healthy")) + "\n")
print("\n| size | scenario | prompt tokens | re-read | prefill s | same text | cos vs reference |\n|---|---|---:|---:|---:|---:|---:|")
for size in SYS:
    for sc in ("E1", "E2", "E3", "E4cold", "E4"):
        rs = [r for r in rows if r["size"] == size and r["scenario"] == sc]
        print(f"| {size} | {sc} | {np.mean([r['total'] for r in rs]):.0f} | {np.mean([r['reread'] for r in rs]):.0f} | {np.mean([r['prefill_s'] for r in rs]):.2f} | {sum(r['same_text'] for r in rs)}/{len(rs)} | {min(r['cos'] for r in rs):.5f} |")
shutil.rmtree(state_dir, ignore_errors=True)
