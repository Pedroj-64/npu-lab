"""Build the bf16 prefill matmul kernels a model needs.
usage: python scripts/build_prefill.py models/qwen [models/qwen15 ...]
Needs .venv/bin on PATH and PEANO_INSTALL_DIR set (see README)."""
import json, os, subprocess, sys
sys.path.insert(0, "src")
from prefill_npu import CH, COLS, shapes, xclbin_name
os.makedirs("build/mm", exist_ok=True)
for d in sys.argv[1:]:
    for K, N in shapes(json.load(open(f"{d}/config.json"))):
        out = xclbin_name(K, N)
        if os.path.exists(out + ".xclbin"):
            continue
        r = subprocess.run([sys.executable, "src/kernels/matmul_whole_array.py", "-m", "32", "-k", "32", "-n", "32", "--n-aie-cols", str(COLS),
                            "--b-col-maj", "0", "--c-col-maj", "0", "--dev", "npu", "-M", str(CH), "-K", str(K), "-N", str(N),
                            "--dtype_in", "bf16", "--dtype_out", "f32", "--use-chess", "0", "--emulate-bf16-mmul-with-bfp16", "0",
                            f"--xclbin-path={out}.xclbin", f"--insts-path={out}.bin"], capture_output=True, text=True)
        print(f"{K}x{N}", "ok" if r.returncode == 0 and os.path.exists(out + ".xclbin") else "FAILED\n" + r.stderr[-500:])
