"""Build 4-column matvec xclbins. usage: python scripts/build_matvec.py 1152x896 896x4864 ...
Needs .venv/bin on PATH and PEANO_INSTALL_DIR set (see README)."""
import os, subprocess, sys
os.makedirs("build/mv", exist_ok=True)
for shape in sys.argv[1:]:
    m, k = shape.split("x")
    out = f"build/mv/mv_{m}x{k}_c4"
    if os.path.exists(out + ".xclbin"):
        continue
    r = subprocess.run([sys.executable, "src/kernels/matvec.py", "--dev", "npu", "-M", m, "-K", k,
                        "--n-cores", "4", f"--xclbin-path={out}.xclbin", f"--insts-path={out}.bin"],
                       capture_output=True, text=True)
    print(shape, "ok" if r.returncode == 0 and os.path.exists(out + ".xclbin") else "FAILED\n" + r.stdout[-600:] + r.stderr[-600:])
