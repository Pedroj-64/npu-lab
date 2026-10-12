"""Build 4-column matvec xclbins.
usage: python scripts/build_matvec.py [--i8] 1152x896 896x4864 ...
  default: int16 weights -> build/mv/mv_MxK_c4.*    --i8: int8 weights -> build/mv8/mv8_MxK_c4.*
Needs .venv/bin on PATH and PEANO_INSTALL_DIR set (see README)."""
import os, subprocess, sys
args = sys.argv[1:]
i8 = "--i8" in args
d, pre, script = ("build/mv8", "mv8", "src/kernels/matvec_i8.py") if i8 else ("build/mv", "mv", "src/kernels/matvec.py")
os.makedirs(d, exist_ok=True)
for shape in (a for a in args if a != "--i8"):
    m, k = shape.split("x")
    out = f"{d}/{pre}_{m}x{k}_c4"
    if os.path.exists(out + ".xclbin"):
        continue
    r = subprocess.run([sys.executable, script, "--dev", "npu", "-M", m, "-K", k, "--n-cores", "4", *(["-k", "128"] if i8 else []),
                        f"--xclbin-path={out}.xclbin", f"--insts-path={out}.bin"], capture_output=True, text=True)
    print(shape, "ok" if r.returncode == 0 and os.path.exists(out + ".xclbin") else "FAILED\n" + r.stdout[-600:] + r.stderr[-600:])
