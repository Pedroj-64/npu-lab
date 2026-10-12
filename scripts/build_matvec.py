"""Build matvec xclbins / instruction streams.
usage: python scripts/build_matvec.py [--i8|--shared] MxK [MxK ...]
  (default)  int16 weights, 4 columns      -> build/mv/mv_MxK_c4.*
  --i8       int8 weights, 4 columns       -> build/mv8/mv8_MxK_c4.*
  --shared   int8 weights, 2 columns       -> build/c2/c2_MxK.*   (used by src/qwen_npu.py)
With --shared the .xclbin only depends on K, so two (K=896 and K=4864) stay resident on the NPU side
by side; the .bin instruction stream is what changes per M. `scripts/build_qwen.sh`-style full set:
  python scripts/build_matvec.py --shared 1152x896 896x896 9728x896 6016x896 896x4864
Needs .venv/bin on PATH and PEANO_INSTALL_DIR set (see README)."""
import os, subprocess, sys
args = sys.argv[1:]
mode = "shared" if "--shared" in args else "i8" if "--i8" in args else "i16"
d, pre, script, extra = {
    "i16": ("build/mv", "mv", "src/kernels/matvec.py", ["--n-cores", "4"]),
    "i8": ("build/mv8", "mv8", "src/kernels/matvec_i8.py", ["--n-cores", "4", "-k", "128"]),
    "shared": ("build/c2", "c2", "src/kernels/matvec_i8.py", ["--n-cores", "2", "-k", "128"]),
}[mode]
os.makedirs(d, exist_ok=True)
for shape in (a for a in args if not a.startswith("--")):
    m, k = shape.split("x")
    out = f"{d}/{pre}_{m}x{k}" + ("" if mode == "shared" else "_c4")
    if os.path.exists(out + ".xclbin"):
        continue
    r = subprocess.run([sys.executable, script, "--dev", "npu", "-M", m, "-K", k, *extra,
                        f"--xclbin-path={out}.xclbin", f"--insts-path={out}.bin"], capture_output=True, text=True)
    print(shape, "ok" if r.returncode == 0 and os.path.exists(out + ".xclbin") else "FAILED\n" + r.stdout[-600:] + r.stderr[-600:])
