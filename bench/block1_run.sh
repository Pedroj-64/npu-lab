#!/bin/sh
# Block 1 driver (bench/PROPUESTA_GPU_SANA.md): one thing at a time on GPU/NPU, pre-flight before every stage, every command bounded. Log: bench/.block1.log
cd "$(dirname "$0")/.." || exit 2
P=.venv/bin/python
stage() { echo; echo "##### $* $(date +%T)"; sh bench/block1_preflight.sh || { echo "ABORT: preflight failed before: $*"; exit 1; }; }
E() { timeout 900 $P bench/energy.py --json bench/.block1_energy.jsonl "$@"; }
stage "A 4B share 0.25"; E --model models/qwen35_4b --share 0.25 --vram-gb 5.0 --runs 3
stage "A 4B share 0.5";  E --model models/qwen35_4b --share 0.5  --vram-gb 5.0 --runs 3 || { stage "A 4B share 0.4"; E --model models/qwen35_4b --share 0.4 --vram-gb 5.0 --runs 3 || { stage "A 4B share 0.3"; E --model models/qwen35_4b --share 0.3 --vram-gb 5.0 --runs 3; }; }
for s in 0.5 0.6 0.65 0.7 0.75; do stage "B 2B share $s"; E --model models/qwen35_2b --share $s --phases decode --runs 3; done
stage "C step_breakdown 2B"; timeout 900 $P bench/step_breakdown.py --model models/qwen35_2b
stage "C step_breakdown 4B"; timeout 900 $P bench/step_breakdown.py --model models/qwen35_4b
stage "D prefix reuse 2B"; timeout 1500 $P bench/prefix_reuse.py --model models/qwen35_2b
stage "D prefix reuse 4B"; timeout 1800 $P bench/prefix_reuse.py --model models/qwen35_4b
for i in 1 2 3; do
  stage "E 1.7B NPU-only $i"; E --model models/qwen3 --engine npu --phases decode --runs 3
  stage "E 1.7B Assisted $i"; E --model models/qwen3 --engine assisted --phases decode --runs 3
done
echo; echo "##### FIN $(date +%T)"
