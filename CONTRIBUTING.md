# Contributing

Thanks for helping. This project is about getting useful work out of the **AMD XDNA1 NPU on Linux** using
custom kernels. Small, measured, reproducible contributions are the most useful.

## What is welcome
- **Hardware reports**: does it run on your chip (XDNA1 variants, XDNA2/Ryzen AI 300, other distros)?
  Open an issue with `xrt-smi examine`, your kernel version, and the numbers from `bench/`.
- **Speed or accuracy improvements** with before/after numbers (same machine, same prompt, several runs).
- **Other platforms**: Windows support for the NPU backends, other models, other GPU vendors for the optional helper.
- Bug fixes, docs, and clearer error messages.

## Ground rules
1. **Measure, don't guess.** Performance claims need a script or command in `bench/` and the result in `bench/RESULTS.md`.
2. **Keep the NPU central.** GPU/CPU paths are optional helpers: never make `torch`/CUDA a hard requirement,
   and keep VRAM and CPU use capped and configurable.
3. **Privacy first.** The file-search feature reads private data. Never commit an index, a rules file with real paths,
   model weights, `build/` output, or anything under `~/.config/npu-lab` or `~/.local/share/npu-lab`.
   Strip usernames, hostnames, e-mails and absolute home paths from logs and examples. Do not weaken the default deny lists.
4. **Small diffs.** Match the surrounding style, add comments only where the reason is not obvious, no speculative abstractions.
5. **Say what you tested** in the pull request: hardware, OS, kernel, commands run, and what you did not test.

## Workflow
1. Fork, create a branch, make a focused change.
2. Run the relevant checks: `python tests/test_server.py`, `cd launcher && cargo test`, `python src/minilm.py models/multi`,
   `python bench/matvec8_npu.py ...`, `python src/qwen_npu.py "prompt"` (see the README for setup).
3. Open a pull request describing the change and the numbers.

Commit messages: a clear English subject line; a second-language line is welcome but optional.

## License of contributions
By contributing you agree your work is released under the repository's **MIT License**. Files derived from mlir-aie keep
their Apache-2.0 WITH LLVM-exception header (see `NOTICE.md`); do not remove it. Only submit code you wrote or have the
right to submit.

## Conduct
Be respectful and constructive (see `CODE_OF_CONDUCT.md`). Report problems through a private GitHub security advisory
or an issue, never by posting someone's personal data.
