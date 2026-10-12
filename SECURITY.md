# Security

This tool can read your files (semantic search) and runs native kernels on your NPU. Please report problems privately via a
**GitHub private security advisory** on this repository, not in public issues.

In scope: reading files that the rules should exclude, leaking indexed content, unsafe handling of model or kernel files,
anything that executes untrusted input. Out of scope: performance bugs and unsupported hardware.

In scope also: the server fetching URLs on the web search's behalf (SSRF), API keys leaking through responses/logs, and the cloud-assist path sending more than documented.

Notes for users:
- The index (`~/.local/share/npu-lab/index.pkl`) is a pickle of your own data. Never load an index someone else gave you.
- Review `~/.config/npu-lab/rules.json` before indexing; it is the only thing deciding what may be read.
