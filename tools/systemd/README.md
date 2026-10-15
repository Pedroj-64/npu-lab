# Weekly timers / Timers semanales

Optional user timers for `bench/watch.py` (continuous measurement) and `tools/research_scan.py` (public XDNA1/NPU work). They are **not enabled by default**. / Timers de usuario opcionales; **no vienen activados**.

```
mkdir -p ~/.config/systemd/user
cp tools/systemd/*.service tools/systemd/*.timer ~/.config/systemd/user/     # edit WorkingDirectory/ExecStart if the repo is not in ~/Documentos/Codigo/npu-lab
systemctl --user daemon-reload
systemctl --user enable --now npu-bench.timer npu-research.timer
systemctl --user list-timers 'npu-*'
```
- `npu-bench` exits 3 and records nothing when the server or another benchmark holds the NPU, and exits 2 on a regression (both count as success for systemd; read `bench/history.jsonl`). / sale con 3 si la NPU está ocupada y con 2 si hay regresión.
- `npu-research` only sends search keywords to GitHub; set `GITHUB_TOKEN` (or `~/.config/npu-lab/github_token`) to avoid the unauthenticated rate limit. / solo envía palabras clave; un token evita el límite de peticiones.
- Disable: `systemctl --user disable --now npu-bench.timer npu-research.timer`.

## Phone bridge / Puente al celular
`npu-telegram.service` runs `bridge/telegram_bridge.py` (see `bridge/README.md`; needs `~/.config/npu-lab/telegram.json`). `systemctl --user enable --now npu-telegram.service`.

## Sentinel / Centinela
`npu-sentinel.service` runs `src/sentinel.py` in the background (a desktop notice when a repo has uncommitted or unpushed work; one notice per change). Enable with `systemctl --user enable --now npu-sentinel.service`; change `--roots` to your code folders first. / Avisa de repos con trabajo sin commitear o sin subir; ajusta `--roots` antes de activarlo.
