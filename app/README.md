# npu-lab desktop window (Tauri)

Spanish below. A native window for the web UI: it starts the local engine server (`src/serve.py`) the same way `launcher/` does and shows the page in its own window instead of the browser. Closing the window stops the server and frees the NPU. No Electron: the page runs in the system WebKitGTK.

## Build and run
```
cd app && cargo build --release            # needs webkit2gtk-4.1, gtk3 (Arch: sudo pacman -S webkit2gtk-4.1 gtk3)
NPU_LAB_ROOT=$PWD/.. ./target/release/npu-lab-app
```
Menu entry: `cp npu-lab.desktop ~/.local/share/applications/` and copy `icons/icon.png` to `~/.local/share/icons/npu-lab.png` (edit the paths in the `Exec` line if the repo is elsewhere).

## Status
- Done: window, server start/stop, one-time login code (no long-lived secret on a command line), start page while the engine boots, menu entry.
- Tray icon (needs `libayatana-appindicator`, Arch: `sudo pacman -S libayatana-appindicator`): left click shows or hides the window, the menu has Show and "Quit (frees the NPU)". Closing the window only hides it; the server keeps running.
- Start with the session: copy `npu-lab-autostart.desktop` to `~/.config/autostart/npu-lab.desktop`. It starts with `--hidden` (tray only, no window). Remove that file to turn it off.
- Not done: packaging (AppImage/deb), which needs `tauri-cli`.

---

Ventana nativa para la interfaz web: arranca el servidor local (`src/serve.py`) igual que `launcher/` y muestra la página en su propia ventana en vez del navegador. Al cerrar la ventana se detiene el servidor y se libera la NPU. Sin Electron: usa el WebKitGTK del sistema.

## Compilar y ejecutar
```
cd app && cargo build --release            # necesita webkit2gtk-4.1 y gtk3
NPU_LAB_ROOT=$PWD/.. ./target/release/npu-lab-app
```
Entrada de menú: copia `npu-lab.desktop` a `~/.local/share/applications/` y `icons/icon.png` a `~/.local/share/icons/npu-lab.png`.

## Estado
- Hecho: ventana, arranque y parada del servidor, código de acceso de un solo uso, página de inicio mientras arranca el motor, entrada de menú.
- Icono de bandeja (necesita `libayatana-appindicator`): clic izquierdo muestra u oculta la ventana, el menú tiene Mostrar y «Quit (frees the NPU)». Cerrar la ventana solo la oculta; el servidor sigue.
- Arranque con la sesión: copia `npu-lab-autostart.desktop` a `~/.config/autostart/npu-lab.desktop`. Arranca con `--hidden` (solo bandeja, sin ventana). Borra ese archivo para quitarlo.
- Falta: empaquetado (AppImage/deb), que necesita `tauri-cli`.
