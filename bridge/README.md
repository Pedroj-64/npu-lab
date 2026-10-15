# Phone bridge (Telegram) / Puente al celular

A Telegram bot that answers only the user ids you list and talks to the npu-lab server on your PC. Standard library only, no extra dependency. / Un bot de Telegram que solo responde a los IDs que listes y habla con el servidor npu-lab de tu PC.

## Setup / Puesta en marcha
1. Create a bot with @BotFather (`/newbot`) and get your numeric id from @userinfobot. / Crea el bot con @BotFather y tu ID numérico con @userinfobot.
2. Write `~/.config/npu-lab/telegram.json` with mode 600 (never inside the repo): `{"token": "123456:ABC...", "allowed_user_ids": [123456789]}`
3. Open your bot in Telegram and press Start (a bot cannot write first). / Abre tu bot y pulsa Start.
4. Run it: `python bridge/telegram_bridge.py`, or as a service: `systemctl --user enable --now npu-telegram.service` (unit in `tools/systemd/`).
5. The server must be running with the shared token (`~/.config/npu-lab/token`) on port 8765: the desktop app and `./npu.sh` both do that. / El servidor debe correr con el token compartido en el puerto 8765.

## Commands / Comandos
`/status` `/repos` `/summary` `/commit NAME` (secret check + draft message) `/explain COMMAND` `/log UNIT` `/bench` `/load [qwen3|qwen3s]` `/unload`; any other text goes to the local model (last 6 messages kept per user).

## Safety / Seguridad
- Users not in `allowed_user_ids` get no answer at all. / Quien no esté en la lista no recibe respuesta.
- No command reads a file or runs a shell command; the bot only handles short answers and counts, never file contents, and secret values found by `/commit` are never shown. / Ningún comando lee archivos ni ejecuta comandos; los valores de secretos nunca se muestran.
- The bot token and the user ids live only in `~/.config/npu-lab/telegram.json`. If the token leaks, revoke it with @BotFather (`/revoke`). / Si el token se filtra, revócalo con @BotFather.
- **Privacy:** messages and the model's answers pass through Telegram's servers (bots are not end-to-end encrypted). Do not ask for anything you would not send to Telegram. / Los mensajes y respuestas pasan por los servidores de Telegram.
