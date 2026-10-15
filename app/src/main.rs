//! npu-lab desktop window (Tauri). Starts the local engine server (src/serve.py) exactly like `launcher/` does and shows its web UI in a native
//! window instead of the browser. Closing the window stops the server, so the NPU is released.
//! The server gets its access token in its environment and the window opens a one-time login code, never a long-lived secret on a command line.
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use tauri::menu::{Menu, MenuItem};
use tauri::tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};
use tauri::{Manager, RunEvent, WebviewUrl, WebviewWindowBuilder, WindowEvent};

fn rand_hex(bytes: usize) -> String {
    let mut b = vec![0u8; bytes];
    getrandom::fill(&mut b).expect("OS random generator unavailable");
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Walk up from `start` until a folder containing src/serve.py is found.
fn find_root(start: &Path) -> Option<PathBuf> {
    start.ancestors().find(|d| d.join("src").join("serve.py").is_file()).map(Path::to_path_buf)
}

fn find_python(root: &Path) -> String {
    let p = root.join(".venv/bin/python");
    if p.is_file() { p.to_string_lossy().into_owned() } else { "python3".into() }
}

/// Pull `"port": N` out of the one JSON line the server prints.
fn parse_port(line: &str) -> Option<u16> {
    let rest = line.split("\"port\"").nth(1)?;
    rest.trim_start_matches(|c: char| c == ':' || c.is_whitespace())
        .chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()
}

/// The access token shared with the phone bridge and editors: ~/.config/npu-lab/token (same file ./npu.sh uses), created with mode 600 if missing.
fn token_from_file() -> Option<String> {
    use std::os::unix::fs::OpenOptionsExt;
    let dir = PathBuf::from(std::env::var_os("HOME")?).join(".config/npu-lab");
    let f = dir.join("token");
    if let Ok(t) = std::fs::read_to_string(&f) {
        let t = t.trim().to_string();
        if t.len() >= 32 { return Some(t); }
    }
    std::fs::create_dir_all(&dir).ok()?;
    let t = rand_hex(32);
    let mut o = std::fs::OpenOptions::new();
    o.write(true).create(true).truncate(true).mode(0o600);
    o.open(&f).ok()?.write_all(t.as_bytes()).ok()?;
    Some(t)
}

/// 8765 (the port editors and the bridge expect) when it is free, otherwise a random one.
fn fixed_port_free() -> bool {
    std::net::TcpListener::bind(("127.0.0.1", 8765)).is_ok()
}

struct Engine { child: Child }

fn start_engine(root: &Path) -> Result<(Engine, u16, String), String> {
    let token = std::env::var("NPU_LAB_TOKEN").ok().filter(|t| t.len() >= 32).or_else(token_from_file).unwrap_or_else(|| rand_hex(32));
    let code = rand_hex(32);
    let mut cmd = Command::new(find_python(root));
    if std::env::var_os("NPU_LAB_PORT").is_none() && fixed_port_free() { cmd.env("NPU_LAB_PORT", "8765"); }
    let mut child = cmd
        .args(["-u", "src/serve.py"])
        .current_dir(root)
        .env("NPU_LAB_TOKEN", &token)
        .env("NPU_LAB_CODE", &code)
        .env("NPU_LAB_WATCH_STDIN", "1")
        .stdin(Stdio::piped()) // kept open: if this app dies, the pipe closes and the server exits by itself
        .stdout(Stdio::piped())
        .spawn()
        .map_err(|e| format!("could not start the Python engine server: {e}"))?;
    let _keep = child.stdin.take().map(|s| Box::leak(Box::new(s)));
    let stdout = child.stdout.take().ok_or("no stdout")?;
    let mut line = String::new();
    if BufReader::new(stdout).read_line(&mut line).unwrap_or(0) == 0 {
        let _ = child.wait();
        return Err("the engine server exited before it was ready (run `python src/serve.py` to see why)".into());
    }
    match parse_port(&line) {
        Some(port) => Ok((Engine { child }, port, code)),
        None => { let _ = child.kill(); Err("unexpected server output".into()) }
    }
}

fn main() {
    let root = std::env::var_os("NPU_LAB_ROOT").map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok().and_then(|d| find_root(&d)))
        .or_else(|| std::env::current_exe().ok().and_then(|e| find_root(&e)))
        .unwrap_or_else(|| { eprintln!("could not find the npu-lab folder (src/serve.py). Set NPU_LAB_ROOT."); std::process::exit(1) });
    let hidden = std::env::args().any(|a| a == "--hidden");       // start in the tray only (used by the autostart entry)
    let engine: Arc<Mutex<Option<Engine>>> = Arc::new(Mutex::new(None));
    let engine_setup = Arc::clone(&engine);

    let app = tauri::Builder::default()
        .setup(move |app| {
            // the start page shows while the engine boots (it can take a few seconds)
            let win = WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                .title("npu-lab").inner_size(1180.0, 860.0).min_inner_size(380.0, 560.0).visible(!hidden).build()?;
            // tray: left click shows or hides the window, the menu has Show and Quit; closing the window only hides it (the server keeps running)
            let show = MenuItem::with_id(app, "show", "Show npu-lab", true, None::<&str>)?;
            let quit = MenuItem::with_id(app, "quit", "Quit (frees the NPU)", true, None::<&str>)?;
            let menu = Menu::with_items(app, &[&show, &quit])?;
            let mut tray = TrayIconBuilder::new().tooltip("npu-lab").menu(&menu).show_menu_on_left_click(false)
                .on_menu_event(|app, ev| match ev.id.as_ref() {
                    "show" => { if let Some(w) = app.get_webview_window("main") { let _ = w.show(); let _ = w.set_focus(); } }
                    "quit" => app.exit(0),
                    _ => {}
                })
                .on_tray_icon_event(|tray, ev| {
                    if let TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } = ev {
                        if let Some(w) = tray.app_handle().get_webview_window("main") {
                            if w.is_visible().unwrap_or(false) { let _ = w.hide(); } else { let _ = w.show(); let _ = w.set_focus(); }
                        }
                    }
                });
            if let Some(icon) = app.default_window_icon() { tray = tray.icon(icon.clone()); }
            tray.build(app)?;
            let w2 = win.clone();
            win.on_window_event(move |ev| if let WindowEvent::CloseRequested { api, .. } = ev { api.prevent_close(); let _ = w2.hide(); });
            let root = root.clone();
            let handle = app.handle().clone();
            std::thread::spawn(move || match start_engine(&root) {
                Ok((eng, port, code)) => {
                    *engine_setup.lock().unwrap() = Some(eng);
                    let url = format!("http://127.0.0.1:{port}/#code={code}");
                    let _ = handle.get_webview_window("main").map(|w| w.navigate(url.parse().unwrap()));
                }
                Err(e) => {
                    eprintln!("{e}");
                    let _ = win.eval(&format!("document.querySelector('p').textContent = {:?}", e));
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the app");

    app.run(move |_handle, event| {
        if let RunEvent::Exit = event {            // window closed or Ctrl+C: stop the server so the NPU is free again
            if let Some(mut e) = engine.lock().unwrap().take() { let _ = e.child.kill(); let _ = e.child.wait(); }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn port_is_parsed() {
        assert_eq!(parse_port("{\"port\": 41231}"), Some(41231));
        assert_eq!(parse_port("{\"port\":8080,\"token\":\"x\"}"), Some(8080));
        assert_eq!(parse_port("garbage"), None);
    }
    #[test]
    fn random_hex_has_right_shape() {
        assert_eq!(rand_hex(32).len(), 64);
        assert_ne!(rand_hex(32), rand_hex(32));
    }
}
