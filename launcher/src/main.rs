//! `npu`: starts the local engine server (src/serve.py) and opens the web UI.
//!
//! The server is never given a long-lived secret on a command line: the access token goes in the child's
//! environment, and the browser URL carries a one-time login code that the server swaps for the token once.
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};

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
    for rel in [".venv/bin/python", ".venv/Scripts/python.exe"] {
        let p = root.join(rel);
        if p.is_file() {
            return p.to_string_lossy().into_owned();
        }
    }
    if cfg!(windows) { "python".into() } else { "python3".into() }
}

fn open_browser(url: &str) -> bool {
    let (prog, args): (&str, Vec<&str>) = if cfg!(target_os = "windows") {
        ("cmd", vec!["/C", "start", "", url])
    } else if cfg!(target_os = "macos") {
        ("open", vec![url])
    } else {
        ("xdg-open", vec![url])
    };
    Command::new(prog).args(args).stdout(Stdio::null()).stderr(Stdio::null()).spawn().is_ok()
}

/// Pull `"port": N` out of the one JSON line the server prints (no JSON dependency needed).
fn parse_port(line: &str) -> Option<u16> {
    let rest = line.split("\"port\"").nth(1)?;
    rest.trim_start_matches(|c: char| c == ':' || c.is_whitespace())
        .chars().take_while(|c| c.is_ascii_digit()).collect::<String>().parse().ok()
}

fn main() {
    let mut no_browser = false;
    let mut root_arg: Option<PathBuf> = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--no-browser" => no_browser = true,
            "--root" => root_arg = args.next().map(PathBuf::from),
            "-h" | "--help" => {
                println!("npu [--no-browser] [--root DIR]\n\nStarts the local engine server and opens the web UI.\n\
                          Run it from inside the npu-lab folder, or pass --root (or set NPU_LAB_ROOT).");
                return;
            }
            "-V" | "--version" => { println!("npu {}", env!("CARGO_PKG_VERSION")); return; }
            other => { eprintln!("unknown option: {other} (try --help)"); std::process::exit(2); }
        }
    }
    let root = root_arg
        .or_else(|| std::env::var_os("NPU_LAB_ROOT").map(PathBuf::from))
        .or_else(|| std::env::current_dir().ok().and_then(|d| find_root(&d)))
        .or_else(|| std::env::current_exe().ok().and_then(|e| find_root(&e)))
        .unwrap_or_else(|| { eprintln!("could not find the npu-lab folder (src/serve.py). Use --root DIR."); std::process::exit(1) });

    let (token, code) = (rand_hex(32), rand_hex(32));
    let mut child: Child = Command::new(find_python(&root))
        .args(["-u", "src/serve.py"])
        .current_dir(&root)
        .env("NPU_LAB_TOKEN", &token)
        .env("NPU_LAB_CODE", &code)
        .stdout(Stdio::piped())
        .spawn()
        .unwrap_or_else(|e| { eprintln!("could not start the Python engine server: {e}"); std::process::exit(1) });

    let stdout = child.stdout.take().expect("piped stdout");
    let mut line = String::new();
    if BufReader::new(stdout).read_line(&mut line).unwrap_or(0) == 0 {
        eprintln!("the engine server exited before it was ready (run `python src/serve.py` to see why)");
        let _ = child.wait();
        std::process::exit(1);
    }
    let port = parse_port(&line).unwrap_or_else(|| { eprintln!("unexpected server output"); let _ = child.kill(); std::process::exit(1) });

    let url = format!("http://127.0.0.1:{port}/#code={code}");
    println!("npu-lab is running. Open this link (it works once):\n\n  {url}\n\nPress Ctrl+C to stop.");
    if !no_browser && !open_browser(&url) {
        println!("(could not open a browser automatically; open the link yourself)");
    }

    // Ctrl+C: stop the server so the NPU is released, then exit.
    let child = Arc::new(Mutex::new(child));
    let c2 = Arc::clone(&child);
    ctrlc::set_handler(move || {
        if let Ok(mut c) = c2.lock() { let _ = c.kill(); let _ = c.wait(); }
        println!("\nstopped.");
        std::process::exit(0);
    }).expect("could not install the Ctrl+C handler");

    loop {
        std::thread::sleep(std::time::Duration::from_millis(500));
        if let Ok(mut c) = child.lock() {
            if let Ok(Some(status)) = c.try_wait() {
                eprintln!("the engine server stopped ({status})");
                std::process::exit(1);
            }
        }
    }
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
        let a = rand_hex(32);
        assert_eq!(a.len(), 64);
        assert_ne!(a, rand_hex(32));
    }
}
