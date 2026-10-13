"""Labelled queries against the LOCAL documentation index (~/.local/share/npu-lab/docs.pkl: man pages, tldr, Arch wiki), written by me (declared limit: the labels are my judgement).
POS: (query, gold titles): a result counts as a hit when the title of its chunk is one of the gold titles (case-insensitive). NEG: questions for which NO excerpt should be injected;
HARD negatives contain words that are also command or page names (typescript/typst, sort, find, time, head, watch...). Split ajuste/prueba (tuning/test) fixed by md5 of the query:
digest mod 10 < 6 -> 'ajuste', else 'prueba'. Parameters are chosen on 'ajuste' only; 'prueba' is evaluated once."""
import hashlib

POS = [
 ("how do I install an AUR package with paru", ["paru"]), ("update all packages on arch linux", ["pacman"]), ("what does the -r flag of cp do", ["cp"]),
 ("how to mount a usb drive from the terminal", ["mount"]), ("how do I list files sorted by size", ["ls"]), ("enable a systemd service at boot", ["systemctl"]),
 ("find a file by name recursively", ["find"]), ("how do I compress a folder with tar", ["tar"]), ("set up a cron job", ["crontab", "cron"]),
 ("how to check which process uses a port", ["ss", "lsof", "netstat"]), ("create a symbolic link", ["ln"]), ("search text recursively with grep", ["grep"]),
 ("copy a folder over ssh with rsync", ["rsync"]), ("change file permissions chmod 755", ["chmod"]), ("see the logs of a systemd service", ["journalctl"]),
 ("show disk usage of folders", ["du", "df"]), ("list block devices and partitions", ["lsblk"]), ("connect to wifi from the command line", ["nmcli"]),
 ("split the terminal into panes with tmux", ["tmux"]), ("download a file with wget", ["wget"]), ("make an HTTP request with curl and show the headers", ["curl"]),
 ("remove orphan packages in arch", ["pacman"]), ("what is the difference between kill and killall", ["kill", "killall"]), ("define an abbreviation in fish shell", ["fish", "abbr"]),
 ("extract a zip file", ["unzip"]), ("show my ip address", ["ip"]), ("how to switch git branch", ["git", "git-checkout", "git-switch"]),
 ("run a command on many files with xargs", ["xargs"]), ("change my default shell", ["chsh"]), ("see which processes use the most cpu", ["top", "htop", "ps"]),
 ("check how long the system took to boot", ["systemd-analyze"]), ("edit the sudoers file safely", ["visudo"]), ("show kernel messages", ["dmesg"]),
 ("run a program with lower priority using nice", ["nice", "renice"]), ("list installed packages with pacman", ["pacman"]), ("view the manual page of a command", ["man"]),
 ("run a command repeatedly every few seconds", ["watch"]), ("count lines words and bytes of a file", ["wc"]), ("sort the lines of a file", ["sort"]), ("show free memory", ["free"]),
 ("como instalo paquetes con pacman", ["pacman"]), ("cómo veo los servicios activos con systemctl", ["systemctl"]), ("como busco un archivo por nombre en todo el disco", ["find", "locate"]),
 ("cómo cambio los permisos de un archivo", ["chmod"]), ("cómo creo un enlace simbólico", ["ln"]), ("como comprimir una carpeta", ["tar", "zip", "gzip"]),
 ("cómo descargo un archivo de internet desde la terminal", ["wget", "curl"]), ("cómo veo el uso del disco", ["df", "du"]), ("como programo una tarea periódica", ["crontab", "cron"]),
 ("cómo cierro un proceso que no responde", ["kill"]), ("como copio archivos a otro equipo por red", ["scp", "rsync"]), ("cómo muestro mi dirección ip", ["ip"]),
 ("como veo los mensajes del kernel", ["dmesg"]), ("como divido la pantalla de la terminal en paneles", ["tmux", "screen"]), ("cómo ordeno la salida de un comando", ["sort"]),
 ("como busco texto dentro de archivos", ["grep"]), ("cómo cambio el nombre de un archivo", ["mv", "rename"]), ("como veo cuánta memoria ram libre tengo", ["free"]),
 ("cómo monto una memoria usb", ["mount"]), ("como actualizo todo el sistema en arch", ["pacman"]), ("como instalo un paquete del AUR", ["paru"]),
 ("como conecto al wifi desde la terminal", ["nmcli"]), ("cómo veo qué programa usa un puerto", ["ss", "lsof", "netstat"]), ("como cambio mi shell por defecto", ["chsh"]),
 ("como escribo un alias en fish", ["fish", "alias", "abbr"]), ("como veo los procesos en ejecución", ["ps", "top", "htop"]), ("como extraigo un archivo comprimido tar.gz", ["tar"]),
 ("como ejecuto un comando en cada archivo encontrado", ["find", "xargs"]), ("cómo cambio de rama en git", ["git", "git-checkout", "git-switch"]), ("como veo los logs de un servicio", ["journalctl"]),
 ("cómo cuento las líneas de un archivo", ["wc"]), ("como edito el archivo sudoers", ["visudo"]), ("como sincronizo dos carpetas", ["rsync"]),
 ("como veo el tiempo de arranque del sistema", ["systemd-analyze"]), ("como ejecuto un comando cada pocos segundos", ["watch"]), ("cómo muestro las particiones de mis discos", ["lsblk", "fdisk", "parted"]),
 ("como descomprimo un archivo zip", ["unzip"]), ("como veo el contenido de un archivo por páginas", ["less", "more"]), ("como reinicio un servicio systemd", ["systemctl"]),
]
NEG_HARD = [
 "how do I make a class in TypeScript", "explain list comprehension in python", "what is a monad in functional programming", "write a function to sort a list in javascript",
 "how to find the maximum of an array in java", "what is the time complexity of quicksort", "who is the current head of state of France", "explain the join operation in SQL",
 "what does the yes keyword mean in YAML", "how do I watch a movie online", "tell me about the history of the calendar", "what is the difference between a link and a hyperlink in html",
 "write a poem about the sea", "how does a hash table work", "explain how recursion works", "what is a pipe in electronics",
 "cómo hago una clase en TypeScript", "ordena estas palabras alfabéticamente por favor", "cuál es el tiempo en Madrid hoy", "encuentra el máximo común divisor de 12 y 18",
 "cómo cierro sesión en Instagram", "explícame qué es la programación orientada a objetos", "qué hace la palabra clave yield en Python", "quién es el jefe de estado de México",
]
NEG_EASY = [
 "what is the capital of France", "write a haiku about autumn", "how many days are in a leap year", "translate good morning to Spanish", "what is the boiling point of water",
 "who wrote Don Quixote", "give me a recipe for pancakes", "what is the square root of 144",
 "cuál es la capital de Francia", "dame una receta de arepas", "qué significa la palabra efímera", "cómo se calcula el interés compuesto",
 "quién ganó el mundial de 2022", "cuántos días tiene un año bisiesto", "escribe un correo para pedir vacaciones", "cuál es la raíz cuadrada de 144",
]
NEG = [(q, True) for q in NEG_HARD] + [(q, False) for q in NEG_EASY]


def split_of(q):
    return "ajuste" if int(hashlib.md5(q.encode()).hexdigest(), 16) % 10 < 6 else "prueba"


def bank():
    """-> list of dicts(query, kind 'pos'|'neg', gold, hard, split, lang)."""
    out = []
    for q, gold in POS:
        out.append(dict(query=q, kind="pos", gold=[g.lower() for g in gold], hard=False, split=split_of(q), lang="es" if any(c in q for c in "áéíóúñ") or q.split()[0] in ("como", "cómo") else "en"))
    for q, hard in NEG:
        out.append(dict(query=q, kind="neg", gold=[], hard=hard, split=split_of(q), lang="es" if any(c in q for c in "áéíóúñ") or q.split()[0] in ("cómo", "cuál", "qué", "quién", "dame", "ordena", "encuentra", "explícame", "escribe", "cuántos") else "en"))
    return out


if __name__ == "__main__":
    import collections
    b = bank(); print(len(b), collections.Counter((x["kind"], x["split"]) for x in b))
