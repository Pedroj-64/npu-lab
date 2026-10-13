"""Hand-labelled questions for two decision tasks of src/decide.py, Spanish and English, written by the author of the repo's evaluation (NOT measured on
users: the labels are my judgement, so treat accuracy on them as optimistic and check the ones near the boundary).
route : which source answers the question: NONE (answer directly), DOCS (a terminal command / installed tool / Arch), WEB (time-sensitive), HISTORY (about the chat so far).
mixer : which model should answer: fast (short facts, translation, small talk), medium (explanations, short code, questions on a document), slow (multi-file code changes, refactors,
        long or multi-step reasoning).
usage: .venv/bin/python bench/decide_data_text.py   -> bench/decide_data/route.jsonl, mixer.jsonl   (split by a hash of the text: ~60 / 20 / 20 %, fixed)"""
import hashlib, json, os

ROUTE = {
    "NONE": [
        "how do I make a class in TypeScript", "explain recursion", "what is the capital of France", "write a haiku about the sea", "diferencia entre TCP y UDP", "cómo funciona un hash table",
        "what is a monad in simple terms", "write a python function that reverses a string", "explícame qué es la programación orientada a objetos", "translate 'good morning' to Spanish",
        "what is the difference between a process and a thread", "give me a recipe idea for dinner with rice and beans", "qué es la fotosíntesis", "how does binary search work",
        "write a short poem about autumn", "cuál es la raíz cuadrada de 144", "what does the SOLID principle mean", "explain big O notation", "escribe un correo formal para pedir vacaciones",
        "what is the Pythagorean theorem", "how do I center a div in CSS", "dame tres ideas para un nombre de empresa de café", "explain what an API is to a ten year old", "what are the planets of the solar system",
        "cómo se calcula el interés compuesto", "summarize the plot of Romeo and Juliet", "what is the difference between let and const in JavaScript", "cuántos días tiene un año bisiesto",
        "write a SQL query that counts rows per country", "who wrote Don Quixote", "explain how a neural network learns", "qué significa la palabra efímero", "how do I reverse a linked list",
        "what is the boiling point of water in Fahrenheit", "ayúdame a escribir una presentación personal para LinkedIn", "difference between stack and heap memory", "how many bits are in a byte",
        "inventa un chiste sobre programadores", "what is dependency injection", "explica la diferencia entre ser y estar", "how does public key cryptography work", "what is the time complexity of quicksort",
        "write a regex that matches an email address", "por qué el cielo es azul", "give me a workout plan for beginners", "what is a closure in programming", "cómo puedo mejorar mi redacción",
        "explain the CAP theorem", "what is the meaning of life according to stoicism", "convierte 5 kilómetros a millas", "how do I write a unit test in pytest",
    ],
    "DOCS": [
        "how do I install an AUR package", "como listo los servicios activos con systemctl", "what does the -r flag of cp do", "cómo cambio los permisos de un archivo", "how to mount a usb drive from the terminal",
        "update all packages on arch", "how do I use paru to search for a package", "qué hace el comando rsync con --delete", "how do I list files sorted by size in the terminal", "como instalo paquetes con pacman",
        "what does journalctl -xe show", "cómo veo el uso de disco con df", "how do I find a file by name with find", "como comprimir una carpeta con tar", "what is the difference between pacman -S and pacman -Syu",
        "how to check which process uses a port with ss", "cómo creo un enlace simbólico con ln", "how do I use grep to search recursively", "cómo funciona el comando chmod 755", "how do I set a cron job",
        "como cambio el nombre de varios archivos desde la terminal", "how do I see the logs of a systemd service", "what does the tldr page of curl say", "cómo muestro mi dirección IP desde la terminal",
        "how to remove orphan packages in arch", "como edito el archivo fstab", "how do I use xargs with find", "qué opciones tiene el comando ls para mostrar archivos ocultos", "how do I enable a service at boot with systemctl",
        "como veo la temperatura de la CPU desde la terminal", "how do I use fish shell abbreviations", "cómo descargo un archivo con wget", "what does the man page of ssh say about -L", "como cierro un proceso con kill",
        "how do I list installed packages with pacman", "cómo cambio de rama en git desde la terminal", "how to use rsync to copy a folder over ssh", "qué hace el flag -v de docker run", "how do I see my disk partitions with lsblk",
        "como uso tmux para dividir la pantalla", "how do I extract a zip file from the command line", "cómo busco el paquete que contiene un archivo en arch", "how do I change my default shell with chsh",
        "como configuro sudo sin contraseña para un comando", "how do I check systemd boot time with systemd-analyze", "cómo veo los procesos con htop o ps", "how to mount an iso file", "como pruebo la velocidad del disco con dd",
        "how do I use nmcli to connect to wifi", "cómo escribo un alias en fish",
    ],
    "WEB": [
        "who won the match yesterday", "latest stable version of the linux kernel", "precio del dólar hoy", "what is the weather in Madrid", "noticias de tecnología", "what happened last week in the stock market",
        "quién ganó las elecciones en 2026", "what is the current price of bitcoin", "últimas noticias sobre inteligencia artificial", "who is the current president of Mexico", "what time does the sun set today in Lima",
        "resultado del partido de anoche", "latest iPhone release date", "qué películas hay en cartelera esta semana", "is there a new version of Python out", "cuál es el clima mañana en Bogotá",
        "what are the top headlines today", "cotización del euro en este momento", "who won the Champions League this year", "when is the next solar eclipse", "qué pasó hoy en la bolsa",
        "latest news about the Mars mission", "cuánto cuesta una RTX 5070 ahora", "current exchange rate USD to COP", "últimas versiones de Node.js", "what is trending on the internet this week",
        "quién es el actual campeón del mundo de ajedrez", "when does the new season of the series start", "precio de la gasolina esta semana", "what was announced at the latest Apple event",
        "estado del tráfico en la autopista ahora", "latest security vulnerability in OpenSSL", "cuándo sale el próximo álbum de ese artista", "what are the release notes of the newest Firefox",
        "who scored in the match today", "tasa de inflación del último mes", "is the service of GitHub down right now", "qué temperatura hace ahora en Madrid", "newest model released by OpenAI this month",
        "resultados de la liga de ayer", "latest Linux Mint version", "cuál es el precio del oro hoy", "what happened in the news this morning", "when is the next Arch Linux ISO release",
        "últimos precios de portátiles con Ryzen", "who is leading the Formula 1 championship now", "noticias de ayer sobre el clima", "current population of Mexico City estimate", "qué día es el próximo festivo",
        "current stock price of AMD", "mejores ofertas de hoy en tecnología",
    ],
    "HISTORY": [
        "what did we talk about", "repite lo último que dijiste", "summarise the above", "y en Python?", "what was my first question", "translate that to English", "and in Rust?", "resume lo anterior en dos frases",
        "can you repeat that", "qué me dijiste antes sobre el puerto", "what did you say about the second option", "y con TypeScript?", "make it shorter", "hazlo más formal", "explain your last answer again",
        "what was the name I told you earlier", "de qué hablábamos hace un momento", "give me the same but in Spanish", "now do it for the other file", "continúa", "and what about the third one?",
        "can you rewrite that last paragraph", "cuál fue mi pregunta anterior", "go on", "summarize our conversation so far", "pero y si no funciona?", "what did I ask you before",
        "dame el código de nuevo", "por qué dijiste eso", "add a docstring to the function you just wrote", "y el segundo punto?", "repeat the list from before", "puedes explicarlo más despacio?",
        "which one did you recommend", "cambia el nombre de la variable que usaste", "tell me again the steps you gave", "ahora hazlo en inglés", "and then?", "qué número te di antes",
        "make the previous example longer", "resúmelo en una línea", "what was the third thing you listed", "otra vez pero con comentarios", "as you said, how do I do it", "traduce eso al francés",
        "can you shorten your previous answer", "y por qué?", "show me your last code again", "vuelve a la primera respuesta", "more details on that please", "dímelo otra vez",
    ],
}

MIXER = {
    "fast": [
        "hola, cómo estás", "what is the capital of Japan", "translate 'thank you very much' to French", "cuánto es 17 por 23", "tell me a joke", "who wrote Hamlet", "qué hora es en Tokio si aquí son las 3",
        "spell 'necessary'", "give me a synonym for happy", "cómo se dice 'mesa' en inglés", "what year did World War II end", "dame una palabra que rime con luz", "is a tomato a fruit", "convierte 10 millas a kilómetros",
        "good morning!", "what does CPU stand for", "dime un dato curioso", "name three colors", "cuál es el plural de lápiz", "how many days are in March", "gracias", "what is 2 to the power of 10",
        "cuál es la capital de Perú", "define serendipity in one line", "traduce 'where is the bathroom' al español", "who painted the Mona Lisa", "qué significa HTTP", "what is the opposite of cold",
        "dime el nombre de un río de Europa", "how do you say 'goodbye' in German", "cuántos metros tiene un kilómetro", "what is the largest planet", "recomiéndame un nombre para mi gato",
        "is Python interpreted or compiled, one word", "qué día de la semana fue el 1 de enero de 2000", "name a programming language that starts with R", "buenas noches", "what is H2O",
        "cuántas letras tiene el abecedario", "who discovered penicillin", "dame un emoji de felicidad", "what's the plural of mouse", "resume en una frase: el gato duerme en el sofá todo el día",
        "what language do they speak in Brazil", "cuánto es el 15% de 200", "capital of Australia?", "define 'latency' briefly", "di hola en tres idiomas", "what is the chemical symbol of gold",
        "convert 100 Fahrenheit to Celsius", "cuál es el color del cielo", "who invented the telephone", "tell me the time zone of Madrid", "una palabra en inglés para 'rápido'", "what does RAM mean",
        "quién escribió Cien años de soledad", "give me a one-line motivational quote", "how many continents are there", "cuántos segundos tiene una hora", "name the days of the week in Spanish",
        "what is the capital of Chile", "cuántas patas tiene una araña", "say hello in Italian", "qué es un byte",
        "is 91 a prime number", "what is the abbreviation of kilogram", "saluda a mi amigo Pedro", "what is the speed of light, roughly", "cuál es el símbolo del euro", "simple word for 'utilize'",
        "answer yes or no: is the sun a star", "traduce 'I am hungry' al español", "who was the first person on the moon", "cuántos lados tiene un hexágono",
    ],
    "medium": [
        "explain how a hash table handles collisions", "escribe una función en python que cuente palabras en un texto", "what is the difference between TCP and UDP and when do I use each", "resume este párrafo y dime las ideas principales",
        "write a bash script that backs up my home folder to an external disk", "explícame cómo funciona git rebase con un ejemplo", "what are the pros and cons of microservices", "dame un plan de estudio de 4 semanas para aprender SQL",
        "write a python class for a bank account with deposit and withdraw", "explain what this error means: KeyError in a dictionary lookup", "cómo configuro un servicio de systemd para que arranque mi script", "compare REST and GraphQL",
        "write unit tests for a function that parses dates", "qué diferencia hay entre una lista y una tupla en Python y cuándo uso cada una", "how do I read a CSV file with pandas and compute the mean per group",
        "explícame el algoritmo de Dijkstra paso a paso", "write a regex to validate a spanish phone number and explain it", "what are good practices for naming variables", "dame un script de fish que renombre archivos .jpeg a .jpg",
        "explain how async await works in JavaScript", "escribe una consulta SQL con join entre pedidos y clientes", "how does garbage collection work in Java", "responde con base en el documento: qué dice la sección 3 sobre plazos",
        "write a function that merges two sorted lists", "explain the difference between processes and threads with an example in Python", "cómo hago un respaldo incremental con rsync", "write a README for a small python tool",
        "what is a race condition and how do I avoid it", "traduce este párrafo técnico al inglés y mantén los términos", "describe how HTTPS establishes a secure connection", "escribe un decorador en python que mida el tiempo de una función",
        "summarize the main arguments of this article", "cómo puedo optimizar una consulta SQL lenta", "explain how to set up a python virtual environment and why", "crea un docker-compose para una app flask y postgres",
        "what is the difference between a mutex and a semaphore", "dame ideas de arquitectura para una app de notas offline", "write a function to check if a string is a palindrome ignoring punctuation", "explícame cómo funciona el cifrado AES a grandes rasgos",
        "review this short function and suggest improvements", "cómo hago paginación en una API REST", "write a python script that downloads a file and shows a progress bar", "explain the observer pattern with a small example",
        "qué hace este comando de awk y cómo lo mejoro", "outline a blog post about local LLMs on NPUs", "write a binary search in C and explain the loop invariant", "dame una lista de pasos para depurar un memory leak",
        "compare two sorting algorithms for nearly sorted data", "escribe un test de integración para un endpoint de login", "how do I structure a small flask project", "explica qué es un índice en una base de datos y cuándo perjudica",
        "write a git commit message from this diff summary", "convierte este json a una tabla markdown", "explain Big-O of nested loops over two lists", "cómo hago que mi script de bash maneje errores correctamente",
        "write a function that flattens a nested list", "describe the differences between Qwen and Llama architectures at a high level", "responde usando las notas: qué decidimos sobre el formato del informe", "give me a checklist for a code review",
        "cómo implemento caché con TTL en Python", "explain what CORS is and how to fix a CORS error", "write a short script to rename files by date", "qué es la inyección de dependencias y cómo se prueba",
        "help me write a cover letter for a junior developer job", "explica paso a paso cómo resolver una ecuación cuadrática y programa la solución",
    ],
    "slow": [
        "refactor the project so that all database access goes through one module and update every caller", "implementa un sistema de autenticación completo con registro, login, tokens y pruebas en todo el proyecto",
        "migrate this codebase from unittest to pytest across all test files and fix the failures", "rediseña la arquitectura de este módulo para separar la lógica de la interfaz y actualiza todos los archivos afectados",
        "write a complete command line tool with subcommands, config file, logging and tests", "encuentra y corrige el bug que hace que el total del pedido sea incorrecto en varios archivos, y añade pruebas de regresión",
        "add pagination, filtering and sorting to every list endpoint of the api and update the docs and tests", "convierte todo el proyecto a async y asegúrate de que las pruebas siguen pasando",
        "analyze this stack trace across the three modules involved and propose a fix with the minimal diff", "implementa un parser para este formato de archivo con manejo de errores, casos límite y una batería de pruebas",
        "design and implement a plugin system for this application, including loading, versioning and an example plugin", "optimiza el rendimiento del programa: perfílalo, identifica los cuellos de botella y reescribe las partes críticas",
        "rename this class everywhere in the repository, update imports, docs and tests, and check nothing else breaks", "escribe un servidor web completo con rutas, base de datos y validación a partir de esta especificación",
        "reason step by step through the concurrency bug in this queue implementation and fix it", "añade soporte multiidioma a toda la aplicación extrayendo todas las cadenas y creando los archivos de traducción",
        "implement an LRU cache with thread safety, write property based tests and benchmark it", "diseña un esquema de base de datos para esta aplicación, escribe las migraciones y adapta los modelos",
        "split this 800-line file into cohesive modules without changing behaviour and keep the tests green", "prueba que este algoritmo es correcto y analiza su complejidad en el peor caso con una demostración detallada",
        "port this script from Python 2 to Python 3 across the whole project and fix every incompatibility", "crea un sistema de caché distribuido sencillo con invalidación y escribe la documentación técnica",
        "plan and carry out a multi-step data migration with rollback, validation and a report", "revisa todo el repositorio en busca de vulnerabilidades de inyección y corrige cada hallazgo con pruebas",
        "build a small interpreter for an arithmetic language with variables, functions and error messages", "reescribe el módulo de pagos para soportar varias monedas y actualiza los informes y las pruebas",
        "debug why the tests fail only in CI: read the logs, the config and the code and propose a fix", "implementa una máquina de estados para este flujo de pedidos y refactoriza el código que la usa",
        "add type hints to the whole package and fix all the mypy errors without changing behaviour", "escribe un generador de informes en PDF con plantillas, gráficos y pruebas automatizadas",
        "implement undo/redo for the editor across the model, the commands and the ui", "diseña e implementa un algoritmo de planificación de tareas con dependencias y detecta ciclos",
        "extract the common code of these five scripts into a library and update each script to use it", "analiza el código de este proyecto, haz un mapa de dependencias y propón un plan de refactor por fases",
        "write a full test suite for the inventory module covering edge cases and then fix the bugs it reveals", "migra la configuración de variables de entorno a un archivo yaml con validación en todo el proyecto",
        "implement oauth login flow end to end with the tests and the documentation", "refactoriza esta función de 200 líneas en piezas pequeñas, con pruebas antes y después para asegurar que no cambia",
        "create a CLI that scans a directory tree, builds an index in sqlite and supports fuzzy search with tests", "resuelve este problema de optimización combinatoria y justifica la solución paso a paso",
        "make the whole application accessible: audit every page, fix the issues and add automated checks", "reemplaza la librería antigua por la nueva en todos los módulos y adapta cada llamada",
        "build a rate limiter middleware, integrate it in every route and write tests for bursts", "diseña el sistema de logs del proyecto, aplícalo en todos los módulos y crea las pruebas",
        "trace the data flow from the request to the database in this codebase and fix the inconsistent validation", "implementa paginación por cursor en la api, el cliente y las pruebas, y documenta el cambio",
        "write a complete chess move validator with check detection and tests", "crea una herramienta que compare dos árboles de directorios y genere un script de sincronización seguro",
        "make the checkout flow handle partial payments: change the models, the api, the emails and the tests", "añade exportación a CSV y Excel en todos los informes con pruebas por cada uno",
        "investigate why memory grows over time in the worker, find the leak and fix it", "implementa búsqueda con resaltado en la interfaz, el servidor y el índice, y prueba todo",
        "convert the project from callbacks to promises everywhere and verify with the existing tests", "optimiza el uso de memoria de la aplicación midiendo antes y después en varios módulos",
        "implement a diff and patch utility for text files with tests for the edge cases", "revisa el diseño de esta API, enumera los problemas y reescribe los endpoints afectados con pruebas",
        "add retries with backoff to every network call in the project and test the failure paths", "genera un sitio estático completo a partir de estos archivos markdown con plantillas y navegación",
        "debug the intermittent failure in the scheduler by reproducing it, finding the race and patching it", "diseña y programa un sistema de permisos por roles que afecte a modelos, vistas y pruebas",
        "convert this repository's build system to a new tool and keep every target working", "escribe un compilador sencillo de expresiones a bytecode con una máquina virtual y pruebas",
    ],
}


def splits(qs):
    """Stratified and fixed: within one label, order by hash of the text; first 20 % test, next 20 % val, the rest train."""
    order = sorted(qs, key=lambda q: hashlib.sha256(q.encode()).hexdigest()); n = len(order)
    return {q: ("test" if i < round(n * .2) else "val" if i < round(n * .4) else "train") for i, q in enumerate(order)}


def write(name, data):
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "decide_data"); os.makedirs(out, exist_ok=True)
    rows, seen = [], set()
    for label, qs in data.items():
        sp = splits(qs)
        for q in qs:
            assert q not in seen, f"duplicate: {q}"; seen.add(q)
            es = any(c in q for c in "áéíóúñ¿¡") or q.split()[0].lower() in {"qué", "cómo", "como", "cuál", "cuántos", "dame", "explica", "escribe", "hola", "y", "pero", "dime", "traduce", "resume", "hazlo", "puedes", "por", "de", "convierte", "crea", "implementa", "diseña", "refactoriza", "añade", "analiza", "migra", "revisa", "rediseña", "reescribe", "encuentra", "optimiza", "genera", "convert", "gracias", "buenas", "saluda", "quién", "cuánto", "recomiéndame", "inventa", "ayúdame", "continúa", "vuelve", "otra", "cambia", "repite", "dímelo", "resuelve", "reemplaza", "prueba", "responde", "unas"}
            rows.append(dict(id=f"{name}-{len(rows)}", task=name, split=sp[q], lang="es" if es else "en", state=q, label=label))
    with open(os.path.join(out, f"{name}.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return rows


if __name__ == "__main__":
    for name, data in (("route", ROUTE), ("mixer", MIXER)):
        rows = write(name, data)
        by = {}
        for r in rows:
            by.setdefault((r["split"], r["label"]), 0); by[(r["split"], r["label"])] += 1
        print(name, len(rows), "rows;", {f"{s}/{l}": n for (s, l), n in sorted(by.items())}, "es:", sum(r["lang"] == "es" for r in rows))
