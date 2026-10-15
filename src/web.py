"""Web search for the chat (fresh information the model does not have).

Providers: duckduckgo (no key), wikipedia (no key), searxng (your own instance), brave (API key).
The server fetches; the model only ever sees short text excerpts. Everything from the web is untrusted data.

Safety: only http(s); every host is resolved and refused if it points to a private, loopback, link-local or reserved
address (this server must never become a way into your LAN); redirects are re-checked hop by hop; size/time capped.
"""
import datetime, html, ipaddress, json, re, socket, time, urllib.error, urllib.parse, urllib.request

UA = "Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0"
MAX_BYTES, TIMEOUT = 400_000, 12
CACHE_SECONDS = 900          # the same question within 15 minutes is answered from memory: no new request, no new page reads
_CACHE = {}


class WebError(Exception):
    pass


def check_url(url, allow_local=False):
    """Raise WebError unless `url` is http(s) and its host resolves only to public addresses."""
    u = urllib.parse.urlsplit(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise WebError("only http(s) URLs are allowed")
    try:
        infos = socket.getaddrinfo(u.hostname, u.port or (443 if u.scheme == "https" else 80), type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise WebError(f"cannot resolve {u.hostname}: {e}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if not allow_local and (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved
                                or ip.is_multicast or ip.is_unspecified):
            raise WebError(f"{u.hostname} resolves to a non-public address ({ip}); refused")
    return u


class _Guard(urllib.request.HTTPRedirectHandler):
    def __init__(self, allow_local):
        self.allow_local = allow_local

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        check_url(newurl, self.allow_local)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def http(url, data=None, headers=None, allow_local=False, max_bytes=MAX_BYTES):
    check_url(url, allow_local)
    req = urllib.request.Request(url, data=data, headers={"User-Agent": UA, **(headers or {})})
    op = urllib.request.build_opener(_Guard(allow_local))
    try:
        with op.open(req, timeout=TIMEOUT) as r:
            body = r.read(max_bytes + 1)[:max_bytes]
            return r.status, r.headers.get("Content-Type", ""), body
    except urllib.error.HTTPError as e:
        raise WebError(f"HTTP {e.code} from {urllib.parse.urlsplit(url).hostname}")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise WebError(f"request failed: {e}")


def _clean(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", "", s or ""))).strip()


def _day(s):
    """'2026-09-30T12:00:00Z' / '2026-09-30 12:00' -> '2026-09-30'; anything that is not a plausible date -> ''."""
    m = re.match(r"\s*(\d{4})-(\d{2})-(\d{2})", str(s or ""))
    try:
        return datetime.date(int(m[1]), int(m[2]), int(m[3])).isoformat() if m and 1990 <= int(m[1]) <= datetime.date.today().year + 1 else ""
    except ValueError:
        return ""


DATE_PATTERNS = (r'(?i)<meta[^>]+(?:property|name|itemprop)=["\'](?:article:published_time|og:published_time|datepublished|pubdate|date|dc\.date(?:\.issued)?|article:modified_time|og:updated_time)["\'][^>]*content=["\']([^"\']+)',
                 r'(?i)<meta[^>]+content=["\']([^"\']+)["\'][^>]*(?:property|name|itemprop)=["\'](?:article:published_time|og:published_time|datepublished|pubdate)["\']',
                 r'"datePublished"\s*:\s*"([^"]+)"', r'(?i)<time[^>]+datetime=["\']([^"\']+)')


def find_date(raw):
    """Publication date (YYYY-MM-DD) declared by a page: meta tags, JSON-LD or <time datetime>; '' when it declares none."""
    for pat in DATE_PATTERNS:
        for m in re.finditer(pat, raw):
            d = _day(m.group(1))
            if d:
                return d
    return ""


# ---- providers: each returns [{title, url, snippet, date?}]
def ddg(q, n, cfg):
    _, _, b = http("https://html.duckduckgo.com/html/", data=urllib.parse.urlencode({"q": q}).encode(),
                   headers={"Content-Type": "application/x-www-form-urlencoded"})
    page, out = b.decode("utf-8", "ignore"), []
    for m in re.finditer(r'<a[^>]+class="result__a"[^>]+href="([^"]+)"[^>]*>(.*?)</a>(.*?)(?=<a[^>]+class="result__a"|$)', page, re.S):
        href = html.unescape(m.group(1))
        if "uddg=" in href:
            href = urllib.parse.unquote(urllib.parse.parse_qs(urllib.parse.urlsplit(href).query).get("uddg", [""])[0])
        elif href.startswith("//"):
            href = "https:" + href
        sn = re.search(r'class="result__snippet"[^>]*>(.*?)</a>', m.group(3), re.S)
        if href.startswith("http"):
            out.append({"title": _clean(m.group(2)), "url": href, "snippet": _clean(sn.group(1)) if sn else ""})
        if len(out) >= n:
            break
    if not out:
        raise WebError("DuckDuckGo returned no results (it may be rate-limiting; try wikipedia or a SearXNG/Brave provider)")
    return out


def wikipedia(q, n, cfg):
    out = []
    for lang in dict.fromkeys([cfg.get("wiki_lang", "es"), "en"]):
        _, _, b = http(f"https://{lang}.wikipedia.org/w/api.php?" + urllib.parse.urlencode(
            {"action": "query", "list": "search", "srsearch": q, "srlimit": n, "srprop": "snippet|timestamp", "format": "json"}))
        for r in json.loads(b)["query"]["search"]:
            out.append({"title": r["title"], "url": f"https://{lang}.wikipedia.org/wiki/" + urllib.parse.quote(r["title"].replace(" ", "_")),
                        "snippet": _clean(r["snippet"]), "date": (r.get("timestamp") or "")[:10]})      # date of the last edit
        if len(out) >= n:
            break
    return out[:n]


def searxng(q, n, cfg):
    base = (cfg.get("searxng_url") or "").rstrip("/")
    if not base:
        raise WebError("set the SearXNG URL in Settings")
    local = urllib.parse.urlsplit(base).hostname in ("localhost", "127.0.0.1", "::1")   # your own instance may be local
    _, _, b = http(base + "/search?" + urllib.parse.urlencode({"q": q, "format": "json"}), allow_local=local)
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", ""), "date": _day(r.get("publishedDate"))}
            for r in json.loads(b).get("results", [])[:n]]


def brave(q, n, cfg):
    if not cfg.get("brave_key"):
        raise WebError("set the Brave Search API key in Settings")
    _, _, b = http("https://api.search.brave.com/res/v1/web/search?" + urllib.parse.urlencode({"q": q, "count": n}),
                   headers={"X-Subscription-Token": cfg["brave_key"], "Accept": "application/json"})
    return [{"title": r["title"], "url": r["url"], "snippet": _clean(r.get("description", "")), "date": _day(r.get("page_age"))}
            for r in json.loads(b).get("web", {}).get("results", [])[:n]]


PROVIDERS = {"duckduckgo": ddg, "wikipedia": wikipedia, "searxng": searxng, "brave": brave}


class _Text:
    """Very small HTML -> text (drops script/style/nav, keeps paragraphs)."""
    @staticmethod
    def of(raw):
        raw = re.sub(r"(?is)<(script|style|nav|header|footer|aside|form|svg)[^>]*>.*?</\1>", " ", raw)
        raw = re.sub(r"(?i)</(p|div|li|h[1-6]|tr|br)>|<br\s*/?>", "\n", raw)
        txt = html.unescape(re.sub(r"<[^>]+>", " ", raw))
        return "\n".join(l for l in (re.sub(r"[ \t]+", " ", x).strip() for x in txt.splitlines()) if len(l) > 40)


def page_info(url, limit=1200):
    """(main readable text of a public page, first `limit` characters; the date it declares) or ('', '') if it is not HTML text."""
    _, ctype, body = http(url, max_bytes=250_000)
    if "html" not in ctype and "text" not in ctype:
        return "", ""
    raw = body.decode("utf-8", "ignore")
    return _Text.of(raw)[:limit], find_date(raw)


def page_text(url, limit=1200):
    return page_info(url, limit)[0]


def prefer_recent(results, today=None, max_age_days=730):
    """Time-sensitive questions: results whose known date is older than `max_age_days` go last; the rest keep the search engine's order."""
    today = today or datetime.date.today()
    old = lambda r: bool(r.get("date")) and (today - datetime.date.fromisoformat(r["date"])).days > max_age_days
    return [r for r in results if not old(r)] + [r for r in results if old(r)]


def search(query, cfg, n=4, read=2, recent=False):
    """Search with the configured provider; read the top `read` pages. Returns [{title,url,snippet,text,date}] (date '' when unknown).
    `recent`: time-sensitive question, old results go last. Answers are cached for CACHE_SECONDS."""
    prov = cfg.get("provider", "duckduckgo")
    if prov not in PROVIDERS:
        raise WebError(f"unknown provider {prov!r}")
    key = (prov, cfg.get("searxng_url", ""), query, n, read)
    hit = _CACHE.get(key)
    if hit and time.monotonic() - hit[0] < CACHE_SECONDS:
        res = [dict(r) for r in hit[1]]
    else:
        res = PROVIDERS[prov](query, n, cfg)
        for r in res:
            r.setdefault("date", "")
        for r in res[:read]:
            try:
                r["text"], d = page_info(r["url"])
                r["date"] = r["date"] or d
            except WebError:
                r["text"] = ""
        if len(_CACHE) > 64:
            _CACHE.clear()
        _CACHE[key] = (time.monotonic(), [dict(r) for r in res])
    return prefer_recent(res) if recent else res


def as_context(results, per_chunk=700):
    """Turn results into the [{src, text, kind}] excerpts the local model gets."""
    out = []
    for r in results:
        host = urllib.parse.urlsplit(r["url"]).hostname or r["url"]
        body = (r["title"] + (f"\nPublished: {r['date']}" if r.get("date") else "") + "\n" + (r.get("text") or r["snippet"]))[:per_chunk]
        out.append({"src": "web:" + host, "text": body, "kind": "web", "url": r["url"], "title": r["title"], "date": r.get("date", "")})
    return out
