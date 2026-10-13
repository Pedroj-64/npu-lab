"""Lines the user wrote that carry a hard fact (numbers, ports, paths, file names, "called X"), kept word for word through a compaction.
A model-written summary can drop a fact or change a digit (measured: 8431 became 8432 on the 4B); copying the user's own line cannot.
  pin(old_turns) -> [str]     block(lines) -> str"""
import re

FACT = re.compile(r"\d|(?:^|\s)[~/.][\w.-]*/|\w+\.(?:py|js|ts|rs|c|h|md|json|toml|ya?ml|sh|txt|cfg|conf)\b|\b(?:called|named|llamo|llama|llamad[oa]|se llama)\b", re.I)
MAX_LINES, MAX_CHARS = 12, 220


def pin(old, max_lines=MAX_LINES, max_chars=MAX_CHARS):
    seen, out = set(), []
    for m in old:
        if m.get("role") != "user":
            continue
        for line in m["content"].splitlines():
            line = line.strip().removeprefix("- ").strip()
            if len(line) < 8 or line.startswith(("[Notes", "Pinned", "Summary", "```")) or not FACT.search(line):
                continue
            line = line[:max_chars]
            if line.lower() not in seen:
                seen.add(line.lower()); out.append(line)
    return out[-max_lines:]                   # the newest facts win when there are too many


def block(lines):
    return "Pinned (the user's own words, kept word for word):\n" + "\n".join("- " + l for l in lines) + "\n\nSummary:\n" if lines else ""
