"""Remembered facts (opt-in by use, local, editable): short notes such as "I use CachyOS" that are added to the system prompt of the LOCAL model.
They never go to a cloud provider. Stored in ~/.config/npu-lab/facts.json.   load() -> [str]   save([str]) -> [str]   as_system(base) -> str"""
import json, os

PATH = os.environ.get("NPU_LAB_FACTS") or os.path.expanduser("~/.config/npu-lab/facts.json")
MAX_FACTS, MAX_CHARS = 30, 200


def load():
    try:
        return [str(f) for f in json.load(open(PATH))][:MAX_FACTS]
    except (OSError, ValueError, TypeError):
        return []


def save(facts):
    if not isinstance(facts, list) or not all(isinstance(f, str) for f in facts):
        raise ValueError("facts must be a list of text lines")
    clean = [" ".join(f.split()) for f in facts]
    clean = [f for f in clean if f]
    if len(clean) > MAX_FACTS or any(len(f) > MAX_CHARS for f in clean):
        raise ValueError(f"at most {MAX_FACTS} facts of {MAX_CHARS} characters each")
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    tmp = PATH + ".tmp"
    json.dump(clean, open(tmp, "w"), ensure_ascii=False, indent=1)
    os.replace(tmp, PATH)
    return clean


def as_system(base):
    f = load()
    return base + ("\nWhat the user told you about themselves (use it only when it helps):\n" + "\n".join("- " + x for x in f) if f else "")
