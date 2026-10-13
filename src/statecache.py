"""Disk cache of what the model has already read: after a long prompt (a document, a code base) the KV cache of the full-attention
layers and the DeltaNet states at a position are saved, keyed by the exact tokens before that position. A later request that starts
with the same tokens (another question about the same document, or the same document next week) loads them in about a second instead of
reading the document again. Only for the hybrid Qwen3.5 engine, whose state is small (~12 KB per token).

  c = StateCache(dir, model_key, max_bytes)
  c.save(ids, pos, core)     store the first `pos` tokens' state (core.snaps must hold a snapshot at `pos`)
  c.restore(ids, core)       -> n: load the longest saved prefix of `ids` into the GPU core, returns its length (0 = none)
"""
import hashlib, json, os, time
import numpy as np

MIN_TOKENS = 4096          # shorter prefixes are cheaper to read again than to cache


def lock_down(root):
    """The saved state is derived from what you read (documents, code): owner only. Also fixes folders and files an older version created with the default permissions."""
    for d, subs, files in os.walk(root):
        os.chmod(d, 0o700)
        for f in files:
            os.chmod(os.path.join(d, f), 0o600)


def _h(ids):
    return hashlib.sha1(np.asarray(ids, dtype=np.int32).tobytes()).hexdigest()


class StateCache:
    def __init__(self, root, model_key, max_bytes=8e9):
        self.dir = os.path.join(root, model_key)
        self.max_bytes = max_bytes
        os.makedirs(self.dir, mode=0o700, exist_ok=True)
        lock_down(root)
        self.idx_path = os.path.join(self.dir, "index.json")

    def _index(self):
        try:
            return json.load(open(self.idx_path))
        except (OSError, json.JSONDecodeError):
            return {}

    def _write(self, idx):
        tmp = self.idx_path + ".tmp"
        with open(os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), "w") as f:
            json.dump(idx, f)
        os.replace(tmp, self.idx_path)

    def save(self, ids, pos, core):
        if self.max_bytes <= 0 or pos < MIN_TOKENS or pos not in core.snaps:
            return False
        h = _h(ids[:pos])
        idx = self._index()
        if h in idx:
            idx[h]["t"] = time.time()
            self._write(idx)
            return False
        conv, rec = core.snaps[pos]
        blob = {"K": core.K[:, :, :pos].cpu(), "V": core.V[:, :, :pos].cpu(), "conv": conv, "rec": rec, "n": pos}
        path = os.path.join(self.dir, h + ".pt")
        import torch
        torch.save(blob, path + ".tmp")
        os.chmod(path + ".tmp", 0o600)
        os.replace(path + ".tmp", path)
        idx[h] = {"n": pos, "bytes": os.path.getsize(path), "t": time.time()}
        while sum(e["bytes"] for e in idx.values()) > self.max_bytes and len(idx) > 1:      # least recently used first
            old = min(idx, key=lambda k: idx[k]["t"])
            if old == h:
                break
            try:
                os.remove(os.path.join(self.dir, old + ".pt"))
            except OSError:
                pass
            del idx[old]
        self._write(idx)
        return True

    def restore(self, ids, core, at_least=0):
        """Load the longest saved prefix of `ids` that is longer than `at_least` into the GPU core."""
        idx = self._index()
        for h, e in sorted(idx.items(), key=lambda kv: -kv[1]["n"]):
            n = e["n"]
            if n <= at_least or n >= len(ids) or n > core.max_ctx or _h(ids[:n]) != h:
                continue
            path = os.path.join(self.dir, h + ".pt")
            try:
                import torch
                blob = torch.load(path, map_location="cpu")
            except (OSError, RuntimeError):
                continue
            core.K[:, :, :n].copy_(blob["K"])
            core.V[:, :, :n].copy_(blob["V"])
            core.reset_states()
            for i, t in blob["conv"].items():
                core.conv[i].copy_(t)
            for i, t in blob["rec"].items():
                core.rec[i].copy_(t)
            core.snaps[n] = (blob["conv"], blob["rec"])
            e["t"] = time.time()
            self._write(idx)
            return n
        return 0
