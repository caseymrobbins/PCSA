"""Run state on disk, so any part of the notebook can be stopped and restarted. Each part saves its result under a name;
a part that is already done is loaded, not recomputed. Writes are atomic (temp file, then rename) so an interrupt cannot leave a
half-written result. Put the root on Google Drive to survive a Colab disconnect."""
import json
import os
import pickle
import shutil
import time
from pathlib import Path


def _safe(name):
    return name.replace("/", "__")


class Store:
    def __init__(self, root):
        self.root = Path(root)
        for sub in ("", "figs", "ckpt", "logs"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.index_path = self.root / "stages.json"
        self.findings_path = self.root / "findings.json"

    # -- stages
    def _index(self):
        return json.loads(self.index_path.read_text()) if self.index_path.exists() else {}

    def _write_json(self, path, obj):
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(obj, indent=1, default=str))
        os.replace(tmp, path)

    def path(self, name):
        return self.root / f"{_safe(name)}.pkl"

    def done(self, name):
        return name in self._index() and self.path(name).exists()

    def save(self, name, obj, minutes=None):
        tmp = self.path(name).with_suffix(".tmp")
        with open(tmp, "wb") as f:
            pickle.dump(obj, f)
        os.replace(tmp, self.path(name))
        idx = self._index()
        idx[name] = {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "minutes": minutes}
        self._write_json(self.index_path, idx)

    def load(self, name):
        with open(self.path(name), "rb") as f:                   # our own files; never load results from someone else
            return pickle.load(f)

    def reset(self, name):
        """Forget a finished part (and its checkpoint) so it runs again."""
        idx = self._index()
        idx.pop(name, None)
        self._write_json(self.index_path, idx)
        for p in (self.path(name), self.ckpt(name)):
            if p.exists():
                p.unlink()

    def status(self, planned=()):
        """Rows for every planned or finished part: name, state, minutes, saved_at."""
        idx = self._index()
        rows = []
        for n in list(dict.fromkeys(list(planned) + list(idx))):
            if n in idx and self.path(n).exists():
                rows.append({"part": n, "state": "done", "minutes": idx[n]["minutes"], "saved_at": idx[n]["saved_at"]})
            elif self.ckpt(n).exists():
                rows.append({"part": n, "state": "in progress (checkpoint)", "minutes": None, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ckpt(n).stat().st_mtime))})
            else:
                rows.append({"part": n, "state": "not started", "minutes": None, "saved_at": ""})
        return rows

    # -- training checkpoints and logs
    def ckpt(self, name):
        return self.root / "ckpt" / f"{_safe(name)}.ckpt"

    def jsonl(self, name):
        return self.root / "logs" / f"{_safe(name)}.jsonl"

    # -- figures and findings
    def fig(self, figure, name):
        p = self.root / "figs" / f"{_safe(name)}.png"
        figure.savefig(p, dpi=130, facecolor=figure.get_facecolor(), bbox_inches="tight")
        return p

    def add_findings(self, stage, findings):
        data = json.loads(self.findings_path.read_text()) if self.findings_path.exists() else {}
        data[stage] = [f if isinstance(f, dict) else f.__dict__ for f in findings]
        self._write_json(self.findings_path, data)

    def all_findings(self):
        return json.loads(self.findings_path.read_text()) if self.findings_path.exists() else {}
