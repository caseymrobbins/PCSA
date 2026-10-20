"""Pre-registration check and run metadata. The run refuses to start without a matching prereg hash."""
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PREREG = ROOT / "prereg" / "prereg.json"
HASH = ROOT / "prereg" / "prereg.sha256"


def freeze(path=PREREG, hash_path=HASH):
    hash_path.write_text(hashlib.sha256(Path(path).read_bytes()).hexdigest() + "\n")


def check_prereg(path=PREREG, hash_path=HASH):
    path, hash_path = Path(path), Path(hash_path)
    if not path.exists() or not hash_path.exists():
        raise FileNotFoundError("pre-registration file or its hash is missing; refusing to run")
    got, want = hashlib.sha256(path.read_bytes()).hexdigest(), hash_path.read_text().strip()
    if got != want:
        raise ValueError(f"pre-registration changed since it was frozen ({got[:12]} != {want[:12]}); "
                         "log the change as a deviation and re-freeze")
    return want


def _git():
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except Exception as e:                         # logged, not hidden
        return f"unavailable: {e}"


def start_run(name, config, seeds, out_dir=ROOT / "runs", prereg=(PREREG, HASH)):
    """Create runs/<timestamp>_<name>/meta.json with config, git commit, seeds, versions and the prereg hash."""
    import numpy
    h = check_prereg(*prereg)
    d = Path(out_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}_{name}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "meta.json").write_text(json.dumps({
        "name": name, "config": config, "seeds": list(seeds), "git_commit": _git(), "prereg_sha256": h,
        "python": sys.version, "numpy": numpy.__version__, "platform": platform.platform()}, indent=2, default=str))
    return d


if __name__ == "__main__":
    if sys.argv[1:] == ["freeze"]:
        freeze()
        print(HASH.read_text().strip())
