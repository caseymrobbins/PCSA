"""Run the LATEST notebook's cells inside the CURRENT runtime, so you can keep the GPU you have without opening a new notebook.
The cells are read from the repo clone (notebooks/pcsa_llm_gpu.ipynb) and executed in this kernel, so every variable (STORE, ARMS, ...)
behaves exactly as if you had run the cells of the new notebook. Parts that are already done load from disk."""
import json
import re
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent.parent / "notebooks" / "pcsa_llm_gpu.ipynb"
MARKERS = {"setup": "#@title Setup", "status": "rows = STORE.status(PLANNED)", "config": "def compute_config", "l0": "def compute_l0", "lab": "def compute_lab",
           "l1": "def compute_l1", "pre": "def compute_pre", "l2": 'run_part(f"L2/{name}"', "l3defs": "def show_run", "armA": "#@title L3 arm A",
           "armB": "#@title L3 arm B", "armC": "#@title L3 arm C", "l3": "#@title L3 analysis", "l4": 'f"L4/{arm}"', "l5": 'name = f"L5/s{s}"',
           "l6": "BYLAM = {}", "report": "build_report(STORE"}
ORDER = ["setup", "status", "config", "l0", "l1", "pre", "l2", "l3defs", "armA", "armB", "armC", "l3", "l4", "l5", "l6", "report"]   # 'lab' is optional


def load(path=None):
    nb = json.load(open(path or DEFAULT))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def find(cells, key):
    hits = [i for i, s in enumerate(cells) if MARKERS[key] in s]
    if not hits:
        raise KeyError(f"no cell for '{key}' (looked for {MARKERS[key]!r}); known: {list(MARKERS)}")
    return hits[0]


def apply_overrides(src, overrides):
    """Replace `NAME = ...  #@param ...` lines, the way the form fields of a notebook would."""
    for name, value in (overrides or {}).items():
        pat = re.compile(rf"^{re.escape(name)} = .*$", re.M)
        if not pat.search(src):
            raise KeyError(f"no setting called {name} in the Setup cell")
        src = pat.sub(f"{name} = {value!r}  #@param (overridden)", src, count=1)
    return src


def run(cells, *keys, overrides=None, shell=None):
    """Execute the named cells in order (e.g. run(cells, 'setup', 'config', 'l0')). Stops with the original error if a cell fails."""
    if shell is None:
        from IPython import get_ipython
        shell = get_ipython()
    for k in keys:
        src = cells[find(cells, k)]
        if k == "setup":
            src = apply_overrides(src, overrides)
        print(f"----- {k} -----")
        shell.run_cell(src).raise_error()


def run_through(cells, until, overrides=None, shell=None):
    """Run setup ... `until` in notebook order. Finished parts are loaded from disk, so re-running from the top is cheap."""
    run(cells, *ORDER[: ORDER.index(until) + 1], overrides=overrides, shell=shell)
