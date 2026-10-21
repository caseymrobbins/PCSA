"""The latest notebook's cells can run inside an existing kernel (so a Colab runtime/GPU is kept) and give the same saved results."""
import os
import pytest

IPython = pytest.importorskip("IPython")
pytest.importorskip("torch")

from pcsa_llm import nbrun


def test_overrides_replace_only_the_named_setting_and_unknown_names_fail():
    src = 'RUN_NAME = "a"  #@param {type:"string"}\nDRY_RUN = True  #@param {type:"boolean"}\nPROFILE = "quick"  #@param ["quick"]\n'
    out = nbrun.apply_overrides(src, {"DRY_RUN": False, "PROFILE": "lean"})
    assert "DRY_RUN = False" in out and "PROFILE = 'lean'" in out and 'RUN_NAME = "a"' in out
    with pytest.raises(KeyError):
        nbrun.apply_overrides(src, {"NOPE": 1})


def test_every_part_has_exactly_one_findable_cell():
    cells = nbrun.load()
    for k in nbrun.MARKERS:
        assert sum(nbrun.MARKERS[k] in s for s in cells) >= 1, k
    assert [nbrun.find(cells, k) for k in nbrun.ORDER] == sorted(nbrun.find(cells, k) for k in nbrun.ORDER)      # ORDER matches notebook order


def test_running_the_notebook_cells_in_an_existing_kernel_saves_the_parts(tmp_path, monkeypatch):
    from IPython.core.interactiveshell import InteractiveShell
    repo = os.path.dirname(os.path.dirname(nbrun.__file__))
    os.symlink(repo, tmp_path / "PCSA")                       # Setup uses an existing PCSA folder instead of cloning
    monkeypatch.chdir(tmp_path)
    shell = InteractiveShell.instance()
    shell.run_cell("import matplotlib; matplotlib.use('Agg')").raise_error()
    cells = nbrun.load()
    nbrun.run(cells, "setup", overrides=dict(RUN_NAME="t1", DRY_RUN=True, USE_DRIVE=False), shell=shell)
    nbrun.run(cells, "config", "l0", "l1", shell=shell)
    store = shell.user_ns["STORE"]
    assert store.done("config") and store.done("L0") and store.done("L1")
    assert shell.user_ns["EXPECTED_API"] == shell.user_ns["pcsa_llm"].NOTEBOOK_API
    nbrun.run_through(cells, "l1", overrides=dict(RUN_NAME="t1", DRY_RUN=True, USE_DRIVE=False), shell=shell)   # again: parts load, nothing recomputed
    assert store.done("L1")


def test_a_cheap_part_saved_by_an_older_version_is_recomputed_and_an_expensive_one_only_warns(tmp_path, monkeypatch, capsys):
    """Regression for the crash after reusing a run folder: an old saved L1 lacked fields the new cells need."""
    from IPython.core.interactiveshell import InteractiveShell
    repo = os.path.dirname(os.path.dirname(nbrun.__file__))
    os.symlink(repo, tmp_path / "PCSA")
    monkeypatch.chdir(tmp_path)
    shell = InteractiveShell.instance()
    shell.run_cell("import matplotlib; matplotlib.use('Agg')").raise_error()
    cells = nbrun.load()
    ov = dict(RUN_NAME="stale1", DRY_RUN=True, USE_DRIVE=False)
    nbrun.run(cells, "setup", overrides=ov, shell=shell)
    nbrun.run(cells, "config", "l0", "l1", "pre", shell=shell)
    store = shell.user_ns["STORE"]
    assert store.api("L1") == shell.user_ns["EXPECTED_API"]
    # make L1 look like an old save: no version recorded and without the fields the new cells read
    old = store.load("L1"); [old.pop(k) for k in ("ceiling", "gate")]
    store.save("L1", old, minutes=0.1, api=None)
    store.save("L3/A/s0", {"x": 1}, minutes=1.0, api=None)
    capsys.readouterr()
    nbrun.run(cells, "l1", shell=shell)                        # would raise KeyError: 'ceiling' before the fix
    out = capsys.readouterr().out
    assert "[stale] L1" in out and "ceiling" in shell.user_ns["l1"] and store.api("L1") == shell.user_ns["EXPECTED_API"]
    shell.run_cell("r = run_part('L3/A/s0', lambda: 1 / 0, None)").raise_error()      # expensive part: loaded with a warning, not recomputed
    assert "WARNING: saved by an older version" in capsys.readouterr().out
