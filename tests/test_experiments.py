"""Smoke tests: each experiment runs end to end at tiny size and returns a verdict with a status."""
import numpy as np

from pcsa import experiments as E
from pcsa import matched
from pcsa.config import PROFILES

PROFILES["tiny"] = dict(seeds=2, eval_eps=100, train_eps=1000, users=("bayes", "deference"), gate0_eps=1000, s5_eps=500)


def test_gate0_runs_and_reports_automatically():
    out = E.gate0("tiny", N=6, T=4)
    assert {"action_match", "sign_match", "pass", "learner_return", "optimum", "best_script_return"} <= set(out)
    assert E.gate0_myopic_equivalence()["pass"]


def test_s1_selective_beats_balanced_for_bayes_user():
    r = E.s1("tiny", n_jobs=1)
    v = r["verdict"][("bayes", "replication")]
    assert all(x["sel_minus_bal"] > 0 for x in v["per_k"].values())


def test_s2_s3_s4_s6_s7_run_and_tag_claims():
    for fn, claim in ((E.s2, "C2"), (E.s3, "C3"), (E.s4, "C4"), (E.s6, "C6")):
        r = fn("tiny", n_jobs=1)
        assert r["verdict"], fn.__name__
        assert all(v["claim"] == claim and v["status"] in ("supports", "weakens", "neutral") for v in r["verdict"].values())
    r7 = E.s7("tiny")
    assert all(v["claim"] == "C9" for v in r7.values())


def test_s5_environment_has_equal_returns_and_runs():
    assert np.abs(matched.exact_q()[:, 0] - matched.exact_q()[:, 1]).max() == 0
    r = matched.s5(seeds=2, episodes=400)
    assert r["verdict"]["status"] in ("supports", "weakens", "neutral", "invalid")
