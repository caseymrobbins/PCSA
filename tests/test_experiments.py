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


def test_gate_binding_marks_learner_tests_uninterpretable():
    from pcsa import report
    v = {"S2": {"claim": "C2", "status": "supports"}, "S1": {"claim": "C1", "status": "supports"}}
    t = report.verdict_table(v, gate0_passed=False)
    assert t.loc[t.test == "S2", "result"].iloc[0].startswith("UNINTERPRETABLE")
    assert t.loc[t.test == "S1", "result"].iloc[0] == "SUPPORTS"
    assert report.verdict_table(v, gate0_passed=True).loc[0, "result"] == "SUPPORTS"


def test_solver_study_runs_exact_and_task_reward_has_no_correction_effect():
    r = E.solver_study("tiny", gammas=(0.0, 0.95), rewards=("general", "task"), B=50, n_jobs=1)
    assert all(v["exact"] for v in r["verdict"].values())
    assert np.allclose(r["res"]["task"][0.95]["dV"], 0.0)


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
