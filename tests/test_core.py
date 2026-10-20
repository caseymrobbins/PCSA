"""Unit tests for the checklist in 'Code requirements for review'."""
import dataclasses
import json
import math

import numpy as np
import pytest

from pcsa.config import EnvCfg, UserSpec, WorldCfg, user_variants
from pcsa.env import AGREE, BAL, SEL, AgentObservation, AuditTrace, DialogueEnv, PrivilegedEvalState
from pcsa.metrics import auc, boot_ci, equivalent, holm
from pcsa.policies import TabularQ, argmax_random, rollout, scripted
from pcsa.rewards import REWARDS, r_general
from pcsa.runlog import check_prereg, freeze
from pcsa.solver import ExactSolver
from pcsa.users import Users
from pcsa.verify import Verifier
from pcsa.world import World, entropy, kl, make_world

DET = UserSpec("bayes", tau=0.05)          # near-deterministic stated position, for hand-computed episodes


def tiny_world(L, p0, hstar, pstar=None):
    L, p0 = np.array([L], float), np.array([p0], float)
    pstar = np.array([pstar], float) if pstar is not None else (lambda q: q / q.sum(1, keepdims=True))(p0 * L.prod(1))
    return World(np.array([hstar]), L, L.argmax(2), p0, p0.argmax(1), pstar)


# ---- randomness and ties -----------------------------------------------------------------------
def test_rerun_reproduces_logs_byte_for_byte():
    def run():
        env = DialogueEnv(EnvCfg(), user_variants()["deference"], "general", 50, 7)
        rollout(env, scripted("sel3_agree"), 7)
        return json.dumps(env.turn_log(50, 7), sort_keys=True)
    assert run() == run()


def test_world_is_determined_by_seed():
    a, b = (make_world(WorldCfg(), 10, np.random.default_rng(3)) for _ in range(2))
    assert all(np.array_equal(getattr(a, f.name), getattr(b, f.name)) for f in dataclasses.fields(a))


def test_argmax_breaks_ties_uniformly_and_respects_mask():
    rng = np.random.default_rng(0)
    q = np.tile([1.0, 1.0, 1.0, 5.0, 0.0], (6000, 1))
    allowed = np.tile([True, True, True, False, True], (6000, 1))
    picks = argmax_random(q, allowed, rng)
    assert set(np.unique(picks)) == {0, 1, 2}
    assert all(abs((picks == k).mean() - 1 / 3) < 0.03 for k in range(3))


# ---- interfaces ---------------------------------------------------------------------------------
def test_types_are_separate_and_agent_sees_no_posterior():
    assert len({AgentObservation, PrivilegedEvalState, AuditTrace}) == 3
    agent = {f.name for f in dataclasses.fields(AgentObservation)}
    assert not agent & {"p", "posterior", "pstar", "kappa", "hstar", "unseen"}
    env = DialogueEnv(EnvCfg(), user_variants()["bayes"], "general", 5, 0)
    assert type(env.reset()) is AgentObservation
    obs, r, done, info = env.step(np.full(5, AGREE))
    assert type(obs) is AgentObservation and type(env.priv()) is PrivilegedEvalState and type(env.audit) is AuditTrace


def test_reward_depends_only_on_observation():
    obs = AgentObservation(2, 15, np.array([0, 1]), np.array([0, 1]), np.array([0, 0]), np.array([False, False]),
                           np.array([AGREE, BAL]), np.array([0, 0]))
    r1 = r_general(obs)
    assert list(r1) == [1.0, -1.0]                       # second user contradicts the assistant (0 vs 1)


# ---- hand-computed math -------------------------------------------------------------------------
def test_bayes_update_matches_closed_form():
    u = Users(UserSpec("bayes"), np.array([[0.6, 0.4]]))
    got = u.peek(np.array([[0.2, 0.8]]))[0]
    assert np.allclose(got, [0.12 / 0.44, 0.32 / 0.44])
    w = Users(UserSpec("weighted", beta=0.5), np.array([[0.6, 0.4]]))
    q = np.array([0.6 * 0.2 ** 0.5, 0.4 * 0.8 ** 0.5])
    assert np.allclose(w.peek(np.array([[0.2, 0.8]]))[0], q / q.sum())


def test_confirm_user_discounts_only_opposing_items():
    u = Users(UserSpec("confirm", beta=1.0, beta_oppose=0.3), np.array([[0.7, 0.3]]))
    sup, opp = u.peek(np.array([[0.8, 0.2]]))[0], u.peek(np.array([[0.2, 0.8]]))[0]
    assert np.allclose(sup, np.array([0.56, 0.06]) / 0.62)
    q = np.array([0.7 * 0.2 ** 0.3, 0.3 * 0.8 ** 0.3])
    assert np.allclose(opp, q / q.sum())


def test_kl_entropy_cw_sb_ds_hand_cases():
    p, q = np.array([0.9, 0.1]), np.array([0.5, 0.5])
    assert math.isclose(float(kl(p, q)), 0.9 * math.log(1.8) + 0.1 * math.log(0.2), rel_tol=1e-9)
    assert math.isclose(float(entropy(q)), math.log(2), rel_tol=1e-9)
    w = tiny_world([[0.8, 0.2], [0.2, 0.8]], [0.9, 0.1], hstar=1, pstar=[0.9, 0.1])
    env = DialogueEnv(EnvCfg(T=1), DET, "replication", 1, 0, world=w)
    env.reset()
    env.step(np.array([SEL]))                              # shows item 0 (favors the lean)
    s = env.summary()
    pT = np.array([0.72, 0.02]) / 0.74
    d = 0.9 * math.log(0.9 / pT[0]) + 0.1 * math.log(0.1 / pT[1])
    assert math.isclose(float(s["D"][0]), d, rel_tol=1e-6)
    assert math.isclose(float(s["DS"][0]), d, rel_tol=1e-6)      # updating on item 1 recovers p* here
    assert s["CW"][0] == 1.0                               # argmax is 0 != h*, max p = .97 > .8
    assert math.isclose(float(s["SB"][0]), 1 - 0.5)        # shown item favors lean; 1 of 2 unseen items did


# ---- environment mechanics ----------------------------------------------------------------------
def test_evidence_shown_without_replacement():
    env = DialogueEnv(EnvCfg(), user_variants()["bayes"], "general", 200, 1)
    rollout(env, scripted("selective"), 1)
    items = np.array(env.hist["item"])                     # [T,B]
    assert all(len(set(items[:, e])) == len(items[:, e]) for e in range(items.shape[1]))


def test_event_order_three_turn_episode_hand_computed():
    L = [[0.01, 0.99]]                                     # one item that flips the user to hypothesis 1
    expected = {"replication": [1, 0, 1], "general": [1, -1, 1], "task": [0, 0, 1]}
    for name, want in expected.items():
        env = DialogueEnv(EnvCfg(T=3), DET, name, 1, 0, world=tiny_world([L[0]], [0.9, 0.1], hstar=1))
        env.reset()
        got = [env.step(np.array([a]))[1][0] for a in (AGREE, SEL, AGREE)]
        assert got == want, name
        assert list(np.array(env.hist["corr"])[:, 0]) == [False, True, False]     # correction flag set in step 5
        assert list(np.array(env.hist["y"])[:, 0]) == [0, 1, 1]


def test_correction_injection_changes_stated_position_only():
    kw = dict(cfg=EnvCfg(), user=user_variants()["deference"], reward="general", B=300, seed=5)
    a, b = DialogueEnv(**kw), DialogueEnv(**kw, inject_at=4)
    a.reset(); b.reset()
    for t in range(5):
        a.step(np.full(300, AGREE if t < 3 else SEL)); b.step(np.full(300, AGREE if t < 3 else SEL))
    assert np.array_equal(a.user.p, b.user.p) and np.array_equal(a.user.kappa, b.user.kappa)
    ya, yb = np.array(a.hist["y"])[4], np.array(b.hist["y"])[4]
    m = a.m
    assert (ya != yb).any() and ((yb[ya != yb] != m[ya != yb])).all()


def test_disallowed_action_fails_loudly():
    env = DialogueEnv(EnvCfg(actions=(0, 1)), user_variants()["bayes"], "general", 4, 0)
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.full(4, SEL))


def test_per_turn_log_schema():
    env = DialogueEnv(EnvCfg(), user_variants()["bayes"], "general", 3, 0)
    rollout(env, scripted("sel3_agree"), 0)
    row = env.turn_log(1, 0)[0]
    assert set(row) == {"episode", "seed", "turn", "state", "action", "cited", "stated", "posterior", "reward",
                        "correction", "verifier_exact_allowed", "monitor"}


# ---- mask and learner ---------------------------------------------------------------------------
def test_mask_removed_from_choice_and_from_learning_target():
    q = TabularQ(gamma=1.0, alpha=0.5)
    q.Q[1] = [1.0, 2.0, 100.0, 0.0, 0.0]
    allowed = np.array([[True, True, False, False, False]])
    # action choice never takes the masked action even though it has the largest value
    assert q.act(np.array([1]), allowed, 0.0, np.random.default_rng(0))[0] == 1
    assert all(q.act(np.array([1]), allowed, 1.0, np.random.default_rng(s))[0] in (0, 1) for s in range(50))
    q.update(np.array([0]), np.array([1]), np.array([0.0]), np.array([1]), allowed, np.array(False), np.zeros(1), np.zeros(1))
    assert 0.0 < q.Q[0, 1] <= 2.0 * (1 - (1 - 0.5))        # moved toward max over allowed (2), not 100


def test_exact_mask_blocks_actions_that_raise_kl():
    # user leans to hypothesis 0 but p* = (0.1, 0.9); item 0 pushes toward 0 (raises KL), item 1 toward 1 (lowers it)
    w = tiny_world([[0.8, 0.2], [0.2, 0.8]], [0.9, 0.1], hstar=1, pstar=[0.1, 0.9])
    for seed in range(8):                                  # balanced picks item 0 or 1 depending on the seed
        env = DialogueEnv(EnvCfg(T=2), DET, "general", 1, seed, verifier=Verifier("exact"), world=w)
        env.reset()
        assert env.cand[0, SEL] == 0 and not env.allowed[0, SEL]
        assert env.allowed[0, AGREE]
        assert env.allowed[0, BAL] == (env.cand[0, BAL] == 1)
    free = DialogueEnv(EnvCfg(T=2), DET, "general", 1, 0, world=w)
    free.reset()
    assert free.allowed[0, SEL]


def test_detector_off_leaves_reward_untouched_and_on_subtracts_only_detections():
    def run(lam):
        env = DialogueEnv(EnvCfg(actions=(0, 1, 2, 3), detector_lambda=lam), user_variants()["bayes"], "general", 400, 2)
        env.reset()
        rs, dets = [], []
        for t in range(15):
            _, r, _, _ = env.step(np.full(400, SEL if t < 5 else AGREE))
            rs.append(r); dets.append(env.audit.detected.copy())
        return np.array(rs), np.array(dets)
    r_off, d_off = run(None)
    r_on, d_on = run(0.5)
    assert not d_off.any() and d_on.any()
    assert np.allclose(r_on, r_off - 0.5 * d_on)


# ---- solver -------------------------------------------------------------------------------------
def test_solver_matches_independent_brute_force():
    L = np.array([[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]])
    p0, T, g = np.array([0.55, 0.45]), 3, 0.9

    def post(seen):
        q = p0.copy()
        for i in seen:
            q = q * L[i]
        return q / q.sum()

    def bf(t, seen, m, yl):                                # returns [agree, balanced, selective]
        free = [i for i in range(3) if i not in seen]
        def branch(agree, item):
            m2, seen2 = (yl, seen) if agree else (m, seen | {item})
            w = post(seen2)
            tot = 0.0
            for y in range(2):
                r = (1.0 if y == m2 else -1.0) if agree else 0.0
                tot += w[y] * (r + (0 if t == T - 1 else g * max(bf(t + 1, seen2, m2, y))))
            return tot
        return [branch(True, None), np.mean([branch(False, i) for i in free]),
                branch(False, max(free, key=lambda i: L[i, yl]))]

    sol = ExactSolver(L, p0, 1, T, g, "replication")
    for yl in (0, 1):
        assert np.allclose(sol.Q(0, 0, -1, yl), bf(0, frozenset(), -1, yl))


def test_solver_myopic_selective_equals_balanced_under_replication():
    w = make_world(WorldCfg(N=8), 1, np.random.default_rng(1))
    sol = ExactSolver(w.L[0], w.p0[0], int(w.hstar[0]), 5, 0.0, "replication")
    sol.Q(0, 0, -1, int(w.lean[0]))
    assert max(abs(v[SEL] - v[BAL]) for v in sol._q.values()) < 1e-12


# ---- pre-registration ---------------------------------------------------------------------------
def test_run_refuses_without_prereg_or_with_changed_prereg(tmp_path):
    f, h = tmp_path / "p.json", tmp_path / "p.sha256"
    with pytest.raises(FileNotFoundError):
        check_prereg(f, h)
    f.write_text("{}"); freeze(f, h)
    assert check_prereg(f, h)
    f.write_text('{"changed": 1}')
    with pytest.raises(ValueError):
        check_prereg(f, h)


def test_committed_prereg_is_intact():
    assert check_prereg()


# ---- statistics helpers -------------------------------------------------------------------------
def test_stats_helpers():
    assert auc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0 and auc([1, 1, 1, 1], [0, 0, 1, 1]) == 0.5
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    m, lo, hi = boot_ci(np.arange(10.0))
    assert lo < m < hi
    assert equivalent(np.random.default_rng(0).normal(0, 0.005, 10))[0]
    assert not equivalent(np.full(10, 0.1))[0]
