"""LLM-suite checks that do not need a GPU or downloaded weights: parsers, validation, gate, metrics, reward-model cache,
probes, and (with a tiny random model on CPU) the PPO/LoRA trainer end to end."""
import json
import numpy as np
import pytest

from pcsa.world import kl
from pcsa_llm.citations import parse_answer, parse_cites
from pcsa_llm.dialogue import DialogueCfg, run_dialogues
from pcsa_llm.l_experiments import l1_reward_incentive, l2_stage_decomposition
from pcsa_llm.llm_user import gate_stats, parse_user_json, run_user_gate, spearman
from pcsa_llm.metrics import episode_metrics, summarize
from pcsa_llm.models import CachedRM, MockChat
from pcsa_llm.probes import confidence_probe, top_answer_probe
from pcsa_llm.ppo import discounted_returns, gae
from pcsa_llm.dry import FakeRM, bayes_user
from pcsa_llm.textworld import make_questions

QS = make_questions(12, 0)


def test_textworld_posterior_is_exact_for_the_stated_likelihoods():
    q = QS[0]
    # recompute p* from the percentages printed in the statements
    import re
    p = q.p0.copy()
    for i, t in enumerate(q.items):
        ell, oth = map(int, re.findall(r"(\d+)%", t))
        like = np.full(q.K, oth / 100)
        like[q.entities.index(re.search(r"names (\w+)\.", t).group(1))] = ell / 100
        p = p * like
    assert np.allclose(p / p.sum(), q.pstar)
    assert all(abs(q.L[i].sum() - 1) < 1e-9 for i in range(q.N))


def test_questions_are_true_statements_drawn_from_hstar():
    fav_true = np.mean([(q.fav == q.hstar).mean() for q in make_questions(300, 1)])
    assert 0.55 < fav_true < 0.75                      # p_true = 0.65 by construction
    assert np.mean([q.lean != q.hstar for q in make_questions(600, 2)]) == pytest.approx(2 / 3, abs=0.06)


# ---- citation parser: malformed and missing tags --------------------------------------------------------------------------
@pytest.mark.parametrize("text,ids,bad", [
    ("no tags at all", [], False),
    ("[[cite: 1, 3]] and [[CITE:7]]", [1, 3, 7], False),
    ("[[cite: 1, 1, 1]]", [1], False),
    ("[[cite: 99]]", [], True),
    ("[[cite: two]]", [], True),
    ("[[cite: ]]", [], True),
    ("[[cite: 2, x, 5]]", [2, 5], True),
    ("[[cite 3]]", [], False),                          # not the tag format: counts as no citation
    ("[[cite: 3", [], False),                           # unterminated tag
])
def test_citation_parser(text, ids, bad):
    assert parse_cites(text, 14) == (ids, bad)


def test_answer_parser_takes_last_tag_and_ignores_unknown_names():
    ents = ["Aldern", "Brask", "Corvel"]
    assert parse_answer("[[answer: aldern]] ... [[answer: Corvel]]", ents) == 2
    assert parse_answer("[[answer: Zed]]", ents) is None and parse_answer("nothing", ents) is None


# ---- LLM user JSON ---------------------------------------------------------------------------------------------------------
def test_user_json_validation():
    e = ["A", "B", "C"]
    ok = '{"stated": "B", "probs": {"A": 0.2, "B": 0.6, "C": 0.2}}'
    assert parse_user_json("noise " + ok + " noise", e)[0] == 1
    assert parse_user_json('{"stated": "B", "probs": {"A": 0.2, "B": 0.62, "C": 0.2}}', e) is not None       # within tolerance, renormalized
    for bad in ['{"stated": "Z", "probs": {"A": 0.2, "B": 0.6, "C": 0.2}}', '{"stated": "A", "probs": {"A": 0.5, "B": 0.6, "C": 0.2}}',
                '{"stated": "A", "probs": {"A": 1}}', "not json", '{"stated": "A", "probs": {"A": -0.1, "B": 0.6, "C": 0.5}}']:
        assert parse_user_json(bad, e) is None


def test_user_gate_passes_for_a_bayesian_mock_and_fails_for_a_contrarian():
    good = run_user_gate(MockChat(lambda c: bayes_user(c)), QS, items_per_q=3)
    assert good["pass"] and good["spearman"] > 0.99 and good["direction"] == 1.0
    # a user whose probabilities ignore the evidence fails the gate
    flat = MockChat(lambda c: '{"stated": "%s", "probs": {%s}}' % (QS[0].entities[0], ", ".join(f'"{e}": 0.3333' for e in QS[0].entities)))
    assert not run_user_gate(flat, QS[:1], items_per_q=3)["pass"]


def test_user_gate_logs_dropped_dialogues_instead_of_filling():
    out = run_user_gate(MockChat(lambda c: "garbage"), QS[:3], items_per_q=2)
    assert out["dropped_dialogues"] == 3 and out["n"] == 0


# ---- dialogue runner and metrics -----------------------------------------------------------------------------------------------
def test_dialogue_runner_with_mock_models_logs_turns_and_computes_metrics():
    q = QS[:6]

    def asst(conv):
        turn = sum(m["role"] == "assistant" for m in conv)
        who = [m["content"] for m in conv if m["role"] == "user"][-1]
        return "No tag here." if turn == 3 else f"Look at [[cite: {turn}]]. [[answer: {q[0].entities[0]}]]"
    eps = run_dialogues(MockChat(asst), MockChat(bayes_user), q, DialogueCfg(T=4), seed=0)
    assert all(len(e.turns) == 4 and e.stopped is None for e in eps)
    e = eps[0]
    assert e.turns[3]["category"] == "agree" and e.turns[0]["cited"] == [0]
    assert e.seen == {0, 1, 2}                                    # the no-tag turn adds no evidence
    m = episode_metrics(e)
    pT = e.turns[-1]["p_after"]
    assert m["D"] == pytest.approx(float(kl(e.q.pstar, pT)))
    s = summarize(eps)
    assert s["n"] == 6 and 0 <= s["CW"] <= 1 and s["stopped"] == 0.0


def test_dialogue_runner_drops_a_dialogue_when_the_user_json_stays_invalid():
    eps = run_dialogues(MockChat(lambda c: "hi [[answer: x]]"), MockChat(lambda c: "nope"), QS[:2], DialogueCfg(T=3, user_retries=1))
    assert all(e.stopped and not e.turns for e in eps)


def test_correction_flag_requires_contradicting_the_assistants_stated_answer():
    q = QS[:1]
    ent = q[0].entities
    user = MockChat(lambda c: json.dumps({"stated": ent[1], "probs": {ent[0]: 0.2, ent[1]: 0.6, ent[2]: 0.2}}))
    eps = run_dialogues(MockChat(lambda c: f"[[answer: {ent[0]}]]"), user, q, DialogueCfg(T=2))
    assert [t["correction"] for t in eps[0].turns] == [True, True]


# ---- L1 / L2 plumbing ------------------------------------------------------------------------------------------------------------
def test_l1_scores_every_turn_and_caches_by_input_hash(tmp_path):
    rm = CachedRM(FakeRM(), tmp_path / "c.jsonl", "fake")
    out = l1_reward_incentive({"fake": rm}, QS[:5], T=6)
    assert set(out["per_rm"]["fake"]["k"]) == {1, 2, 3, 5} and out["verdict"]["status"] in ("supports", "weakens", "neutral")
    first_calls = rm.rm.__class__ and rm.calls
    rm2 = CachedRM(FakeRM(), tmp_path / "c.jsonl", "fake")
    l1_reward_incentive({"fake": rm2}, QS[:5], T=6)
    assert rm2.hits == rm2.calls                               # everything served from the cache file


def test_l2_runs_every_stage_on_the_same_questions():
    asst = MockChat(lambda c: "[[cite: 0]] [[answer: x]]")
    out = l2_stage_decomposition({"base": asst, "sft": asst}, MockChat(bayes_user), QS[:4], seeds=(0, 1), cfg=DialogueCfg(T=2))
    assert set(out) == {"base", "sft"} and len(out["base"]["D"]) == 2


# ---- probes ------------------------------------------------------------------------------------------------------------------------------
def test_probes_recover_a_planted_direction_and_split_by_question():
    rng = np.random.default_rng(0)
    n, d = 600, 24
    groups = np.repeat(np.arange(30), 20)
    direction = rng.normal(size=d); direction /= np.linalg.norm(direction)
    conf = rng.uniform(0.3, 1.0, n)
    X = rng.normal(size=(n, d)) * 0.3 + np.outer(conf, direction) * 8
    dvec, r = confidence_probe(X, conf, groups)
    assert r > 0.9 and abs(float(dvec @ direction)) > 0.8
    top = rng.integers(0, 3, n)
    X2 = rng.normal(size=(n, d)) * 0.3; X2[np.arange(n), top] += 2
    acc, base = top_answer_probe(X2, top, groups, 3)
    assert acc > 0.9 > base + 0.3 or acc > base + 0.3
    chance = top_answer_probe(rng.normal(size=(n, d)), top, groups, 3)
    assert chance[0] < 0.6


def test_spearman_and_ppo_math():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    r = np.array([1.0, 0.0, 2.0])
    assert np.allclose(discounted_returns(r, 0.0), r) and np.allclose(discounted_returns(r, 1.0), [3, 2, 2])
    adv, ret = gae(r, np.zeros(3), 0.0, 0.95)                    # gamma = 0: advantage is reward minus value
    assert np.allclose(adv, r) and np.allclose(ret, r)
    adv, _ = gae(r, np.zeros(3), 1.0, 1.0)
    assert np.allclose(adv, [3, 2, 2])


def _run(reward, D, SB=0.1, CW=0.0):
    return {"history": [{"reward": reward, "summary": {"flag_rate": 0.0}}] * 3, "eval": [{"D": D, "SB": SB, "CW": CW}]}


def test_l5_l6_verdicts_are_neutral_when_there_is_no_steering_to_remove():
    from pcsa_llm.verdicts import l5_verdict, l6_verdict
    b, c, f = [_run(1.0, 1.0)], [_run(1.0, 1.0)], [_run(1.0, 1.0)]
    assert l5_verdict(f, b, c)["status"] == "neutral"                        # C not above B: vacuous, not 'supports'
    assert l5_verdict([_run(0.9, 1.0)], b, [_run(1.2, 3.0)])["status"] == "supports"
    assert l5_verdict([_run(0.9, 3.0)], b, [_run(1.2, 3.0)])["status"] == "weakens"
    base = {"SB": 0.0, "D": 1.0}
    assert l6_verdict({0.5: [_run(1, 1.0, 0.0)], 2.0: [_run(1, 1.0, 0.0)]}, base, arm_c_D=1.0)["status"] == "neutral"
