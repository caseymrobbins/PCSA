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


def test_hub_preflight_distinguishes_ok_gated_with_access_gated_without_access_and_missing(monkeypatch):
    import huggingface_hub
    from pcsa_llm.hub import blocking, check_models

    class Info:
        def __init__(self, gated): self.gated = gated

    class GatedRepoError(Exception):
        pass

    def model_info(self, i):
        if i == "missing": raise RuntimeError("404")
        return Info("manual" if i.startswith("gated") else False)

    def download(i, f):
        if i == "gated-no": raise GatedRepoError("401")
        return "x"
    monkeypatch.setattr(huggingface_hub.HfApi, "model_info", model_info)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    st = check_models(["open", "gated-yes", "gated-no", "missing"])
    assert st["open"] == "ok" and st["gated-yes"].startswith("ok (gated") and st["gated-no"].startswith("NO ACCESS")
    assert st["missing"].startswith("UNAVAILABLE") and set(blocking(st)) == {"gated-no", "missing"}


# ---- early-stop guards ---------------------------------------------------------------------------------------------------------
def _d(cite, kl=0.1, finite=True, dead=0.0):
    return {"cite_rate": cite, "kl": kl, "finite": finite, "dead_frac": dead}


def test_guards_stop_on_sustained_problems_but_tolerate_single_spikes_and_a_policy_that_never_cited():
    from pcsa_llm.l_train import DEFAULT_GUARDS, check_guards
    g = DEFAULT_GUARDS
    healthy = [_d(0.6)] * 8
    assert check_guards(healthy, g) is None
    collapse = [_d(0.6)] * 5 + [_d(0.02)] * 6
    assert "citation rate collapsed" in check_guards(collapse, g)
    assert check_guards([_d(0.6)] * 5 + [_d(0.02)] * 3, g) is None                 # not yet 'patience' iterations
    # real logs: isolated KL spikes of 15-100 followed by normal iterations must not abort a run
    assert check_guards(healthy + [_d(0.6, kl=77.0)], g) is None
    assert check_guards(healthy + [_d(0.6, kl=77.0), _d(0.6, kl=0.4)], g) is None
    assert "sustained" in check_guards(healthy + [_d(0.6, kl=9.0), _d(0.6, kl=12.0), _d(0.6, kl=8.0)], g)
    assert "non-finite" in check_guards(healthy + [_d(0.6, finite=False)], g)
    assert check_guards([_d(0.0)] * 12, g) is None                                  # never cited: nothing was lost
    assert check_guards(collapse, None) is None                                     # guards can be switched off


def test_guard_stops_when_every_dialogue_is_dropped_repeatedly_but_not_once():
    from pcsa_llm.l_train import DEFAULT_GUARDS, check_guards
    base = [_d(0.5)] * 5
    assert check_guards(base + [_d(0.5, dead=1.0)], DEFAULT_GUARDS) is None         # one all-dropped iteration happens (seen in real logs)
    assert check_guards(base + [_d(0.5, dead=1.0)] * 2, DEFAULT_GUARDS) is None
    assert "every dialogue was dropped" in check_guards(base + [_d(0.5, dead=1.0)] * 3, DEFAULT_GUARDS)
    assert check_guards(base + [_d(0.5, dead=1.0), _d(0.5, dead=0.5), _d(0.5, dead=1.0)], DEFAULT_GUARDS) is None


# ---- user assertion-resistance probe ------------------------------------------------------------------------------------------------
def _sycophant(conv):
    """Moves 40 points of probability toward whatever the assistant asserts, evidence or not."""
    import json
    import re
    sys_ = conv[0]["content"]
    ents = re.search(r"these entities is the true source of a signal: ([^.]*)\.", sys_).group(1).split(", ")
    m = re.search(r"the answer is (\w+);", conv[-1]["content"])
    p = np.full(len(ents), 0.6 / (len(ents) - 1))
    if m:
        p[ents.index(m.group(1))] = 0.4
    p = p / p.sum()
    return json.dumps({"stated": ents[int(p.argmax())], "probs": {e: float(x) for e, x in zip(ents, p)}})


def test_assertion_probe_separates_a_user_that_ignores_assertions_from_one_that_follows_them():
    from pcsa_llm.llm_user import assertion_probe
    good = assertion_probe(MockChat(bayes_user), QS)
    assert good["pass"] and abs(good["mean_shift"]) < 1e-9 and good["n"] == len(QS)
    bad = assertion_probe(MockChat(_sycophant), QS)
    assert not bad["pass"] and bad["mean_shift"] > 0.05 and bad["share_shifted"] > 0.5
    broken = assertion_probe(MockChat(lambda c: "garbage"), QS[:3])
    assert not broken["pass"] and broken["dropped"] == 3                            # logged, not filled in


def test_l0_reports_both_the_spec_gate_and_the_assertion_probe():
    from pcsa_llm.l_experiments import l0_user_gate
    out = l0_user_gate(MockChat(bayes_user), QS, items_per_q=3)
    assert out["pass"] and out["assertion"]["pass"] and out["pass_all"]
    out = l0_user_gate(MockChat(_sycophant), QS, items_per_q=3)
    assert not out["pass_all"]
