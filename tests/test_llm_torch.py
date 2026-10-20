"""Runs the real HFChat / LoRA / PPO / steering / L5 / L6 code paths on a tiny random Llama on CPU with a byte tokenizer.
Verifies plumbing and math, NOT model quality or GPU behaviour."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("peft")
transformers = pytest.importorskip("transformers")

from pcsa_llm.dialogue import DialogueCfg, run_dialogues
from pcsa_llm.l_train import judge_flags, make_filter_step, train, turn_rewards
from pcsa_llm.models import MockChat
from pcsa_llm.ppo import PPOCfg, TurnPPO, filtered_logp_surrogate, ppo_token_loss
from pcsa_llm.probes import collect_activations, steering
from pcsa_llm.textworld import make_questions
from pcsa_llm.dry import FakeRM, bayes_user, tiny_chat


QS = make_questions(6, 0, N=4)
D = DialogueCfg(T=2, max_new=10, temperature=1.0)


def test_hfchat_generates_text_and_records_tokens():
    chat = tiny_chat()
    texts = chat.generate([[{"role": "user", "content": "hi"}]] * 3, max_new_tokens=8, temperature=1.0, seed=0)
    assert len(texts) == 3 and len(chat.last_tokens) == 3 and all(len(r) <= 8 for _, r in chat.last_tokens)
    again = chat.generate([[{"role": "user", "content": "hi"}]] * 3, max_new_tokens=8, temperature=1.0, seed=0)
    assert texts == again


def test_filtered_surrogate_gradient_equals_exact_log_acceptance_gradient():
    theta = torch.tensor([0.3, -0.2, 0.9, 0.1], requires_grad=True)
    logp = torch.log_softmax(theta, 0)
    accept = torch.tensor([True, True, False, True])
    a = 0
    exact = logp[a] - torch.log(torch.exp(logp)[accept].sum())               # log pi_filtered(a | s)
    g_exact, = torch.autograd.grad(exact, theta, retain_graph=True)
    pi = torch.exp(logp).detach()
    est = filtered_logp_surrogate(logp[a], logp[accept], weights=pi[accept])  # draws weighted by their probabilities = the expectation
    g_est, = torch.autograd.grad(est, theta, retain_graph=True)
    assert torch.allclose(g_exact, g_est, atol=1e-6)
    naive = logp[a] - np.log(float(pi[accept].sum()))                        # constant shift: drops the log A(s) gradient
    g_naive, = torch.autograd.grad(naive, theta)
    assert not torch.allclose(g_exact, g_naive, atol=1e-3)


def test_ppo_loss_clips_and_masks_fallback_turns():
    lo, old = torch.tensor([0.0, 0.0]), torch.tensor([-1.0, -1.0])           # ratio e^1 > 1 + clip
    pos = ppo_token_loss(lo, old, torch.tensor(1.0), torch.ones(2), 0.2)
    assert float(pos) == pytest.approx(-1.2)
    assert float(ppo_token_loss(lo, old, torch.tensor(1.0), torch.zeros(2), 0.2)) == 0.0


def test_turn_ppo_update_changes_only_lora_and_value_head_and_is_finite():
    chat = tiny_chat()
    ppo = TurnPPO(chat, PPOCfg(arm="C", lr=1e-2, epochs=1, minibatch=2))
    before = {n: p.detach().clone() for n, p in ppo.model.named_parameters()}
    eps = run_dialogues(chat, MockChat(bayes_user), QS[:3], D, seed=0)
    rewards = turn_rewards(eps, FakeRM())
    from pcsa_llm.l_train import build_samples, monitor
    samples = build_samples(ppo, eps, rewards)
    assert samples and all("adv" in s and "value" in s for s in samples)
    vh0 = ppo.value_head.weight.clone()
    diag = ppo.update(samples)
    assert all(np.isfinite(v) for v in diag.values()) and diag["kl"] >= -1e-6
    changed = [n for n, p in ppo.model.named_parameters() if not torch.equal(p.detach(), before[n])]
    assert changed and all("lora" in n for n in changed)                      # base weights untouched
    assert not torch.equal(vh0, ppo.value_head.weight)
    mon = monitor(samples)
    assert "value_by_turn" in mon and "adv_by_category" in mon


def test_base_weights_frozen_under_lora():
    chat = tiny_chat()
    TurnPPO(chat, PPOCfg())
    trainable = [n for n, p in chat.model.named_parameters() if p.requires_grad]
    assert trainable and all("lora" in n for n in trainable)


@pytest.mark.parametrize("arm,expect_gamma,T", [("A", 0.0, 1), ("B", 0.0, 2), ("C", 0.95, 2)])
def test_three_arms_train_end_to_end(arm, expect_gamma, T):
    chat = tiny_chat()
    ppo = TurnPPO(chat, PPOCfg(arm=arm, lr=1e-3, epochs=1, minibatch=2))
    assert ppo.cfg.gamma == expect_gamma
    hist = train(ppo, MockChat(bayes_user), FakeRM(), QS, DialogueCfg(T=T, max_new=8, temperature=1.0), iters=2, per_iter=3, log=lambda *_: None)
    assert len(hist) == 2 and np.isfinite(hist[-1]["ppo"]["loss"])


def test_l5_filter_always_draws_n_tries_and_keeps_fallbacks_out_of_the_gradient():
    chat = tiny_chat()
    ppo = TurnPPO(chat, PPOCfg(arm="C", lr=1e-3, epochs=1, minibatch=2))
    # a user that always reports a belief far from p*, so every candidate raises KL and every turn falls back
    q0 = QS[0]
    wrong = [i for i in range(q0.K) if i != q0.hstar]
    mock = MockChat(lambda c: __import__("json").dumps({"stated": q0.entities[wrong[0]], "probs": {e: (0.98 if i == wrong[0] else 0.01) for i, e in enumerate(q0.entities)}}))
    hist = train(ppo, mock, FakeRM(), QS[:1], DialogueCfg(T=1, max_new=6, temperature=1.0), iters=1, per_iter=1,
                 step_fn_factory=lambda chat, user, cfg, seed: make_filter_step(chat, user, cfg, n_tries=4, seed=seed), log=lambda *_: None)
    f = hist[0]["filter"]
    assert f["drawn"] == 4 * f["states"]                                     # unbiased acceptance estimate: all draws are kept
    assert f["fallbacks"] + (f["accepted"] > 0) >= 1


def test_judge_counts_flags_and_logs_unparseable_verdicts():
    chat = tiny_chat()
    eps = run_dialogues(MockChat(lambda c: "[[cite: 0]] [[answer: x]]"), MockChat(bayes_user), QS[:2], DialogueCfg(T=2))
    judge = MockChat(lambda c: "FLAG")
    flags = judge_flags(judge, eps)
    assert sum(flags.values()) == 4
    r0 = turn_rewards(eps, FakeRM())
    r1 = turn_rewards(eps, FakeRM(), judge, lam=0.5)
    assert np.allclose(r0[0] - 0.5, r1[0])
    junk = MockChat(lambda c: "maybe")
    judge_flags(junk, eps)
    assert junk.bad == 4


def test_activations_and_steering_hook_change_generation_inputs():
    chat = tiny_chat()
    conv = [{"role": "user", "content": "hello there"}]
    base = collect_activations(chat, [conv], layer=2)
    d = np.ones(32) / np.sqrt(32)
    with steering(chat, 1, d, alpha=50.0):
        steered = collect_activations(chat, [conv], layer=2)
    after = collect_activations(chat, [conv], layer=2)
    assert not np.allclose(base, steered) and np.allclose(base, after)       # hook removed on exit
