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


def _tiny_rm():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=258, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
                                   max_position_embeddings=8192, num_labels=1)
    assert cfg.pad_token_id is None                                           # the failing case from the first real run
    return transformers.LlamaForSequenceClassification(cfg).float()


def _convs():
    return [[{"role": "user", "content": "a" * n}, {"role": "assistant", "content": "ok"}] for n in (5, 40, 17)]


def test_reward_model_without_a_pad_token_scores_batches_like_single_conversations():
    from pcsa_llm.dry import ByteTok
    from pcsa_llm.models import HFRewardModel

    class NoPadTok(ByteTok):
        pad_token, pad_token_id = None, None

        def get_vocab(self):
            return {}

    rm = HFRewardModel("tiny", device="cpu", model=_tiny_rm(), tokenizer=NoPadTok())
    assert rm.batch_size == 1                                                  # no usable pad token -> no padding
    batched = rm.score(_convs())
    single = [rm.score([c])[0] for c in _convs()]
    assert batched == pytest.approx(single, abs=1e-5) and len(set(np.round(batched, 5))) == 3


def test_reward_model_with_a_distinct_pad_token_batches_without_changing_scores():
    from pcsa_llm.dry import ByteTok
    from pcsa_llm.models import HFRewardModel
    rm = HFRewardModel("tiny", device="cpu", model=_tiny_rm(), tokenizer=ByteTok())      # pad id 0, eos id 1: distinct
    assert rm.batch_size == 8 and rm.model.config.pad_token_id == 0
    assert rm.score(_convs()) == pytest.approx(rm.score(_convs(), batch_size=1), abs=1e-4)


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


def test_gradient_checkpointing_gives_the_same_update_as_without():
    from pcsa_llm.l_train import build_samples
    outs = []
    for ckpt in (False, True):
        chat = tiny_chat()
        ppo = TurnPPO(chat, PPOCfg(arm="C", lr=1e-2, epochs=1, minibatch=2, grad_ckpt=ckpt))
        torch.manual_seed(0); np.random.seed(0)
        eps = run_dialogues(chat, MockChat(bayes_user), QS[:3], D, seed=0)
        samples = build_samples(ppo, eps, turn_rewards(eps, FakeRM()))
        diag = ppo.update(samples)
        outs.append((diag, torch.cat([p.detach().flatten() for n, p in ppo.model.named_parameters() if "lora" in n])))
        assert not ppo.model.training                                          # back in eval mode afterwards
    assert outs[0][0]["loss"] == pytest.approx(outs[1][0]["loss"], rel=1e-4, abs=1e-6)
    assert torch.allclose(outs[0][1], outs[1][1], atol=1e-5)


def test_memory_plan_by_gpu_size():
    from pcsa_llm.loaders import memory_plan
    assert memory_plan(96)["name"] == "large" and not memory_plan(96)["grad_ckpt"]
    p80 = memory_plan(79.6)                                                   # what an 80 GB A100 reports
    assert p80["name"] == "a100_80" and p80["aux_quant"] is None and p80["grad_ckpt"]
    p40 = memory_plan(40)
    assert p40["aux_quant"] == "4bit" and p40["grad_ckpt"] and p40["gen_batch"] < p80["gen_batch"]


def test_policy_and_critic_are_clipped_separately_so_critic_error_cannot_shrink_policy_gradients(monkeypatch):
    """Regression for the first real L3 log: one joint clip let a large critic error (reward scale ~ -8) scale down the LoRA gradients."""
    from pcsa_llm.l_train import build_samples
    chat = tiny_chat()
    ppo = TurnPPO(chat, PPOCfg(arm="C", lr=1e-3, epochs=1, minibatch=2))
    groups = []
    real = torch.nn.utils.clip_grad_norm_
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", lambda params, m: (groups.append({id(p) for p in params}), real(params, m))[1])
    eps = run_dialogues(chat, MockChat(bayes_user), QS[:3], D, seed=0)
    samples = build_samples(ppo, eps, turn_rewards(eps, FakeRM()))
    for s in samples:
        s["ret"] = -1e4                                                       # enormous critic error
    ppo.update(samples)
    pol, val = {id(p) for p in ppo.policy_params}, {id(p) for p in ppo.value_params}
    assert groups and all(g == pol or g == val for g in groups)               # never a mixture
    assert any(g == pol for g in groups) and any(g == val for g in groups)
    lrs = {g["lr"] for g in ppo.opt.param_groups}
    assert lrs == {1e-3, ppo.cfg.vf_lr}


def test_train_logs_diagnostics_saves_history_each_iteration_and_stops_on_time_budget(tmp_path):
    import json
    chat = tiny_chat()
    ppo = TurnPPO(chat, PPOCfg(arm="B", lr=1e-3, epochs=1, minibatch=2))
    lines = []
    f = tmp_path / "h.jsonl"
    hist = train(ppo, MockChat(bayes_user), FakeRM(), QS, DialogueCfg(T=2, max_new=6, temperature=1.0), iters=5, per_iter=2,
                 log=lines.append, jsonl=str(f), max_minutes=1e-9, samples_every=1)
    assert len(hist) == 1 and hist[0]["stopped"].startswith("time budget")           # stopped after the first iteration, with the reason
    assert len(f.read_text().splitlines()) == 1 and "diag" in json.loads(f.read_text())
    assert any("ETA" in l for l in lines) and any("sample [" in l for l in lines)


def test_rehearsal_projects_time_without_a_gpu_and_runs_every_arm():
    from pcsa_llm.loaders import Loaders, memory_plan
    from pcsa_llm.runs import rehearsal
    L = Loaders({"l3_policy": "dry", "stages": {}, "rms": {}}, dry=True, plan=memory_plan(1000))
    out = rehearsal(L, FakeRM(), MockChat(bayes_user), QS, DialogueCfg(T=2, max_new=6, temperature=1.0), plan_iters=40, plan_per_iter=16,
                    iters=1, per_iter=2, log=lambda *_: None)
    assert set(out) == {"A", "B", "C"} and all(v["projected_minutes"] > 0 and v["peak_gpu_gb"] is None for v in out.values())
    assert out["A"]["projected_minutes"] == pytest.approx(out["A"]["sec_per_iter_at_rehearsal_size"] * 40 * 8 / 60)


def test_training_resumes_from_a_checkpoint_and_continues_the_history(tmp_path):
    import json
    ck, jl = str(tmp_path / "a.ckpt"), str(tmp_path / "a.jsonl")
    mk = lambda: TurnPPO(tiny_chat(), PPOCfg(arm="B", lr=1e-3, epochs=1, minibatch=2))
    kw = dict(user=MockChat(bayes_user), rm=FakeRM(), questions=QS, dcfg=DialogueCfg(T=2, max_new=6, temperature=1.0), per_iter=2,
              log=lambda *_: None, samples_every=0, ckpt=ck, ckpt_every=1, jsonl=jl)
    ppo1 = mk()
    h1 = train(ppo1, iters=2, **kw)                                          # "interrupted" after 2 of 4 iterations
    assert [h["iter"] for h in h1] == [0, 1]
    w_saved = {n: p.detach().clone() for n, p in ppo1.model.named_parameters() if "lora" in n}
    ppo2 = mk()
    lines = []
    kw2 = dict(kw, log=lines.append)
    h2 = train(ppo2, iters=4, **kw2)                                         # fresh process: resumes from the checkpoint
    assert any("RESUMED" in l for l in lines)
    assert [h["iter"] for h in h2] == [0, 1, 2, 3]                           # continues, does not restart
    assert all(h2[i]["reward"] == h1[i]["reward"] for i in (0, 1))           # earlier history is kept as it was
    rows = [json.loads(l)["iter"] for l in open(jl)]
    assert rows == [0, 1, 2, 3]                                              # jsonl has no duplicates after the resume
    # the weights loaded at resume are the saved ones: before the first resumed update, they matched
    ppo3 = mk(); last, _ = ppo3.load_checkpoint(ck)
    assert last == 3


def test_checkpoint_restores_adapter_value_head_and_optimizer(tmp_path):
    ck = str(tmp_path / "b.ckpt")
    a = TurnPPO(tiny_chat(), PPOCfg(arm="C", lr=1e-2, epochs=1, minibatch=2))
    eps = run_dialogues(a.chat, MockChat(bayes_user), QS[:3], D, seed=0)
    from pcsa_llm.l_train import build_samples
    a.update(build_samples(a, eps, turn_rewards(eps, FakeRM())))
    a.save_checkpoint(ck, 7, [{"x": 1}])
    b = TurnPPO(tiny_chat(), PPOCfg(arm="C", lr=1e-2, epochs=1, minibatch=2))
    last, hist = b.load_checkpoint(ck)
    assert last == 7 and hist == [{"x": 1}]
    for (n, pa), (_, pb) in zip(a.model.named_parameters(), b.model.named_parameters()):
        if "lora" in n:
            assert torch.equal(pa.detach(), pb.detach())
    assert torch.equal(a.value_head.weight, b.value_head.weight)
    assert len(b.opt.state_dict()["state"]) == len(a.opt.state_dict()["state"]) > 0


def test_environment_smoke_check_passes_here_and_explains_the_torchao_failure(monkeypatch):
    import peft
    from pcsa_llm import envcheck
    assert envcheck.smoke().startswith("environment ok")
    def broken(*a, **k):
        raise ImportError("Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported")
    monkeypatch.setattr(peft, "get_peft_model", broken)
    with pytest.raises(RuntimeError) as e:
        envcheck.smoke()
    assert "pip uninstall -y torchao" in str(e.value) and "restart" in str(e.value).lower()
