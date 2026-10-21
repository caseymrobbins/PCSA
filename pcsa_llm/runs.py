"""Higher-level runs used by the notebook: one PPO arm, evaluation, critic report, probe dataset, steering effect."""
import numpy as np

from .dialogue import DialogueCfg, run_dialogues
from .l_train import train
from .metrics import episode_metrics, summarize
from .ppo import PPOCfg, TurnPPO
from .probes import collect_activations, steering


def evaluate(chat, user, questions, dcfg, seeds):
    """-> (list of per-seed summaries, episodes of the last seed)."""
    sums, eps = [], None
    for s in seeds:
        eps = run_dialogues(chat, user, questions, dcfg, seed=1000 + s)
        sums.append(summarize(eps))
    return sums, eps


def critic_report(ppo, eps):
    """Critic value by user state: settled (max private p > 0.8) vs not, and the turn after a correction vs not."""
    rows = []
    for e in eps:
        for i, t in enumerate(e.turns):
            if t["tokens"] is None:
                continue
            rows.append((ppo.values([{"p_ids": t["tokens"][0], "r_ids": t["tokens"][1]}])[0], float(t["p_before"].max() > 0.8),
                         float(i > 0 and e.turns[i - 1]["correction"])))
    if not rows:
        return {}
    v, settled, after = (np.array(x) for x in zip(*rows))
    m = lambda mask: float(v[mask].mean()) if mask.any() else float("nan")
    return {"value_settled": m(settled == 1), "value_unsettled": m(settled == 0), "value_after_correction": m(after == 1),
            "value_no_correction": m(after == 0), "n": len(rows)}


def run_arm(loaders, arm, seed, rm, user, train_q, eval_q, dcfg, iters, per_iter, eval_seeds=(0,), step_fn_factory=None,
            judge=None, lam=0.0, save_dir=None, tag=None, log=print, guards=None, max_minutes=None, ckpt=None, ckpt_every=10, jsonl=None):
    """Train one arm (A single-turn, B gamma = 0, C gamma = 0.95) and evaluate it on held-out questions with the full horizon."""
    chat = loaders.policy(loaders.cfg["l3_policy"], seed=seed)
    ppo = TurnPPO(chat, PPOCfg(arm=arm, lr=loaders.cfg.get("lr", 1e-5), grad_ckpt=loaders.plan["grad_ckpt"], minibatch=loaders.plan["minibatch"]))
    tcfg = DialogueCfg(T=1 if arm == "A" else dcfg.T, max_new=dcfg.max_new, temperature=dcfg.temperature, user_retries=dcfg.user_retries)
    from .l_train import DEFAULT_GUARDS
    jsonl = jsonl or (f"{save_dir}/{tag or 'arm'}{arm}_seed{seed}.jsonl" if save_dir else None)
    hist = train(ppo, user, rm, train_q, tcfg, iters, per_iter, seed=seed, step_fn_factory=step_fn_factory, judge=judge, lam=lam, log=log,
                 guards=DEFAULT_GUARDS if guards is None else guards, jsonl=jsonl, max_minutes=max_minutes, ckpt=ckpt, ckpt_every=ckpt_every)
    sums, eps = evaluate(chat, user, eval_q, dcfg, eval_seeds)
    path = None
    if save_dir:
        import torch
        path = f"{save_dir}/{tag or 'arm'}{arm}_seed{seed}"
        ppo.model.save_pretrained(path)
        torch.save(ppo.value_head.state_dict(), f"{path}/value_head.pt")
    import os
    if ckpt and os.path.exists(ckpt):
        os.remove(ckpt)                                  # finished: the saved adapter and the stage result supersede it
    # no model handles are returned, so the caller can free GPU memory before the next arm
    return {"arm": arm, "seed": seed, "history": hist, "eval": sums, "critic": critic_report(ppo, eps), "eval_eps": eps, "path": path}


def load_arm(loaders, path, seed=0):
    """Reload a trained arm for L4: base policy + LoRA adapter + value head. -> (chat, value_fn)."""
    import torch
    chat = loaders.policy(loaders.cfg["l3_policy"], seed=seed)
    from peft import PeftModel
    chat.model = PeftModel.from_pretrained(chat.model, path).eval()
    vh = torch.nn.Linear(chat.model.config.hidden_size, 1).to(chat.device).float()
    vh.load_state_dict(torch.load(f"{path}/value_head.pt", map_location=chat.device))

    def value_fn(conv, text=None):
        """Critic value of the state after the assistant's response (a proxy for continuation value: the user's reply is not yet known)."""
        c = conv if text is None else conv + [{"role": "assistant", "content": text}]
        enc = chat.tok(chat.format(c), return_tensors="pt", add_special_tokens=False).to(chat.device)
        with torch.no_grad():
            h = chat.model(**enc, output_hidden_states=True).hidden_states[-1][0, -1].float()
            return float(vh(h)[0])
    return chat, value_fn


def activation_dataset(chat, eps, layer):
    """Activations at assistant-turn states (conversation up to and including the response), labelled with the LLM user's private
    belief entering the turn: confidence and top answer. Groups are question ids (train/test split by question)."""
    convs, conf, top, groups = [], [], [], []
    for e in eps:
        for t in e.turns:
            convs.append(t["prompt_msgs"] + [{"role": "assistant", "content": t["text"]}])
            conf.append(float(t["p_before"].max())); top.append(int(t["p_before"].argmax())); groups.append(e.q.qid)
    return collect_activations(chat, convs, layer), np.array(conf), np.array(top), np.array(groups)


def steering_effect(chat, user, questions, dcfg, direction, layer, alphas, seed=0):
    """Add alpha * (confidence direction) while the policy generates: negative alpha = it represents the user as less certain."""
    out = {}
    for a in alphas:
        if a == 0:
            eps = run_dialogues(chat, user, questions, dcfg, seed=seed)
        else:
            with steering(chat, layer, direction, a):
                eps = run_dialogues(chat, user, questions, dcfg, seed=seed)
        s = summarize(eps)
        out[a] = {k: s[k] for k in ("SB", "D", "CW", "agree", "selective")}
    return out


def l4_states(eps, n):
    """(conversation, question, last stated position) for the first turns of the first n episodes."""
    out = []
    for e in eps[:n]:
        if e.turns:
            t = e.turns[0]
            out.append((t["prompt_msgs"], e.q, e.q.lean))
    return out


def rehearsal(loaders, rm, user, train_q, dcfg, plan_iters, plan_per_iter, arms="ABC", iters=2, per_iter=8, log=print):
    """Dress rehearsal of the real training loop: a couple of tiny iterations per arm to catch crashes and out-of-memory errors,
    measure seconds per iteration, and project the time of the planned run (linear in dialogues per iteration)."""
    import time
    out = {}
    for arm in arms:
        chat = loaders.policy(loaders.cfg["l3_policy"], seed=0)
        ppo = TurnPPO(chat, PPOCfg(arm=arm, lr=loaders.cfg.get("lr", 1e-5), grad_ckpt=loaders.plan["grad_ckpt"], minibatch=loaders.plan["minibatch"]))
        tcfg = DialogueCfg(T=1 if arm == "A" else dcfg.T, max_new=dcfg.max_new, temperature=dcfg.temperature, user_retries=dcfg.user_retries)
        peak = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            torch = None
        t0 = time.time()
        hist = train(ppo, user, rm, train_q, tcfg, iters, per_iter, seed=0, log=log, guards=None, samples_every=1)
        sec = (time.time() - t0) / iters
        if torch is not None and torch.cuda.is_available():
            peak = torch.cuda.max_memory_allocated() / 1e9
        out[arm] = {"sec_per_iter_at_rehearsal_size": sec, "projected_minutes": sec * plan_iters * (plan_per_iter / per_iter) / 60,
                    "peak_gpu_gb": peak, "cite_rate": hist[-1]["diag"]["cite_rate"], "resp_tokens": hist[-1]["diag"]["resp_tokens"],
                    "kl_after": hist[-1]["diag"]["kl"]}
        del ppo, chat
        loaders.free()
    return out
