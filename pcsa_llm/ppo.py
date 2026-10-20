"""Turn-level PPO with a value head and LoRA, for L3 (arms A/B/C), L5 (filtered policy) and L6 (judge penalty).
No maintained library offers PPO with a critic and per-turn multi-turn rewards (TRL 1.16 ships GRPO/RLOO/DPO only),
so this is written here; the math lives in small pure functions that the unit tests check directly.

Arms:  A = single-turn episodes, each response rewarded alone.  B = multi-turn, gamma = 0.  C = multi-turn, gamma = 0.95."""
from dataclasses import dataclass
import numpy as np


# ---- pure math ---------------------------------------------------------------------------------------------
def gae(rewards, values, gamma, lam):
    """Turn-level GAE for one episode. rewards, values: length-T arrays (no value after the last turn)."""
    T = len(rewards)
    adv, nxt_adv, nxt_v = np.zeros(T), 0.0, 0.0
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * nxt_v - values[t]
        nxt_adv = delta + gamma * lam * nxt_adv
        adv[t], nxt_v = nxt_adv, values[t]
    return adv, adv + np.asarray(values)


def discounted_returns(rewards, gamma):
    out, g = np.zeros(len(rewards)), 0.0
    for t in reversed(range(len(rewards))):
        g = rewards[t] + gamma * g
        out[t] = g
    return out


def filtered_logp_surrogate(logp_a, alt_logps, weights=None):
    """L5. Under the filtered policy an accepted sample has log-probability log pi(a|s) - log A(s), A(s) = P(accept | s).
    A(s) depends on the policy, and grad log A = E[1{accept} grad log pi] / A, which the n draws in the state estimate by the
    mean of grad log pi over the ACCEPTED draws. So the surrogate is log pi(a) - mean_accepted log pi(a_i).
    (Subtracting a constant log A, as in a naive estimate, cancels in the PPO ratio and would drop this term.)
    logp_a: scalar tensor; alt_logps: 1-D tensor of the accepted draws' sequence log-probabilities."""
    if weights is None:
        return logp_a - alt_logps.mean()
    return logp_a - (weights * alt_logps).sum() / weights.sum()


@dataclass
class PPOCfg:
    arm: str = "C"               # A | B | C
    gamma: float = None          # default by arm
    lam: float = 0.95
    lr: float = 1e-5
    clip: float = 0.2
    kl_coef: float = 0.02
    vf_coef: float = 0.5
    epochs: int = 2
    minibatch: int = 4
    lora_r: int = 16
    lora_alpha: int = 32
    max_grad_norm: float = 1.0

    def __post_init__(self):
        if self.gamma is None:
            self.gamma = {"A": 0.0, "B": 0.0, "C": 0.95}[self.arm]


def ppo_token_loss(logp_new, logp_old, adv, mask, clip):
    """Clipped surrogate averaged over response tokens (torch tensors, [n_tokens])."""
    import torch
    ratio = torch.exp(logp_new - logp_old)
    s = -torch.minimum(ratio * adv, torch.clamp(ratio, 1 - clip, 1 + clip) * adv)
    return (s * mask).sum() / mask.sum().clamp(min=1)


# ---- trainer ------------------------------------------------------------------------------------------------
class TurnPPO:
    """policy: HFChat whose .model is the base LM. A LoRA adapter and a value head are attached here."""

    def __init__(self, chat, cfg=PPOCfg()):
        import torch
        from peft import LoraConfig, get_peft_model
        self.torch, self.cfg, self.chat = torch, cfg, chat
        base = chat.model
        self.model = get_peft_model(base, LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, target_modules="all-linear",
                                                     lora_dropout=0.0, task_type="CAUSAL_LM"))
        chat.model = self.model
        hidden = base.config.hidden_size if hasattr(base.config, "hidden_size") else base.config.n_embd
        self.value_head = torch.nn.Linear(hidden, 1).to(chat.device).float()
        params = [p for p in self.model.parameters() if p.requires_grad] + list(self.value_head.parameters())
        self.opt = torch.optim.AdamW(params, lr=cfg.lr)
        self.params = params

    # -- forward helpers
    def _logprobs(self, p_ids, r_ids, grad):
        torch = self.torch
        ids = torch.tensor([p_ids + r_ids], device=self.chat.device)
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            out = self.model(input_ids=ids, output_hidden_states=True)
            logits = out.logits[0, len(p_ids) - 1:-1].float()
            logp = torch.log_softmax(logits, -1).gather(-1, ids[0, len(p_ids):, None])[:, 0]
            h = out.hidden_states[-1][0, len(p_ids) - 1].float()
        return logp, h

    def _value(self, h):
        return self.value_head(h.detach())[0]

    def update(self, samples):
        """samples: dicts with p_ids, r_ids, adv, ret, trainable (False for L5 fallback turns). Returns diagnostics."""
        torch, cfg = self.torch, self.cfg
        advs = np.array([s["adv"] for s in samples])
        norm = (advs - advs.mean()) / (advs.std() + 1e-8) if len(advs) > 1 else advs
        for s, a in zip(samples, norm):
            s["adv_n"] = float(a)
            lp, _ = self._logprobs(s["p_ids"], s["r_ids"], grad=False)
            s["logp_old"] = lp
            if s.get("logp_adj") is not None:
                s["logp_old"] = lp + s["logp_adj"]
            with self.model.disable_adapter():
                s["logp_ref"], _ = self._logprobs(s["p_ids"], s["r_ids"], grad=False)
        diag = {"loss": [], "kl": [], "vloss": []}
        for _ in range(cfg.epochs):
            order = np.random.permutation(len(samples))
            for i in range(0, len(order), cfg.minibatch):
                self.opt.zero_grad()
                batch = [samples[j] for j in order[i:i + cfg.minibatch]]
                for s in batch:
                    lp, h = self._logprobs(s["p_ids"], s["r_ids"], grad=True)
                    new = lp + s["logp_adj"] if s.get("logp_adj") is not None else lp
                    mask = torch.ones_like(lp) * (1.0 if s.get("trainable", True) else 0.0)
                    pol = ppo_token_loss(new, s["logp_old"], torch.tensor(s["adv_n"], device=lp.device), mask, cfg.clip)
                    kl = ((s["logp_ref"] - lp) + torch.exp(lp - s["logp_ref"]) - 1.0).mean()      # k3 estimator, >= 0
                    if s.get("alts") and s.get("trainable", True):         # L5: gradient of -log A(s) through the accepted draws
                        alt = torch.stack([self._logprobs(s["p_ids"], r, grad=True)[0].sum() for r in s["alts"]])
                        pol = pol + s["adv_n"] * alt.mean()
                    v = self._value(h)
                    vloss = (v - torch.tensor(float(s["ret"]), device=v.device)) ** 2
                    loss = (pol + cfg.kl_coef * kl + cfg.vf_coef * vloss) / len(batch)
                    loss.backward()
                    diag["loss"].append(float(pol.detach())); diag["kl"].append(float(kl.detach())); diag["vloss"].append(float(vloss.detach()))
                torch.nn.utils.clip_grad_norm_(self.params, cfg.max_grad_norm)
                self.opt.step()
        return {k: float(np.mean(v)) for k, v in diag.items()}

    def values(self, samples):
        """Critic value of each sample's state (prompt end), no grad."""
        out = []
        for s in samples:
            _, h = self._logprobs(s["p_ids"], s["r_ids"][:1] or [0], grad=False)
            with self.torch.no_grad():
                out.append(float(self._value(h)))
        return out
