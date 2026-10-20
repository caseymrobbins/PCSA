"""Scripted policies and the tabular Q-learner. Policies read only AgentObservation (+ the allowed mask)."""
import numpy as np
from .env import AGREE, BAL, SEL, IG

N_STATE, N_ACT = 18, 5


def argmax_random(q, allowed, rng):
    """Row-wise argmax over allowed actions; ties broken uniformly at random."""
    qm = np.where(allowed, q, -np.inf)
    ties = qm == qm.max(1, keepdims=True)
    return (ties * rng.random(q.shape)).argmax(1)


def state_id(obs):
    """(confidence bucket, phase, last turn was a correction) -> 0..17"""
    phase = np.minimum(obs.t * 3 // obs.T, 2)
    return obs.conf_bucket * 6 + phase * 2 + obs.correction.astype(int)


def scripted(name, T=15):
    """always-agree | always-balanced | always-selective | {sel,bal,ig}<k>_agree"""
    if name == "agree":
        seq = [AGREE] * T
    elif name == "balanced":
        seq = [BAL] * T
    elif name == "selective":
        seq = [SEL] * T
    else:
        head, tail = name.split("_")
        assert tail == "agree"
        kind = head.rstrip("0123456789")
        k = int(head[len(kind):])
        seq = [{"sel": SEL, "bal": BAL, "ig": IG}[kind]] * k + [AGREE] * (T - k)

    def policy(obs, allowed, rng):
        return np.full(len(obs.last_user), seq[obs.t], int)

    policy.name = name
    return policy


class TabularQ:
    def __init__(self, gamma, alpha=0.01):
        self.gamma, self.alpha = gamma, alpha
        self.Q = np.zeros((N_STATE, N_ACT))
        self.Rsum, self.Rn = np.zeros((N_STATE, N_ACT)), np.zeros((N_STATE, N_ACT))
        self.dH, self.dRep = np.zeros((N_STATE, N_ACT)), np.zeros((N_STATE, N_ACT))
        self.visits = np.zeros((N_STATE, N_ACT))

    def act(self, s, allowed, eps, rng):
        greedy = argmax_random(self.Q[s], allowed, rng)
        rand = argmax_random(rng.random((len(s), N_ACT)), allowed, rng)
        return np.where(rng.random(len(s)) < eps, rand, greedy)

    def greedy(self, obs, allowed, rng):
        return argmax_random(self.Q[state_id(obs)], allowed, rng)

    def update(self, s, a, r, s2, allowed2, done, dH, dRep):
        # the max in the target runs over allowed actions only (masked actions are not bootstrapped)
        nxt = np.where(allowed2, self.Q[s2], -np.inf).max(1)
        td = r + self.gamma * np.where(done, 0.0, nxt) - self.Q[s, a]
        tot, cnt = np.zeros_like(self.Q), np.zeros_like(self.Q)
        np.add.at(tot, (s, a), td)
        np.add.at(cnt, (s, a), 1)
        # a cell hit n times in the batch gets the step of n sequential updates against the same target
        self.Q += (1 - (1 - self.alpha) ** cnt) * tot / np.maximum(cnt, 1)
        for arr, v in ((self.Rsum, r), (self.dH, dH), (self.dRep, dRep), (self.Rn, 1.0)):
            np.add.at(arr, (s, a), v)
        self.visits += cnt


def train_q(env, n_episodes, gamma, seed, alpha=0.01, eps=(1.0, 0.05)):
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    q = TabularQ(gamma, alpha)
    nb = max(1, n_episodes // env.B)
    for b in range(nb):
        e = eps[0] + (eps[1] - eps[0]) * b / max(nb - 1, 1)
        s = state_id(env.reset())
        while True:
            a = q.act(s, env.allowed, e, rng)
            obs, r, done, info = env.step(a)
            s2 = state_id(obs)
            q.update(s, a, r, s2, env.allowed, done, info["dH"], info["dRep"])
            s = s2
            if done:
                break
    return q


def rollout(env, policy, seed):
    """Run one batch of episodes under `policy`; returns env.summary()."""
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
    obs = env.reset()
    done = False
    while not done:
        obs, _, done, _ = env.step(policy(obs, env.allowed, rng))
    return env.summary()
