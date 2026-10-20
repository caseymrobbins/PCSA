"""Exact dynamic programming on a small, fixed instance (Bayes/weighted user; order-free belief).
State = (t, seen mask, assistant's stated position m, user's last stated position). Actions: agree, balanced, selective."""
import numpy as np
from .env import AGREE, AgentObservation
from .rewards import REWARDS


def reward_table(kind, K, T, hstar):
    """R[is_agree, is_last, y+1, m+1], built by calling the shared reward function once on all combinations."""
    combos = [(ag, last, y, m) for ag in (0, 1) for last in (0, 1) for y in range(-1, K) for m in range(-1, K)]
    ag, last, y, m = (np.array(c) for c in zip(*combos))
    n = len(combos)
    out = np.zeros((2, 2, K + 1, K + 1))
    for lst in (0, 1):
        sel = last == lst
        obs = AgentObservation(T if lst else 0, T, np.zeros(sel.sum(), int), m[sel], np.zeros(sel.sum(), int),
                               np.zeros(sel.sum(), bool), np.where(ag[sel] == 1, AGREE, 1), y[sel])
        r = REWARDS[kind](obs, hstar)
        for (a_, _, y_, m_), r_ in zip([c for c in combos if c[1] == lst], r):
            out[a_, lst, y_ + 1, m_ + 1] = r_
    return out


class ExactSolver:
    def __init__(self, L, p0, hstar, T, gamma, reward, beta=1.0, tau=1.0):
        self.L, self.p0, self.T, self.gamma, self.beta, self.tau = L, p0, T, gamma, beta, tau
        self.N, self.K = L.shape
        self.R = reward_table(reward, self.K, T, hstar)
        self._post, self._q = {0: p0}, {}

    def post(self, mask):
        if mask not in self._post:
            low = mask & -mask
            i = low.bit_length() - 1
            q = self.post(mask ^ low) * self.L[i] ** self.beta
            self._post[mask] = q / q.sum()
        return self._post[mask]

    def _w(self, p):
        w = p ** (1 / self.tau)
        return w / w.sum()

    def sel_item(self, mask, yl):
        free = [i for i in range(self.N) if not mask >> i & 1]
        return max(free, key=lambda i: self.L[i, yl])

    def Q(self, t, mask, m, yl):
        key = (t, mask, m, yl)
        if key not in self._q:
            last = int(t + 1 == self.T)
            free = [i for i in range(self.N) if not mask >> i & 1]
            qa = self._branch(t, mask, m, yl, last, agree=True, item=None)
            qs = self._branch(t, mask, m, yl, last, False, self.sel_item(mask, yl))
            qb = float(np.mean([self._branch(t, mask, m, yl, last, False, i) for i in free]))
            self._q[key] = np.array([qa, qb, qs])
        return self._q[key]

    def _branch(self, t, mask, m, yl, last, agree, item):
        m2 = yl if agree else m
        mask2 = mask if agree else mask | 1 << item
        w = self._w(self.post(mask2))
        tot = 0.0
        for y in range(self.K):
            fut = 0.0 if last else self.gamma * self.Q(t + 1, mask2, m2, y).max()
            tot += w[y] * (self.R[int(agree), last, y + 1, m2 + 1] + fut)
        return tot
