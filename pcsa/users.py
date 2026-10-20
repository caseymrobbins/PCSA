"""Simulated users. One vectorised class; the variant is a UserSpec."""
import numpy as np
from .world import entropy


class Users:
    def __init__(self, spec, p0):
        self.spec = spec
        self.p = p0.copy()
        self.K = p0.shape[1]
        self.kappa = np.full(len(p0), spec.kappa0)

    def _beta(self, Lrow, p):
        s = self.spec
        if s.kind == "confirm":
            return np.where(Lrow.argmax(1) != p.argmax(1), s.beta_oppose, s.beta)
        return np.full(len(Lrow), s.beta)

    def peek(self, Lrow, p=None):
        """Posterior the user would hold after seeing Lrow (does not change state)."""
        p = self.p if p is None else p
        q = p * Lrow ** self._beta(Lrow, p)[:, None]
        return q / q.sum(1, keepdims=True)

    def update(self, Lrow, mask):
        self.p = np.where(mask[:, None], self.peek(Lrow), self.p)

    def updated_on_all(self, L, unseen):
        """p_T updated on every unseen item (disclosure probe)."""
        p = self.p
        for i in range(L.shape[1]):
            p = np.where(unseen[:, i, None], self.peek(L[:, i], p), p)
        return p

    def erode(self, agreed):
        if self.spec.kind == "deference":
            self.kappa = np.maximum(0.0, self.kappa - self.spec.delta * agreed)

    def _stated_dist(self):
        w = self.p ** (1.0 / self.spec.tau)
        return w / w.sum(1, keepdims=True)

    def stated(self, m, u_y, u_req, u_corr):
        """Return (y, requested, d_eff). y = -1 when the user requests an item instead of stating.
        d_eff is the probability that the user's belief disagrees with m and it is not a request."""
        w = self._stated_dist()
        y_b = np.minimum((u_y[:, None] > np.cumsum(w, 1)).sum(1), self.K - 1)
        has_m = m >= 0
        ar = np.arange(len(m))
        d = np.where(has_m, 1.0 - w[ar, np.maximum(m, 0)], 0.0)
        if self.spec.kind != "deference":
            return y_b, np.zeros(len(m), bool), d
        q_req = self.spec.req_rate * entropy(self.p) / np.log(self.K)
        req = u_req < q_req
        challenge = u_corr < self.kappa
        y = np.where(has_m & (y_b != m) & ~challenge, m, y_b)   # silent deference
        y = np.where(req, -1, y)
        return y, req, (1.0 - q_req) * d
