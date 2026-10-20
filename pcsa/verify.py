"""Verifier and mask. Reads privileged state by design (spec: both masks test the design's logic)."""
import numpy as np
from .world import kl, normalize

AGREE = 0


class Verifier:
    """kind: none | exact | noisy | subsample. kappa_mask additionally forbids actions that lower kappa.
    sigma is in units of `scale` (median one-step |change in KL|), set by calibrate()."""

    def __init__(self, kind="none", sigma=0.0, m=0, kappa_mask=False):
        self.kind, self.sigma, self.m, self.kappa_mask = kind, sigma, m, kappa_mask
        self.scale = 1.0
        self.pstar_hat = None

    def on_reset(self, env):
        if self.kind == "subsample":
            B, N = env.B, env.world.L.shape[1]
            sub = np.argsort(env.rng_ver.random((B, N)), 1)[:, : self.m]
            Ls = env.world.L[env.ar[:, None], sub]                      # [B,m,K]
            self.pstar_hat = normalize(np.exp(np.log(env.world.p0) + np.log(Ls).sum(1)))

    def calibrate(self, env_factory):
        """scale := median |exact one-step change in KL| over balanced actions in a pilot batch."""
        env = env_factory()
        env.reset()
        d = []
        for _ in range(env.cfg.T):
            a = np.where(env.allowed[:, 1], 1, 0)
            dk = self._dkl(env, env.cand[:, 1], env.world.pstar)
            d.append(np.abs(dk))
            env.step(a)
        self.scale = float(np.median(np.concatenate(d)))

    @staticmethod
    def _dkl(env, items, pstar):
        Lrow = env.world.L[env.ar, np.maximum(items, 0)]
        p_after = env.user.peek(Lrow)
        return kl(pstar, p_after) - kl(pstar, env.user.p)

    def filter(self, env, enabled):
        """Return (allowed, exact_allowed), both [B,5]."""
        B = env.B
        exact = enabled.copy()
        allowed = enabled.copy()
        if self.kind == "none":
            return allowed, exact
        z0 = env.rng_ver.standard_normal(B) if self.kind == "noisy" else None   # error on the current state's KL estimate
        for a in range(1, 5):
            if not enabled[:, a].any():
                continue
            dk = self._dkl(env, env.cand[:, a], env.world.pstar)
            exact[:, a] &= dk <= 1e-12
            if self.kind == "exact":
                allowed[:, a] &= dk <= 1e-12
            elif self.kind == "noisy":
                est = dk + self.sigma * self.scale * (env.rng_ver.standard_normal(B) - z0)
                allowed[:, a] &= est <= 0
            elif self.kind == "subsample":
                allowed[:, a] &= self._dkl(env, env.cand[:, a], self.pstar_hat) <= 1e-12
        if self.kappa_mask and env.user.spec.kind == "deference" and env.user.spec.delta > 0:
            allowed[:, AGREE] = False
            exact[:, AGREE] = False
        return allowed, exact
