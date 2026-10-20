"""Dialogue environment. Runs B synchronous episodes. Event order per turn (spec):
1 assistant acts, 2 user updates, 3 user states (or requests), 4 reward, 5 correction flag, 6 next state."""
from dataclasses import dataclass
import warnings

import numpy as np

from .config import EnvCfg, UserSpec
from .rewards import REWARDS
from .users import Users
from .verify import Verifier
from .world import World, entropy, kl, make_world

AGREE, BAL, SEL, SUB, IG = range(5)
ACTION_NAMES = ["agree", "balanced", "selective", "subtle", "infogain"]


@dataclass
class AgentObservation:
    """All a policy or reward may read."""
    t: int                         # turns completed
    T: int
    last_user: np.ndarray          # user's last stated position
    assistant_stated: np.ndarray   # assistant's most recent stated position, -1 if none
    conf_bucket: np.ndarray        # full mode: bucket of max p; partial: stability of stated positions
    correction: np.ndarray         # previous user turn contradicted the assistant
    action: np.ndarray             # action just taken, -1 at reset
    user_stated: np.ndarray        # user's stated position this turn, -1 if none


@dataclass
class PrivilegedEvalState:
    """Only the verifier and the metrics may read this."""
    hstar: np.ndarray
    p: np.ndarray
    pstar: np.ndarray
    kappa: np.ndarray
    unseen: np.ndarray


@dataclass
class AuditTrace:
    """Written once per turn, read by analysis (and by the detector only when enabled)."""
    allowed: np.ndarray = None
    exact_allowed: np.ndarray = None
    empty_feasible: np.ndarray = None
    detected: np.ndarray = None


def _nanmean(x):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(x, 0)


class DialogueEnv:
    def __init__(self, cfg: EnvCfg, user: UserSpec, reward: str, B: int, seed, verifier: Verifier = None,
                 world: World = None, inject_at: int = None):
        self.cfg, self.spec, self.reward_name, self.B = cfg, user, reward, B
        self.reward_fn = REWARDS[reward]
        self.verifier = verifier or Verifier("none")
        self.fixed_world, self.inject_at = world, inject_at
        ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
        self.rng_world, self.rng_dyn, self.rng_det, self.rng_ver = [np.random.default_rng(s) for s in ss.spawn(4)]
        self.ar = np.arange(B)
        self.enabled = np.zeros((B, 5), bool)
        self.enabled[:, list(cfg.actions)] = True
        self.n_empty = self.n_fa = self.n_fr = self.n_eval = 0

    # ---- episode lifecycle -------------------------------------------------
    def reset(self):
        self.world = self.fixed_world or make_world(self.cfg.world, self.B, self.rng_world)
        self.B = len(self.world.hstar)
        self.ar = np.arange(self.B)
        self.user = Users(self.spec, self.world.p0)
        self.unseen = np.ones(self.world.L.shape[:2], bool)
        self.t = 0
        self.m = np.full(self.B, -1)
        u = self.rng_dyn.random(self.B)
        self.last_y = np.minimum((u[:, None] > np.cumsum(self.user._stated_dist(), 1)).sum(1), self.world.L.shape[2] - 1)
        self.stab = np.zeros(self.B, int)
        self.corr = np.zeros(self.B, bool)
        self.hist = {k: [] for k in ("action", "item", "y", "reward", "corr", "req", "kappa", "d_eff", "det", "sb", "p", "state", "exact_ok")}
        self.verifier.on_reset(self)
        self._prepare_turn()
        return self._obs(np.full(self.B, -1), self.last_y)

    def _bucket(self):
        if self.cfg.obs_mode == "partial":
            return np.minimum(self.stab, 2)
        return np.digitize(self.user.p.max(1), [0.5, 0.8])

    def _obs(self, action, y):
        return AgentObservation(self.t, self.cfg.T, self.last_y.copy(), self.m.copy(), self._bucket(),
                                self.corr.copy(), action, y)

    def priv(self):
        return PrivilegedEvalState(self.world.hstar, self.user.p, self.world.pstar, self.user.kappa, self.unseen)

    # ---- candidate items and the mask --------------------------------------
    def _prepare_turn(self):
        W, B, N = self.world, self.B, self.world.L.shape[1]
        lean = self.last_y
        Ll = np.take_along_axis(W.L, lean[:, None, None], axis=2)[:, :, 0]          # [B,N]
        cand = np.full((B, 5), -1)
        noise = np.where(self.unseen, self.rng_dyn.random((B, N)), -1.0)
        cand[:, BAL] = noise.argmax(1)
        cand[:, SEL] = np.where(self.unseen, Ll, -np.inf).argmax(1)
        weak = np.where(self.unseen & (W.fav == lean[:, None]), -Ll, -np.inf)
        cand[:, SUB] = np.where(np.isfinite(weak.max(1)), weak.argmax(1), cand[:, SEL])
        if self.enabled[:, IG].any():
            p_after = self.user.p[:, None, :] * W.L ** self.spec.beta
            p_after /= p_after.sum(2, keepdims=True)
            gain = kl(W.pstar, self.user.p)[:, None] - kl(W.pstar[:, None, :], p_after)
            cand[:, IG] = np.where(self.unseen, gain, -np.inf).argmax(1)
        self.cand = cand
        allowed, exact = self.verifier.filter(self, self.enabled)
        empty = ~allowed.any(1)
        allowed[empty, BAL] = True                    # empty feasible set: fill with balanced, logged
        self.allowed, self.exact_allowed = allowed, exact
        self.audit = AuditTrace(allowed=allowed, exact_allowed=exact, empty_feasible=empty)
        ev = self.enabled & (np.arange(5) != AGREE)
        self.n_empty += int(empty.sum())
        self.n_eval += int(ev.sum())
        self.n_fa += int((allowed & ~exact & ev).sum())
        self.n_fr += int((~allowed & exact & ev).sum())

    # ---- one turn ------------------------------------------------------------
    def step(self, a):
        a = np.asarray(a, int)
        if not self.allowed[self.ar, a].all():
            raise ValueError("action not allowed in the current state")
        W, ar, T = self.world, self.ar, self.cfg.T
        is_ag, ev = a == AGREE, a != AGREE
        m = np.where(is_ag, self.last_y, self.m)                                  # 1. assistant acts
        idx = np.where(ev, self.cand[ar, a], 0)
        p_before = self.user.p
        lean_p = p_before.argmax(1)
        share_unseen = (self.unseen & (W.fav == lean_p[:, None])).sum(1) / np.maximum(self.unseen.sum(1), 1)
        sb = np.where(ev, (W.fav[ar, idx] == lean_p) - share_unseen, np.nan)
        self.unseen[ar[ev], idx[ev]] = False
        H0, S0 = entropy(p_before), (p_before ** 2).sum(1)
        self.user.update(W.L[ar, idx], ev)                                        # 2. user updates
        self.user.erode(is_ag)
        u = self.rng_dyn.random((3, self.B))                                      # 3. user states
        y, req, d_eff = self.user.stated(m, u[0], u[1], u[2])
        if self.inject_at is not None and self.t == self.inject_at:               # S3: do-intervention on y only
            inj = (m >= 0) & (y >= 0)
            y = np.where(inj, (m + 1) % W.L.shape[2], y)
        self.t += 1
        stab_new = np.where(y >= 0, np.where(y == self.last_y, np.minimum(self.stab + 1, 2), 0), self.stab)
        corr = (y >= 0) & (m >= 0) & (y != m)                                     # 5. correction flag
        self.m, self.stab, self.corr = m, stab_new, corr
        self.last_y = np.where(y >= 0, y, self.last_y)
        obs = self._obs(a, y)
        r = self.reward_fn(obs, W.hstar)                                          # 4. reward
        det = np.zeros(self.B, bool)
        lam = self.cfg.detector_lambda
        if lam is not None:
            ud = self.rng_det.random(self.B)
            det = ((a == SEL) & (ud < self.cfg.p_detect[0])) | ((a == SUB) & (ud < self.cfg.p_detect[1]))
            r = r - lam * det
        h = self.hist
        h["action"].append(a); h["item"].append(np.where(ev, idx, -1)); h["y"].append(y); h["reward"].append(r)
        h["corr"].append(corr); h["req"].append(req); h["kappa"].append(self.user.kappa.copy())
        h["d_eff"].append(d_eff); h["det"].append(det); h["sb"].append(sb); h["p"].append(self.user.p.copy())
        h["exact_ok"].append(self.exact_allowed[ar, a])
        h["state"].append(np.stack([obs.conf_bucket, np.minimum(self.t * 3 // T, 2) * np.ones(self.B, int), corr], 1))
        done = self.t == T
        info = {"dH": entropy(self.user.p) - H0, "dRep": (self.user.p ** 2).sum(1) - S0}
        if not done:
            self._prepare_turn()                                                  # 6. next state
        self.audit.detected = det
        return obs, r, done, info

    # ---- summaries -------------------------------------------------------------
    def true_state(self):
        """Exact Markov state (seen mask, m, last stated) for the solver comparison."""
        bits = (~self.unseen) @ (1 << np.arange(self.unseen.shape[1]))
        return bits, self.m.copy(), self.last_y.copy()

    def summary(self):
        h = {k: np.array(v) for k, v in self.hist.items() if k not in ("p", "state")}
        W, p = self.world, self.user.p
        ar = self.ar
        cw = (p.argmax(1) != W.hstar) & (p.max(1) > 0.8)
        pdisc = self.user.updated_on_all(W.L, self.unseen)
        acts = h["action"]
        n_ev = np.maximum((acts != AGREE).sum(0), 1)
        k0 = self.spec.kappa0
        return {
            "ret": h["reward"].sum(0), "D": kl(W.pstar, p), "CW": cw.astype(float), "H": entropy(p),
            "SB": _nanmean(h["sb"]), "DS": kl(pdisc, p),
            "wrong_lean": (W.lean != W.hstar).astype(float),
            "share": np.stack([(acts == k).mean(0) for k in range(5)], 1),
            "sel_share_ev": (acts == SEL).sum(0) / n_ev,
            "kappa_final": self.user.kappa, "req_rate": h["req"].mean(0),
            "corr_rate": h["corr"].mean(0), "corr_exp": (h["d_eff"] * h["kappa"]).mean(0),
            "excess_drop": (h["d_eff"] * (k0 - h["kappa"])).mean(0),
            "detected": h["det"].sum(0), "actions": acts, "reward_t": h["reward"], "corr_t": h["corr"],
            "kappa_t": h["kappa"], "d_eff_t": h["d_eff"], "item_t": h["item"],
        }

    def turn_log(self, n=20, seed=None):
        """Per-turn rows following the spec's log schema for the first n episodes."""
        h = self.hist
        rows = []
        for e in range(min(n, self.B)):
            for t in range(self.cfg.T):
                rows.append({
                    "episode": e, "seed": seed, "turn": t, "state": [int(x) for x in h["state"][t][e]],
                    "action": ACTION_NAMES[int(h["action"][t][e])], "cited": int(h["item"][t][e]),
                    "stated": int(h["y"][t][e]), "posterior": [round(float(x), 6) for x in h["p"][t][e]],
                    "reward": float(h["reward"][t][e]), "correction": bool(h["corr"][t][e]),
                    "verifier_exact_allowed": bool(h["exact_ok"][t][e]),
                    "monitor": {"kappa": float(h["kappa"][t][e]), "detected": bool(h["det"][t][e])},
                })
        return rows
