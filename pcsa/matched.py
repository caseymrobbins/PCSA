"""S5: matched-return variance. Two action families with equal expected return and different user-response variance.
Family L: the predictable user always gives 0.5. Family H: the unpredictable user gives 1 w.p. 0.5, else 0."""
import numpy as np
from .metrics import boot_ci

H = 5          # turns per episode
LOW, HIGH = 0, 1


def exact_q(gamma=1.0):
    """Exact DP: Q[t, a]. Equal across families by construction; the test checks the environment is built right."""
    Q = np.zeros((H, 2))
    V = np.zeros(H + 1)
    for t in reversed(range(H)):
        Q[t, LOW] = 0.5 + gamma * V[t + 1]
        Q[t, HIGH] = 0.5 + gamma * V[t + 1]
        V[t] = Q[t].max()
    return Q


def _reward(a, rng):
    return 0.5 if a == LOW else float(rng.random() < 0.5)


class Learner:
    """Per-turn value learners on the matched environment. `pref()` = mean P(choose LOW) over turns."""

    def __init__(self, kind, rng, alpha=0.1, eps=0.1):
        self.kind, self.rng, self.alpha, self.eps = kind, rng, alpha, eps
        self.Q = np.zeros((H, 2))
        self.Q2 = np.zeros((H, 2))                  # second table for double Q
        self.w = np.zeros(4)                        # linear: features [1, t/H] per action
        self.theta = np.zeros((H, 2))               # REINFORCE logits
        self.baseline = 0.0

    def _phi(self, t, a):
        f = np.zeros(4)
        f[2 * a:2 * a + 2] = (1.0, t / H)
        return f

    def q(self, t):
        if self.kind == "linear":
            return np.array([self.w @ self._phi(t, a) for a in (0, 1)])
        if self.kind == "double":
            return self.Q[t] + self.Q2[t]
        return self.Q[t]

    def probs(self, t):
        if self.kind == "reinforce":
            e = np.exp(self.theta[t] - self.theta[t].max())
            return e / e.sum()
        q = self.q(t)
        greedy = (q == q.max()) / (q == q.max()).sum()
        return self.eps / 2 + (1 - self.eps) * greedy

    def pref(self):
        """Greedy preference for the low-variance family (ties count 1/2); softmax prob for REINFORCE."""
        if self.kind == "reinforce":
            return float(np.mean([self.probs(t)[LOW] for t in range(H)]))
        return float(np.mean([(self.q(t)[LOW] > self.q(t)[HIGH]) + 0.5 * (self.q(t)[LOW] == self.q(t)[HIGH]) for t in range(H)]))

    def episode(self):
        traj = []
        for t in range(H):
            a = int(self.rng.choice(2, p=self.probs(t)))
            traj.append((t, a, _reward(a, self.rng)))
        if self.kind == "reinforce":
            G = 0.0
            for t, a, r in reversed(traj):
                G += r
                p = self.probs(t)
                g = -p
                g[a] += 1.0
                self.theta[t] += 0.05 * (G - self.baseline) * g
                self.baseline += 0.001 * (G - self.baseline)
            return
        for i, (t, a, r) in enumerate(traj):
            last = t == H - 1
            if self.kind == "double":
                A, B = (self.Q, self.Q2) if self.rng.random() < 0.5 else (self.Q2, self.Q)
                nxt = 0.0 if last else B[t + 1, A[t + 1].argmax()]
                A[t, a] += self.alpha * (r + nxt - A[t, a])
            elif self.kind == "linear":
                nxt = 0.0 if last else self.q(t + 1).max()
                self.w += 0.05 * (r + nxt - self.w @ self._phi(t, a)) * self._phi(t, a)
            else:                                    # "plain" and "risk"
                nxt = 0.0 if last else self.Q[t + 1].max()
                td = r + nxt - self.Q[t, a]
                if self.kind == "risk":              # Mihatsch-Neuneier: weight bad surprises more (risk-averse control)
                    td *= 0.5 if td > 0 else 1.5
                self.Q[t, a] += self.alpha * td


KINDS = ("plain", "double", "linear", "reinforce", "risk")


def run(kind, seed, episodes=20000, checkpoints=20):
    rng = np.random.default_rng(seed)
    lr = Learner(kind, rng)
    prefs = []
    for e in range(1, episodes + 1):
        lr.episode()
        if e % (episodes // checkpoints) == 0:
            prefs.append(lr.pref())
    return np.array(prefs)


def s5(seeds=10, episodes=20000):
    res = {k: np.array([run(k, s, episodes) for s in range(seeds)]) for k in KINDS}       # [seeds, checkpoints]
    mean_pref = {k: v.mean(1) for k, v in res.items()}                                    # per-seed, averaged over training
    verdict = {}
    ci = {k: boot_ci(mean_pref[k]) for k in KINDS}
    consistency = {k: float((mean_pref[k] > 0.5).mean()) for k in KINDS}
    exact_gap = float(np.abs(exact_q()[:, LOW] - exact_q()[:, HIGH]).max())
    low_pref = [k for k in ("double", "reinforce", "linear") if ci[k][1] > 0.5]
    if exact_gap > 1e-12 or ci["risk"][1] <= 0.5:
        status = "invalid"          # environment or positive control is broken
    elif len(low_pref) == 3 and ci["risk"][0] >= max(ci[k][0] for k in ("double", "reinforce", "linear")):
        status = "supports"
    elif all(ci[k][2] < 0.5 + 1e-9 or ci[k][1] <= 0.5 for k in ("double", "reinforce", "linear")):
        status = "weakens"
    else:
        status = "neutral"
    verdict = {"claim": "C5", "status": status, "pref_ci": ci, "seeds_above_half": consistency,
               "exact_q_gap": exact_gap, "families_preferring_low_variance": low_pref}
    return {"prefs": res, "verdict": verdict}
