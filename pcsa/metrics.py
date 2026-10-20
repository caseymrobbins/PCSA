"""Statistics helpers and the agency measures (rho, toy capacity, probability-weighted capacity)."""
import numpy as np
from .world import entropy

GRID = 0.05


def boot_ci(x, rng=None, n=2000, level=0.95):
    """Mean and percentile-bootstrap CI over the first axis (seed-level values)."""
    x = np.asarray(x, float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = rng or np.random.default_rng(0)
    idx = rng.integers(0, len(x), (n, len(x)))
    means = x[idx].mean(1)
    a = (1 - level) / 2
    return float(x.mean()), float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def boot_p(d, rng=None, n=4000):
    """One-sided bootstrap p-value for H1: mean(d) > 0."""
    d = np.asarray(d, float)
    rng = rng or np.random.default_rng(0)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float((np.sum(means <= 0) + 1) / (n + 1))


def holm(pvals):
    """Holm step-down adjusted p-values."""
    p = np.asarray(pvals, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def equivalent(d, margin=0.02, rng=None):
    """Two one-sided tests at alpha=.05 == the 90% bootstrap CI of the mean lies inside +-margin."""
    _, lo, hi = boot_ci(d, rng, level=0.90)
    return bool(lo > -margin and hi < margin), (lo, hi)


def auc(score, label):
    """P(score of a random positive > score of a random negative), ties count 1/2."""
    score, label = np.asarray(score, float), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order))
    allv = np.concatenate([pos, neg])[order]
    # average ranks for ties
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


# ---- agency measures (DeferenceUser), evaluated over a forward window of H turns ----------------
def _kappa_path(kappa, delta, H, regime):
    j = np.arange(H)[None, :]
    k = kappa[:, None] - (delta * j if regime == "agree" else 0 * j)
    return np.maximum(k, 0.0)


def rho(env, H=5, regime="balanced"):
    """P(user issues >=1 correction over the next H turns if the assistant states a wrong position every turn).
    Belief is held fixed; regime 'agree' lets kappa erode by delta per turn, 'balanced' leaves it alone."""
    sp, p, hs = env.spec, env.user.p, env.world.hstar
    w = p ** (1 / sp.tau)
    w = w / w.sum(1, keepdims=True)
    wrong = np.where(np.arange(p.shape[1])[None, :] == hs[:, None], -1.0, w)
    d = 1.0 - wrong.max(1)                                  # assistant states the likeliest wrong position
    q_req = sp.req_rate * entropy(p) / np.log(p.shape[1])
    c = (1 - q_req)[:, None] * d[:, None] * _kappa_path(env.user.kappa, sp.delta, H, regime)
    return 1.0 - np.prod(1.0 - c, 1)


def _cells(p):
    return np.rint(p / GRID).astype(int)


def _key(cells):
    return cells[..., :-1] @ (np.array([22 ** i for i in range(cells.shape[-1] - 1)]))


def capacity(env, idx, H=5, regime="balanced", paths=200, seed=0):
    """Toy capacity C_user = log #reachable belief cells; C_user^p = exp entropy of the visited-cell distribution
    under the user's own request policy. Bayes update; windows shrink when kappa reaches 0 (regime 'agree' erodes it)."""
    sp, rng = env.spec, np.random.default_rng(seed)
    K = env.user.p.shape[1]
    C, Cp = [], []
    for e in idx:
        kap = _kappa_path(env.user.kappa[e:e + 1], sp.delta, H, regime)[0]
        R = int((kap > 1e-9).sum())
        Lu = env.world.L[e][env.unseen[e]]                                  # [Nu,K]
        p0 = env.user.p[e]
        reps = {tuple(_cells(p0)[:-1]): p0}
        for _ in range(R):
            cur = np.array(list(reps.values()))
            nxt = cur[:, None, :] * Lu[None]
            nxt = (nxt / nxt.sum(2, keepdims=True)).reshape(-1, K)
            for pp, k in zip(nxt, map(tuple, _cells(nxt)[:, :-1])):
                reps.setdefault(k, pp)
        C.append(np.log(len(reps)))
        # probability-weighted: simulate the user's own request policy
        cur = np.tile(p0, (paths, 1))
        visited = [tuple(c) for c in _cells(cur)[:, :-1]]
        for _ in range(R):
            q_req = sp.req_rate * entropy(cur) / np.log(K)
            go = rng.random(paths) < q_req
            pick = rng.integers(0, len(Lu), paths)
            new = cur * Lu[pick]
            new /= new.sum(1, keepdims=True)
            cur = np.where(go[:, None], new, cur)
            visited += [tuple(c) for c in _cells(cur)[:, :-1]]
        _, cnt = np.unique(np.array(visited), axis=0, return_counts=True)
        pr = cnt / cnt.sum()
        Cp.append(np.exp(-(pr * np.log(pr)).sum()))
    return np.array(C), np.array(Cp)
