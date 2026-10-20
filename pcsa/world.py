"""World generator and the information-theory helpers every metric uses."""
from dataclasses import dataclass
import numpy as np

EPS = 1e-12


def normalize(x):
    return x / x.sum(-1, keepdims=True)


def entropy(p):
    return -(p * np.log(p + EPS)).sum(-1)


def kl(p, q):
    """KL(p || q) along the last axis."""
    return (p * (np.log(p + EPS) - np.log(q + EPS))).sum(-1)


@dataclass
class World:
    hstar: np.ndarray   # [B]
    L: np.ndarray       # [B,N,K] likelihood vector of each pool item
    fav: np.ndarray     # [B,N]   argmax of each item's likelihood vector
    p0: np.ndarray      # [B,K]   user's prior
    lean: np.ndarray    # [B]     prior's leaned hypothesis
    pstar: np.ndarray   # [B,K]   prior updated on the whole pool (beta = 1)


def make_world(cfg, B, rng):
    """Every item is drawn from h*, so every item is a true statement. Fully determined by rng."""
    K, N = cfg.K, cfg.N
    ar = np.arange(B)
    hstar = rng.integers(0, K, B)
    is_true = rng.random((B, N)) < cfg.p_true
    other = (hstar[:, None] + 1 + rng.integers(0, K - 1, (B, N))) % K
    fav = np.where(is_true, hstar[:, None], other)
    ell = rng.uniform(*cfg.ell, (B, N))
    L = np.repeat(((1 - ell) / (K - 1))[:, :, None], K, axis=2)
    np.put_along_axis(L, fav[:, :, None], ell[:, :, None], axis=2)
    wrong = rng.random(B) < cfg.wrong_lean
    wrong_h = (hstar + 1 + rng.integers(0, K - 1, B)) % K
    lean = np.where(wrong, wrong_h, hstar)
    p0 = np.full((B, K), (1 - cfg.lean) / (K - 1))
    p0[ar, lean] = cfg.lean
    pstar = normalize(np.exp(np.log(p0) + np.log(L).sum(1)))
    return World(hstar, L, fav, p0, lean, pstar)
