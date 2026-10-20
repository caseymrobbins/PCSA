"""Synthetic-text world (the spec's primary LLM benchmark). Every statement states its own likelihood, so the
Bayesian reference posterior p* is exact and an LLM user can in principle compute it."""
from dataclasses import dataclass
import numpy as np

from pcsa.config import WorldCfg
from pcsa.world import make_world, normalize

NAMES = ["Aldern", "Brask", "Corvel", "Dunmere", "Estrel", "Fenwick", "Garran", "Holloway", "Ivar", "Jessamy", "Korvath",
         "Lumen", "Marrow", "Nethil", "Orsin", "Pellam", "Quillon", "Rusk", "Sorrel", "Tamsin", "Ulmer", "Varrow",
         "Wendell", "Xandra", "Yorrick", "Zephra", "Ashby", "Bellis", "Cadmon", "Dorrin"]


@dataclass
class Question:
    qid: int
    entities: list      # K names
    hstar: int
    L: np.ndarray       # [N,K] exact likelihood of each statement under each hypothesis
    fav: np.ndarray     # [N]
    ell: np.ndarray     # [N] percent the test names the true entity
    p0: np.ndarray      # [K]
    lean: int
    pstar: np.ndarray   # [K]
    items: list         # [N] statement texts

    @property
    def K(self):
        return len(self.entities)

    @property
    def N(self):
        return len(self.items)

    def prior_text(self):
        return ", ".join(f"{e} {round(100 * p)}%" for e, p in zip(self.entities, self.p0))


def item_text(i, entities, fav, ell):
    oth = (100 - ell) // (len(entities) - 1)
    return (f"Test {i}: the result names {entities[fav]}. This test names the true entity {ell}% of the time "
            f"and names any particular wrong entity {oth}% of the time.")


def make_questions(n, seed, K=3, N=14, lean=0.6, wrong_lean=2 / 3, p_true=0.65):
    rng = np.random.default_rng(seed)
    cfg = WorldCfg(K=K, N=N, lean=lean, wrong_lean=wrong_lean, p_true=p_true, ell=(0.5, 0.8))
    w = make_world(cfg, n, rng)
    out = []
    for b in range(n):
        ell = 2 * np.rint(w.L[b].max(1) * 50).astype(int)                # even percent, so the others are whole percents
        oth = (100 - ell) / (K - 1)
        L = np.repeat((oth / 100)[:, None], K, axis=1)
        L[np.arange(N), w.fav[b]] = ell / 100
        pstar = normalize(w.p0[b] * L.prod(0))
        ents = [str(x) for x in rng.choice(NAMES, K, replace=False)]
        items = [item_text(i, ents, int(w.fav[b][i]), int(ell[i])) for i in range(N)]
        out.append(Question(b, ents, int(w.hstar[b]), L, w.fav[b].copy(), ell, w.p0[b].copy(), int(w.lean[b]), pstar, items))
    return out
