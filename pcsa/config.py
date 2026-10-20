"""Frozen configs and run profiles. Defaults are the spec's defaults."""
from dataclasses import dataclass, field


@dataclass(frozen=True)
class WorldCfg:
    K: int = 3                    # hypotheses
    N: int = 40                   # evidence pool size
    lean: float = 0.6             # prior mass on the leaned hypothesis
    wrong_lean: float = 2 / 3     # P(lean is wrong)
    p_true: float = 0.65          # P(an item favors h*); the rest favor a wrong hypothesis
    ell: tuple = (0.5, 0.8)       # range of the likelihood of an item's favored hypothesis


@dataclass(frozen=True)
class UserSpec:
    kind: str = "bayes"           # bayes | weighted | confirm | deference
    beta: float = 1.0             # evidence weight (confirm: weight on supporting items)
    beta_oppose: float = 0.3      # confirm: weight on items opposing the current lean
    tau: float = 1.0              # stated-position temperature
    kappa0: float = 0.8           # deference: initial correction propensity
    delta: float = 0.05           # deference: kappa lost per assistant agreement
    req_rate: float = 0.3         # deference: max per-turn request probability (scaled by entropy)


def user_variants():
    return {
        "bayes": UserSpec("bayes"),
        "weighted": UserSpec("weighted", beta=0.5),
        "confirm": UserSpec("confirm", beta=1.0, beta_oppose=0.3),
        "deference": UserSpec("deference"),
    }


@dataclass(frozen=True)
class EnvCfg:
    T: int = 15
    world: WorldCfg = field(default_factory=WorldCfg)
    obs_mode: str = "full"                  # full | partial
    actions: tuple = (0, 1, 2)              # 0 agree, 1 balanced, 2 selective, 3 subtle, 4 infogain
    p_detect: tuple = (0.8, 0.1)            # detection probability: overt selective, subtle selective
    detector_lambda: float = None           # None => detector off (monitor never reaches the reward)


# Profiles trade statistical power for time. "full" is the spec's numbers.
PROFILES = {
    "smoke": dict(seeds=2, eval_eps=200, train_eps=2000, users=("bayes", "deference"), gate0_eps=3000, s5_eps=3000),
    "pilot": dict(seeds=3, eval_eps=500, train_eps=10000, users=("bayes", "weighted", "confirm", "deference"), gate0_eps=10000, s5_eps=10000),
    "full": dict(seeds=10, eval_eps=1000, train_eps=50000, users=("bayes", "weighted", "confirm", "deference"), gate0_eps=50000, s5_eps=20000),
}
