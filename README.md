# PCSA

Project repository for PCSA.

## Documents

- [Prediction, Control, and the Second Agent (1).md](./Prediction,%20Control,%20and%20the%20Second%20Agent%20(1).md)
- [Test Spec Prediction, Control, and the Second Agent (1).md](./Test%20Spec%20Prediction,%20Control,%20and%20the%20Second%20Agent%20(1).md)

## Running the tests

`notebooks/pcsa_experiments.ipynb` runs the simulation suite from the test spec (Gate 0, exact-solver study, S1–S7) in Colab (CPU).
Set `PROFILE` (`smoke` → `pilot` → `full`) and run all cells; it prints a SUPPORTS / WEAKENS / neutral scorecard per claim.

`notebooks/pcsa_llm_gpu.ipynb` runs the LLM suite (L0–L6) on a Colab GPU (A100 40/80 GB or G4; the memory plan is chosen automatically). Run it once with `DRY_RUN = True` (tiny random model, mocks, CPU, ~2 min) to check the plumbing, then set it to `False` and start with `PROFILE = "quick"`. Model ids are unverified defaults; the notebook checks them against the Hub first.

```
pip install -e . pytest      # numpy is the only runtime dependency (pandas + matplotlib for the notebook)
pip install torch transformers peft   # only for the LLM tests; they skip without torch
pytest                       # the spec's "Code requirements for review" checklist as unit tests (~2.5 min)
```

| module | spec component |
|---|---|
| `pcsa/world.py` | world generator, KL / entropy |
| `pcsa/users.py` | Bayes, Weighted, Confirm, Deference users |
| `pcsa/env.py` | dialogue environment; `AgentObservation` / `PrivilegedEvalState` / `AuditTrace` |
| `pcsa/rewards.py` | R-replication, R-general, R-task |
| `pcsa/verify.py` | exact / noisy / subsample verifier and mask |
| `pcsa/policies.py` | scripted policies, tabular Q-learner |
| `pcsa/solver.py` | exact dynamic programming (Gate 0) |
| `pcsa/metrics.py` | bootstrap CIs, equivalence test, Holm, AUC, ρ and capacity measures |
| `pcsa/experiments.py`, `pcsa/matched.py` | Gate 0, S1–S4, S6, S7; S5 |
| `pcsa/runlog.py`, `prereg/` | pre-registration hash check, run metadata |
| `pcsa_llm/textworld.py`, `citations.py`, `llm_user.py` | synthetic-text world (exact p\*), citation parser, LLM user + gate |
| `pcsa_llm/dialogue.py`, `metrics.py`, `scripted.py`, `l_experiments.py` | dialogue runner, LLM metrics, L1 scripted dialogues, L2 |
| `pcsa_llm/ppo.py`, `l_train.py`, `runs.py` | turn-level PPO (arms A/B/C), L5 filter, L6 judge, L3 runs |
| `pcsa_llm/probes.py`, `verdicts.py`, `dry.py` | L4 probes/steering/predicted-user, verdicts, CPU mocks |

Deviations from the spec are logged in `prereg/prereg.json` (`deviation_log`). The LLM code has been run only on a tiny random model on CPU; the Hugging Face adapters and PPO loop have not yet run on a real 7B model.
