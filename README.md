# PCSA

Project repository for PCSA.

## Documents

- [Prediction, Control, and the Second Agent (1).md](./Prediction,%20Control,%20and%20the%20Second%20Agent%20(1).md)
- [Test Spec Prediction, Control, and the Second Agent (1).md](./Test%20Spec%20Prediction,%20Control,%20and%20the%20Second%20Agent%20(1).md)

## Running the tests

`notebooks/pcsa_experiments.ipynb` runs the simulation suite from the test spec (Gate 0, S1–S7) in Colab.
Set `PROFILE` (`smoke` → `pilot` → `full`) and run all cells; it prints a SUPPORTS / WEAKENS / neutral scorecard per claim.

```
pip install -e . pytest      # numpy is the only runtime dependency (pandas + matplotlib for the notebook)
pytest                       # the spec's "Code requirements for review" checklist as unit tests (~70 s)
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

Not implemented: the LLM suite (L1–L6). Deviations from the spec are listed in `prereg/prereg.json`.
