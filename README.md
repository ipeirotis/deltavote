# deltavote

Python implementation of the δ-margin majority voting formulas and
estimators from:

> Boyarskaya, M. and Ipeirotis, P. *A Design Calculus for δ-margin
> Majority Voting: Quality, Cost, and Payment.* Submitted to JAIR.

**Status:** In progress. The §4 core formulas, the §5.1 shifted-start
in-flight estimators, the §6 equivalence/payment results, and the
practitioner design helpers are implemented and tested; the §5 Bayesian
estimators (`bayes.py`) are next. See [`TASKS.md`](TASKS.md) for the v0.1
task list and [`AGENTS.md`](AGENTS.md) for design principles and the
paper-to-code notation map.

## Scope

`deltavote` is a small, lean library — `numpy` and `scipy` only — whose
public API maps one-to-one to results in the paper:

- §4 (Theorems 4.1–4.5) — closed-form consensus quality `Q(φ, δ)`,
  expected votes, variance, and pmf of time-to-consensus.
- §5.1 (shifted start, Eqs. H_s/T_s and Prop. 5.1) — in-flight estimates
  of remaining quality and remaining votes given the current state
  `(n₁, n₂)`.
- §5 — Bayesian estimation of worker accuracy `p` under the canonical
  `Beta(k, 1)` prior and mixture-of-Betas priors, with the model-averaged
  consensus-correctness estimate `Q̂` (Prop. 5.2) and posterior-predictive
  quality and cost. *(planned — `bayes.py`)*
- §6 (Theorems 6.1–6.2) — δ-equivalence and payment-equivalence across
  worker pools of different accuracy.
- Practitioner helpers — `recommend_delta`, `expected_cost`.

Multi-class voting, weighted votes, iterative refinement, dataset
loaders, and plotting utilities are explicitly **out of scope**.
