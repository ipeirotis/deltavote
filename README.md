# deltavote

Python implementation of the δ-margin majority voting formulas and
estimators from:

> Boyarskaya, M. and Ipeirotis, P. *Theoretical Foundations of δ-margin
> Majority Voting for Quality Assurance in High-Stakes Machine Learning.*
> Submitted to JAIR.

**Status:** Planning. See [`TASKS.md`](TASKS.md) for the v0.1 task list and
[`AGENTS.md`](AGENTS.md) for design principles and the paper-to-code
notation map. No code has been written yet.

## Scope

`deltavote` is a small, lean library — `numpy` and `scipy` only — whose
public API maps one-to-one to results in the paper:

- §4 — closed-form consensus quality `Q(φ, δ)`, expected votes,
  variance, and pmf of time-to-consensus.
- §4 (shifted start) — in-flight estimates of remaining quality and
  remaining votes given the current state `(n₁, n₂)`.
- §5 — Bayesian estimation of worker accuracy `p` under Beta and
  mixture-of-Betas priors, with posterior-predictive quality and cost.
- §6 — δ-equivalence and payment-equivalence across worker pools of
  different accuracy.
- Practitioner helpers — `recommend_delta`, `expected_cost`.

Multi-class voting, weighted votes, iterative refinement, dataset
loaders, and plotting utilities are explicitly **out of scope**.
