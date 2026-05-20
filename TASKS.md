# deltavote — v0.1 Task List

Initial-release task list. Tackle roughly in order; tasks within a section can
generally be parallelized. Every task references the paper section or theorem
it implements — keep that link explicit in commits and docstrings.

## A. Repository scaffolding

- [ ] `pyproject.toml` — project name `deltavote`, runtime deps `numpy>=1.20`,
      `scipy>=1.7`, dev deps `pytest`, `pytest-cov`.
- [ ] `LICENSE` — confirm with authors (default suggestion: MIT).
- [ ] `README.md` — quickstart, paper citation (BibTeX), install instructions,
      one minimal end-to-end example per module.
- [ ] `.gitignore` — standard Python.
- [ ] `.github/workflows/test.yml` — pytest on Python 3.9, 3.10, 3.11, 3.12.
- [ ] `CITATION.cff` — point to the JAIR paper once accepted.
- [ ] `src/deltavote/__init__.py` — re-export the public API from each module.

## B. Core formulas — `core.py` (paper §4)

- [ ] `consensus_quality(phi, delta)` — Theorem 1: Q = φ^δ / (1 + φ^δ).
- [ ] `expected_votes(phi, delta)` — Theorem 2.
- [ ] `var_votes(phi, delta)` — Theorem 3 (quarter-squares coefficients).
- [ ] `votes_pmf(m, phi, delta)` — Theorem 4 (discrete phase-type via the
      Markov-chain transition matrix).
- [ ] Vectorize all of the above over `phi` and `delta`.
- [ ] Edge cases: `phi == 1` (random voter — Q = 1/2), `phi → ∞`, `delta == 1`.
- [ ] Input validation (`phi > 0`, `delta >= 1` integer).

## C. In-flight estimators — `inflight.py`

Gambler's-Ruin formulas with a shifted starting state `d = n1 - n2`,
`d ∈ (-δ, δ)`. These are mid-process estimates given the votes seen so far.

- [ ] `remaining_quality(n1, n2, phi, delta)` — P(absorb at +δ | start at d).
- [ ] `remaining_expected_votes(n1, n2, phi, delta)`.
- [ ] `remaining_var_votes(n1, n2, phi, delta)`.
- [ ] `remaining_votes_pmf(m, n1, n2, phi, delta)`.
- [ ] Verify that `(n1=0, n2=0)` reduces exactly to the `core.py` functions
      (cross-module unit test).
- [ ] Error if `abs(n1 - n2) >= delta` (the process has already absorbed).

## D. Equivalence — `equivalence.py` (paper §5, §6)

- [ ] `equivalent_delta(phi1, delta1, phi2)` — Theorem 5:
      `δ₂ = δ₁ · ln(φ₁) / ln(φ₂)`.
- [ ] `equivalent_payment(phi)` — Theorem 6: `pay(φ) ∝ ln(φ) · (φ-1) / (φ+1)`.
- [ ] `quality_matched_pools(phi1, delta1, phi_list)` — convenience wrapper
      that returns the matched δ for each pool.

## E. Bayesian estimators — `bayes.py` (paper §5)

- [ ] `BetaPrior(a, b, truncated_above_half=False)` — Beta(α, β) prior on
      worker accuracy `p`, optionally truncated to `p > 1/2`.
- [ ] `MixtureBetaPrior(components, weights)` — finite mixture of Beta
      priors (§5.2).
- [ ] `posterior_p(prior, n1, n2)` — posterior over `p` after observing
      `(n1, n2)` votes.
- [ ] `posterior_quality(prior, delta, n1, n2)` — `E[Q(φ, δ) | observed]`
      integrated over the posterior on `p`.
- [ ] `posterior_expected_votes(prior, delta, n1, n2)` — `E[n_remaining |
      observed]` integrated over the posterior.
- [ ] `method_of_moments(sample_accuracies)` — derive Beta(α, β) parameters
      from a sample of observed worker accuracies (returns `(alpha, beta)`,
      with the `ν = α + β` convention from the paper).
- [ ] All `*posterior_*` functions accept a `Prior` instance or a callable
      density on `p`; they should not require numeric integration tuning
      from the caller for the standard Beta and mixture-Beta cases.

## F. Design helpers — `design.py`

- [ ] `recommend_delta(phi, target_quality)` — smallest integer `δ` such
      that `Q(φ, δ) ≥ target_quality`.
- [ ] `expected_cost(phi, delta, cost_per_vote=1.0)` — wraps `expected_votes`.
- [ ] `cost_for_target_quality(phi, target_quality, cost_per_vote=1.0)` —
      composes `recommend_delta` and `expected_cost`.

## G. Tests — `tests/`

- [ ] `test_paper_examples.py` — port every worked numerical example from
      the paper (§4 through §6 and Appendix A). One test per example.
- [ ] Port the checks in `scripts/verify_numerics.py` from the paper repo.
- [ ] `test_properties.py` — monotonicity of Q in δ (for `phi > 1`),
      monotonicity in φ, symmetry under `(n1, n2) ↔ (n2, n1)` with
      `phi ↔ 1/phi`.
- [ ] `test_simulation.py` — Monte-Carlo simulation of the Markov chain;
      verify closed-form Q, E[n], Var[n] within tolerance.
- [ ] `test_inflight_reduction.py` — `inflight.*` with `(0, 0)` matches
      `core.*` to machine precision.
- [ ] CI green on the supported Python matrix.

## H. Documentation

- [ ] README quickstart — 5-line example for each module's headline function.
- [ ] Per-module docstrings citing the paper result implemented.
- [ ] `docs/formula-table.md` — formula-to-function mapping (one row per
      public function, with paper theorem and equation number).
- [ ] Optional: simple `mkdocs` or `sphinx` site — defer until after v0.1
      ships.

## I. Integration with the paper repository

These tasks live in `ipeirotis/margin-voting`, not in `deltavote`, but
they should ship together with the package release.

- [ ] Footnote in §4 of `main.tex` pointing to the package and its PyPI
      page.
- [ ] Line in the reproducibility checklist appendix referencing the
      package.
- [ ] One companion notebook in `notebooks/` of the paper repo that uses
      `deltavote` to regenerate a figure, as a sanity check that the API
      reproduces the published numbers.

## Out of scope — do not add

- Multi-class extensions (paper §9 future work).
- Continuous-vote or weighted-vote variants (paper §9 future work).
- Iterative refinement or active-learning policies (paper §9 future work).
- Dataset loaders (Bluebirds, AML, or otherwise).
- Plotting utilities or notebook helpers in the installed package.
- Worker assignment, scheduling, or any orchestration logic.
- Anything not derivable from a numbered result in the paper.
