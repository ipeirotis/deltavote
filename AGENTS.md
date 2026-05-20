# deltavote — Agent Guide

## Purpose

`deltavote` is a Python package that implements the formulas and estimators
from the paper:

> Boyarskaya, M. and Ipeirotis, P. *Theoretical Foundations of δ-margin
> Majority Voting for Quality Assurance in High-Stakes Machine Learning.*
> Submitted to JAIR.

The package is the practitioner-facing companion to the paper. **Every public
function should correspond to a result (theorem, proposition, or worked
example) in the paper.** If a feature is not in the paper, it does not belong
in this package — propose a follow-up paper instead.

## Source of truth

The authoritative reference for all formulas, notation, and worked examples
is the paper, maintained in the companion repository `ipeirotis/margin-voting`
(private). The paper's `CLAUDE.md` and section files under `tex_files/` are
the canonical specification for what this package implements.

When implementing or modifying a function:

1. Find the corresponding theorem or equation in the paper.
2. Cite it in the function docstring (e.g. *"Theorem 2, Section 4"*).
3. Reproduce a worked numerical example from the paper as a unit test in
   `tests/test_paper_examples.py`.

If you find a discrepancy between the paper and the code, **the paper wins**
unless the discrepancy is itself a paper bug — in which case open an issue
in `ipeirotis/margin-voting` rather than silently diverging.

## Notation map (paper ↔ code)

| Paper symbol | Code name           | Meaning                                  |
|--------------|---------------------|------------------------------------------|
| p            | `p`                 | Per-voter accuracy                       |
| φ = p/(1-p)  | `phi`               | Odds of a correct vote                   |
| δ            | `delta`             | Margin threshold                         |
| n₁, n₂       | `n1`, `n2`          | Current vote counts for the two labels   |
| Q(φ, δ)      | `consensus_quality` | Probability the consensus label is right |
| E[n_votes]   | `expected_votes`    | Expected number of votes to consensus    |
| Var[n_votes] | `var_votes`         | Variance of votes to consensus           |
| pmf(m)       | `votes_pmf`         | Distribution of votes to consensus       |

## Module layout

```
src/deltavote/
├── __init__.py
├── core.py         # §4, Theorems 1–4: Q, E[n], Var[n], pmf — from scratch (n1 = n2 = 0)
├── inflight.py     # §4 (shifted start): same quantities given current (n1, n2)
├── equivalence.py  # §5, §6: delta equivalence (Thm 5), payment equivalence (Thm 6)
├── bayes.py        # §5: Beta and mixture-of-Betas priors, posterior over p,
│                   #     posterior-predictive quality and expected votes
└── design.py       # Practitioner helpers: recommend_delta, expected_cost
```

Tests live in `tests/`. Example notebooks (if any) live in `examples/` and
are **not** installed with the package.

## Design principles

1. **Lean dependencies.** `numpy` + `scipy` only. No `pandas`, no plotting,
   no dataset bundling. Heavy or domain-specific dependencies belong in
   downstream applications, not here.
2. **Vectorized.** Functions accept Python scalars *or* NumPy arrays for
   `phi`, `delta`, `n1`, `n2`. Broadcasting follows NumPy rules.
3. **No I/O.** No file readers, no dataset loaders, no `requests`. The
   Bluebirds and AML datasets stay in the paper repository.
4. **Paper-traceable.** Every public function's docstring cites the paper
   result it implements.
5. **No future work.** Multi-class voting, continuous votes, iterative
   refinements, active-learning policies, and any other items listed in
   §9 *Future Work* of the paper are explicitly out of scope.
6. **No silent fallbacks.** Validate inputs at module boundaries; raise
   `ValueError` for invalid `phi`, `delta`, or `(n1, n2)` rather than
   coercing.

## Testing

- Framework: `pytest`.
- Every worked example in the paper becomes a unit test in
  `tests/test_paper_examples.py`.
- The numerical-verification logic from `scripts/verify_numerics.py` in the
  paper repository should be ported into the test suite so the package
  becomes the canonical numerical reference.
- Property tests (e.g. monotonicity of Q in δ, monotonicity in φ) live in
  `tests/test_properties.py`.
- Closed-form ↔ Monte-Carlo sanity checks live in `tests/test_simulation.py`.

## Versioning and release

- `0.1.0` — Initial release covering §4, §5, §6 of the paper. No earlier
  pre-releases.
- Semantic versioning. Patch bumps for bug fixes and docstring fixes; minor
  bumps for added functions that still map to paper results; major bumps
  reserved for incompatible API changes.
- Publish to PyPI under the name `deltavote` (verify availability before
  first release).

## License

To be confirmed by the paper authors. Default suggestion: MIT.

## Out of scope (do not add)

- Multi-class extensions (paper §9 future work).
- Continuous-vote or weighted-vote variants (paper §9 future work).
- Iterative refinement or active-learning policies (paper §9 future work).
- Dataset loaders (Bluebirds, AML, or otherwise).
- Plotting utilities or notebook helpers in the installed package.
- Worker assignment, scheduling, or any system-level orchestration logic.
- Anything not derivable from a numbered result in the paper.
