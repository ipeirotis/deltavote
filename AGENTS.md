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

## Repository layout

The **installed package** is intentionally lean:

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

The **project repository** additionally contains:

```
tests/        # pytest suite — paper-example tests, property tests, simulation cross-checks
examples/     # Jupyter notebooks demonstrating practitioner workflows (NOT installed)
docs/         # Documentation source — narrative pages plus rendered example notebooks
.github/      # CI workflows, issue templates
```

`examples/` and `docs/` are first-class deliverables of the project — they
just are not shipped inside the installable wheel. See *Examples and
documentation* below.

## Design principles

1. **Lean dependencies in the wheel.** `numpy` + `scipy` only as runtime
   requirements. Heavier dependencies (e.g. `pandas`, `matplotlib`,
   `jupyter`, the docs toolchain) belong in `[project.optional-dependencies]`
   groups (`docs`, `examples`, `dev`), not in the default install.
2. **Vectorized.** Functions accept Python scalars *or* NumPy arrays for
   `phi`, `delta`, `n1`, `n2`. Broadcasting follows NumPy rules.
3. **No I/O in the package.** No file readers, no dataset loaders, no
   `requests`. The Bluebirds and AML datasets stay in the paper
   repository. Notebooks may load whatever they need locally.
4. **Paper-traceable.** Every public function's docstring cites the paper
   result it implements.
5. **No future work.** Multi-class voting, continuous votes, iterative
   refinements, active-learning policies, and any other items listed in
   §9 *Future Work* of the paper are explicitly out of scope.
6. **No silent fallbacks.** Validate inputs at module boundaries; raise
   `ValueError` for invalid `phi`, `delta`, or `(n1, n2)` rather than
   coercing.

## Examples and documentation

Notebooks and a documentation site are **in scope** for this project. They
are how a practitioner will actually discover and learn the package.

- **`examples/` directory.** Self-contained Jupyter notebooks showing
  realistic workflows: estimating `p` from historical data, choosing `δ`
  for a target quality, monitoring an in-progress item, comparing two
  worker pools. Notebooks are *not* installed with the wheel but *are*
  committed to the repo, tested in CI (execute without error), and
  rendered into the docs site.
- **`docs/` directory.** Narrative documentation plus rendered example
  notebooks. Built with `mkdocs` + `mkdocs-jupyter` (recommended for
  speed and simplicity) or `sphinx` + `myst-nb` (recommended if heavier
  cross-referencing is needed). The site is published to GitHub Pages.
- **Plotting in notebooks is fine.** Use `matplotlib` directly. Do *not*
  add a `deltavote.plot` module to the installed package — plotting
  helpers, if any are repeated across notebooks, belong in a small
  `examples/_plotting.py` helper module that is not shipped.

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
- Example notebooks in `examples/` are executed in CI (e.g. via
  `jupyter nbconvert --execute` or `pytest --nbmake`) to catch breakage
  as the API evolves.

## Versioning and release

- `0.1.0` — Initial release covering §4, §5, §6 of the paper. No earlier
  pre-releases.
- Semantic versioning. Patch bumps for bug fixes and docstring fixes; minor
  bumps for added functions that still map to paper results; major bumps
  reserved for incompatible API changes.
- Publish to PyPI under the name `deltavote` (verify availability before
  first release).
- The docs site is rebuilt and republished on every tagged release.

## License

To be confirmed by the paper authors. Default suggestion: MIT.

## Out of scope (do not add)

The following are out of scope for the **installed package**. Several may
appear in notebooks or the docs site — that is fine.

- Multi-class extensions (paper §9 future work) — out of scope everywhere.
- Continuous-vote or weighted-vote variants (paper §9 future work) — out of
  scope everywhere.
- Iterative refinement or active-learning policies (paper §9 future work) —
  out of scope everywhere.
- Dataset loaders or bundled datasets — out of scope in the package;
  notebooks may load datasets from local paths or URLs.
- `deltavote.plot` or any plotting module inside the wheel — plotting
  belongs in notebooks, using `matplotlib` directly.
- Worker assignment, scheduling, or any system-level orchestration logic —
  out of scope everywhere.
- Anything not derivable from a numbered result in the paper — out of scope
  in the package; notebooks may sketch follow-up ideas as long as they
  are clearly labelled "beyond the paper".
