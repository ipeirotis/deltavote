# deltavote — v0.1 Task List

Initial-release task list. Tackle roughly in order; tasks within a section can
generally be parallelized. Every task references the paper section or theorem
it implements — keep that link explicit in commits and docstrings.

## A. Repository scaffolding

- [x] `pyproject.toml` — project name `deltavote`, runtime deps `numpy>=1.20`,
      `scipy>=1.7`; optional-dependency groups `dev` (`pytest`, `pytest-cov`,
      `pytest-nbmake`), `docs` (`mkdocs`, `mkdocs-material`, `mkdocs-jupyter`),
      `examples` (`matplotlib`, `jupyter`).
- [x] `LICENSE` — MIT (default suggestion; confirm with co-author).
- [ ] `README.md` — quickstart, paper citation (BibTeX), install instructions,
      one minimal end-to-end example per module, link to docs site.
- [x] `.gitignore` — standard Python, plus `site/` and `.ipynb_checkpoints/`.
- [x] `.github/workflows/test.yml` — pytest on Python 3.9, 3.10, 3.11, 3.12.
- [ ] `.github/workflows/docs.yml` — build and deploy docs to GitHub Pages on
      pushes to `main` and on tagged releases.
- [x] `CITATION.cff` — point to the JAIR paper once accepted.
- [x] `src/deltavote/__init__.py` — re-export the public API from each module.

## B. Core formulas — `core.py` (paper §4)

- [x] `consensus_quality(phi, delta)` — Theorem 1: Q = φ^δ / (1 + φ^δ).
- [x] `expected_votes(phi, delta)` — Theorem 2.
- [x] `var_votes(phi, delta)` — Theorem 3 (quarter-squares coefficients).
- [x] `votes_pmf(m, phi, delta)` — Theorem 4 (discrete phase-type via the
      Markov-chain transition matrix).
- [x] Vectorize all of the above over `phi` and `delta`.
- [x] Edge cases: `phi == 1` (random voter — Q = 1/2), `phi → ∞`, `delta == 1`.
- [x] Input validation (`phi > 0`, `delta >= 1` integer).

## C. In-flight estimators — `inflight.py`

Gambler's-Ruin formulas with a shifted starting state `d = n1 - n2`,
`d ∈ (-δ, δ)`. These are mid-process estimates given the votes seen so far.

- [x] `remaining_quality(n1, n2, phi, delta)` — P(absorb at +δ | start at d).
- [x] `remaining_expected_votes(n1, n2, phi, delta)`.
- [x] `remaining_var_votes(n1, n2, phi, delta)`.
- [x] `remaining_votes_pmf(m, n1, n2, phi, delta)`.
- [x] Verify that `(n1=0, n2=0)` reduces exactly to the `core.py` functions
      (cross-module unit test).
- [x] Error if `abs(n1 - n2) >= delta` (the process has already absorbed).

## D. Equivalence — `equivalence.py` (paper §5, §6)

- [x] `equivalent_delta(phi1, delta1, phi2)` — Theorem 5:
      `δ₂ = δ₁ · ln(φ₁) / ln(φ₂)`.
- [x] `equivalent_payment(phi)` — Theorem 6: `pay(φ) ∝ ln(φ) · (φ-1) / (φ+1)`.
- [x] `quality_matched_pools(phi1, delta1, phi_list)` — convenience wrapper
      that returns the matched δ for each pool.

## E. Bayesian estimators — `bayes.py` (paper §5)

**Resolve the open API questions in `HANDOVER.md` before implementing this
section.**

- [ ] `BetaPrior(a, b, truncated_above_half=False)` — Beta(α, β) prior on
      worker accuracy `p`, optionally truncated to `p > 1/2`.
- [ ] `MixtureBetaPrior(components, weights)` — finite mixture of Beta
      priors (§5.2).
- [ ] `posterior_p(prior, n1, n2)` — posterior over `p` after observing
      `(n1, n2)` votes.
- [ ] Population-level posterior-predictive functions (from-scratch quality
      and expected votes integrated over the prior/posterior on `p`).
      Final names TBD pending HANDOVER decision.
- [ ] Item-level mid-process functions (remaining quality and remaining
      expected votes from state `(n1, n2)` integrated over the prior on `p`).
      Final names TBD pending HANDOVER decision.
- [ ] `method_of_moments(sample_accuracies)` — derive Beta(α, β) parameters
      from a sample of observed worker accuracies (returns `(alpha, beta)`,
      with the `ν = α + β` convention from the paper).
- [ ] All posterior functions accept a `Prior` instance or a callable
      density on `p`; they should not require numeric integration tuning
      from the caller for the standard Beta and mixture-Beta cases.

## F. Design helpers — `design.py`

- [x] `recommend_delta(phi, target_quality)` — smallest integer `δ` such
      that `Q(φ, δ) ≥ target_quality`.
- [x] `expected_cost(phi, delta, cost_per_vote=1.0)` — wraps `expected_votes`.
- [x] `cost_for_target_quality(phi, target_quality, cost_per_vote=1.0)` —
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
- [ ] `test_notebooks.py` (or `pytest --nbmake examples/`) — execute every
      notebook under `examples/` end-to-end and fail on errors.
- [ ] CI green on the supported Python matrix.

## H. Documentation and example notebooks

### Example notebooks — `examples/`

Each notebook should be self-contained, runnable on a fresh checkout with
`pip install -e .[examples]`, and cite the paper section it illustrates.

- [ ] `01-quickstart.ipynb` — minimal "hello world" using one function from
      each module.
- [ ] `02-choosing-delta.ipynb` — given `φ`, pick `δ` for a target quality
      and budget (§4 + design helpers).
- [ ] `03-in-flight-monitoring.ipynb` — given current `(n1, n2)`, what is the
      expected remaining cost and quality, and when should you stop?
- [ ] `04-comparing-worker-pools.ipynb` — equivalence and payment across two
      pools with different `φ` (§6).
- [ ] `05-bayesian-unknown-p.ipynb` — estimating `p` from historical labels
      with a Beta or mixture-of-Betas prior (§5).
- [ ] `06-reproducing-paper-figures.ipynb` — regenerate at least one figure
      from the paper using only the `deltavote` API. Acts as an
      end-to-end sanity check that the package matches the published numbers.

### Documentation site — `docs/`

- [ ] Choose toolchain: **`mkdocs` + `mkdocs-material` + `mkdocs-jupyter`**
      (recommended) or `sphinx` + `myst-nb`. Record the decision in
      `AGENTS.md`.
- [ ] `docs/index.md` — landing page with quickstart and links.
- [ ] `docs/api/` — per-module API reference, auto-generated from docstrings
      (`mkdocstrings` or `sphinx.ext.autodoc`).
- [ ] `docs/formula-table.md` — one row per public function, mapping it to
      the paper's theorem/equation number.
- [ ] `docs/notebooks/` — rendered versions of every notebook in `examples/`.
- [ ] `docs/paper.md` — short page citing the paper and linking to the
      published version (or arXiv preprint) once available.
- [ ] Per-module docstrings citing the paper result implemented — required
      for the API reference to render usefully.
- [ ] GitHub Pages deployment configured in `.github/workflows/docs.yml`.

## I. Integration with the paper repository

These tasks live in `ipeirotis/margin-voting`, not in `deltavote`, but
they should ship together with the package release.

- [ ] Footnote in §4 of `main.tex` pointing to the package and its PyPI
      page.
- [ ] Line in the reproducibility checklist appendix referencing the
      package.
- [ ] One companion notebook in `notebooks/` of the paper repo that uses
      `deltavote` to regenerate a figure, as a sanity check that the API
      reproduces the published numbers. (May share content with
      `examples/06-reproducing-paper-figures.ipynb`.)

## Out of scope — do not add

The following are out of scope for the **installed package**. Items marked
with † may appear in notebooks or the docs site if clearly labelled.

- Multi-class extensions (paper §9 future work).
- Continuous-vote or weighted-vote variants (paper §9 future work).
- Iterative refinement or active-learning policies (paper §9 future work).
- Dataset loaders inside the package (notebooks may load data locally). †
- `deltavote.plot` or any plotting module in the wheel (notebooks may use
  `matplotlib` directly). †
- Worker assignment, scheduling, or any orchestration logic.
- Anything not derivable from a numbered result in the paper.
