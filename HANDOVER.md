# Handover — `deltavote`

This document briefs a fresh Claude Code session that has just opened the
`deltavote` repository. Read it end-to-end before doing any work.

## What this project is

`deltavote` is a Python package implementing the formulas and estimators
from the paper:

> Boyarskaya, M. and Ipeirotis, P. *Theoretical Foundations of δ-margin
> Majority Voting for Quality Assurance in High-Stakes Machine Learning.*
> Submitted to the Journal of Artificial Intelligence Research (JAIR).

The package is intentionally small. Its sole job is to make the paper's
results usable from Python with `numpy` and `scipy` as the only runtime
dependencies. Every public function corresponds one-to-one to a numbered
result (theorem, proposition, or worked example) in the paper.

**Current status: planning only.** No source code, tests, or
`pyproject.toml` exist yet. The repo contains three files: `README.md`,
`AGENTS.md`, `TASKS.md`. Your job is to start implementing v0.1.

## Authoritative documents in this repo

Read these in order:

1. **`AGENTS.md`** — design principles, paper-to-code notation map, module
   layout, testing conventions, out-of-scope list. This is the canonical
   spec.
2. **`TASKS.md`** — v0.1 task list, grouped A through I, in roughly
   dependency order. Treat it as a checklist.
3. **`README.md`** — short public-facing overview. Update its quickstart
   once functions exist.

If any instruction here conflicts with `AGENTS.md`, `AGENTS.md` wins.

## The paper repository

The paper itself lives in a separate, private GitHub repository:
`ipeirotis/margin-voting`. Treat it as **read-only reference material**.

What you need from there:

- The compiled PDF (`main.pdf` after building) is the authoritative
  statement of every formula you implement. Section numbers cited in
  `AGENTS.md` and `TASKS.md` refer to that document.
- `tex_files/4-model-properties.tex` — Theorems 1–4 (`core.py`, `inflight.py`).
- `tex_files/5-unknown_p.tex` — Bayesian estimation (`bayes.py`).
- `tex_files/6-payment.tex` — Theorems 5–6 (`equivalence.py`).
- `tex_files/appendices/` — proofs and worked-example tables that become
  unit tests.
- `scripts/verify_numerics.py` — numerical checks already encoded for the
  paper's worked examples; port these into `tests/test_paper_examples.py`.

Do **not** push to `ipeirotis/margin-voting` from a `deltavote` session.
The only changes that repo needs (a footnote in §4, a reproducibility
checklist entry, an optional companion notebook) are listed in
`TASKS.md` section I and should be made in a separate session that has
the paper repo as its working directory.

If you do not have access to `ipeirotis/margin-voting`, ask the user to
either grant access or paste the relevant theorem statements into the
session before you implement them. Do not guess or reconstruct the
formulas from memory.

## Open decisions to confirm before implementing `bayes.py`

These two questions were flagged at the end of the planning conversation
and have not been resolved. Get an explicit answer from the user before
writing code in `bayes.py`.

### 1. Posterior-predictive functions: population-level, item-level, or both?

`TASKS.md` section E lists `posterior_quality(prior, delta, n1, n2)` and
`posterior_expected_votes(prior, delta, n1, n2)`. The signature does not
disambiguate two distinct use cases:

- **Population-level.** The posterior on `p` comes from *historical*
  labels pooled across many items; `(n1, n2)` is the from-scratch state
  for a *new* item. Answers: "for a typical new item from this pool,
  what quality and cost should I expect?"
- **Item-level mid-process.** The prior on `p` is combined with the
  current item's `(n1, n2)` to give *remaining* quality and *remaining*
  expected votes. Combines `bayes` and `inflight`.

The cleanest resolution is to support both with distinct names, e.g.
`predictive_quality(prior, delta)` (no state argument; integrates
from-scratch `Q` over the prior on `p`) and
`remaining_quality_bayes(prior, delta, n1, n2)` (integrates the
*remaining* quality from state `(n1, n2)` over the prior on `p`). Ask
the user before settling on names and signatures.

### 2. Per-item online updating of the posterior on `p`?

The paper estimates `p` from historical labels across many items and
treats it as fixed for the current item. The planning conversation
asked whether the package should also support *updating the posterior
on `p` as votes arrive on the current item*. This is a distinct
inference task — it conflates "what is this worker pool's accuracy?"
with "what is this item's label?"

The default decision in `AGENTS.md` is **out of scope**, matching the
paper. Confirm with the user before implementing it, and if it stays
out of scope, add an explicit one-line note to `AGENTS.md` so the
omission is a stated choice rather than a silent gap.

## Bootstrap checklist for your first session

1. Read `AGENTS.md` and `TASKS.md` in full.
2. Ask the user to resolve the two open decisions above.
3. Confirm access to `ipeirotis/margin-voting` for paper reference.
4. Confirm the license (default suggestion in `AGENTS.md`: MIT).
5. Start with `TASKS.md` section A (scaffolding) — `pyproject.toml`,
   `LICENSE`, `.gitignore`, CI workflow, `src/deltavote/__init__.py`.
6. Then section B (`core.py`), since it is the foundation everything
   else depends on, and its worked examples are the easiest tests to
   port from the paper.
7. After section B is green in CI, work sections C, D, E in parallel
   if helpful — they share no internal dependencies beyond `core.py`.
8. Section F (`design.py`) depends on B; section G (tests) is
   continuous and should grow alongside each module.

## Working conventions

- **Do not invent functions.** If a function does not correspond to a
  numbered result in the paper, do not add it. Propose it to the user
  for a follow-up paper instead.
- **Cite the paper in every public docstring.** Include the theorem or
  equation number, e.g. *"Implements Theorem 2 (Section 4)."*
- **Vectorize.** Functions accept Python scalars or NumPy arrays;
  follow NumPy broadcasting rules.
- **Validate at module boundaries.** Raise `ValueError` for invalid
  `phi`, `delta`, or `(n1, n2)` rather than coercing silently.
- **No backward-compatibility shims.** This is pre-v0.1. If you change
  a signature, change it everywhere.
- **No CHANGELOG entries yet.** Start a `CHANGELOG.md` only once v0.1
  is tagged on PyPI.

## Definition of done for v0.1

- All tasks in `TASKS.md` sections A–H are checked off.
- `pytest` passes on Python 3.9 through 3.12 in CI.
- Every worked numerical example in the paper has a corresponding test.
- The README quickstart runs end-to-end without modification.
- The package is published to PyPI as `deltavote==0.1.0`.
- `TASKS.md` section I (paper-repo integration) is queued as a separate
  PR against `ipeirotis/margin-voting` — not blocked on, but tracked.
