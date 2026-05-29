# Handover — `deltavote`

This document briefs a fresh Claude Code session that has just opened the
`deltavote` repository. Read it end-to-end before doing any work.

## What this project is

`deltavote` is a Python package implementing the formulas and estimators
from the paper:

> Boyarskaya, M. and Ipeirotis, P. *A Design Calculus for δ-margin
> Majority Voting: Quality, Cost, and Payment.* Submitted to the Journal
> of Artificial Intelligence Research (JAIR).

The **installed package** is intentionally small. Its sole job is to make
the paper's results usable from Python with `numpy` and `scipy` as the
only runtime dependencies. Every public function corresponds one-to-one
to a numbered result (theorem, proposition, or worked example) in the
paper.

The **project as a whole** is broader than the installed package: it also
contains example notebooks under `examples/` and a documentation site
under `docs/`. Both are first-class deliverables — they are how a
practitioner will discover and learn the package — they are just not
shipped inside the wheel.

**Current status: in progress.** `pyproject.toml`, the package skeleton,
and four of the five modules are implemented and tested (203 tests
passing): `core.py` (§4, Theorems 4.1–4.5), `inflight.py` (§5.1
shifted-start estimators), `equivalence.py` (§6, Theorems 6.1–6.2), and
`design.py` (practitioner helpers). The remaining module is `bayes.py`
(§5 Bayesian estimators) — the two open decisions below have now been
resolved (see "Open decisions", now answered). Example notebooks and the
docs site (TASKS.md sections H) are not yet started.

## Authoritative documents in this repo

Read these in order:

1. **`AGENTS.md`** — design principles, paper-to-code notation map,
   repository layout (installed package vs project repo), examples and
   documentation policy, testing conventions, out-of-scope list. This
   is the canonical spec.
2. **`TASKS.md`** — v0.1 task list, grouped A through I, in roughly
   dependency order. Treat it as a checklist. Sections H (docs and
   notebooks) and I (paper-repo integration) are part of v0.1, not
   afterthoughts.
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

In Claude Code on the web sessions, **read access to the paper repo is
available via the `gh` CLI** (the environment provides an authenticated
`GITHUB_TOKEN`). Clone it read-only for reference, e.g.
`gh repo clone ipeirotis/margin-voting -- --depth 1` into `/tmp`, and read
`tex_files/` and the compiled PDF under `jair-submission/`. See
[`CLAUDE.md`](CLAUDE.md) for details. Still **do not push** to
`ipeirotis/margin-voting` from a `deltavote` session. If `gh` access is
ever unavailable, ask the user to grant access or paste the relevant
theorem statements — do not guess or reconstruct the formulas from memory.

## Open decisions for `bayes.py` — RESOLVED (2026-05-29)

Both questions below have now been answered by the author (P. Ipeirotis).
The original framing is kept for context; the **Resolution** notes are
authoritative and align with the current §5 of the paper.

> **Note on §5.** Since these questions were first written, §5 of the
> paper has evolved substantially. The canonical prior is now
> **`Beta(k, 1)`** (integer `k ≥ 2`), *not* a truncated symmetric prior
> (which the paper now calls a "legacy" earlier-version choice). The
> central monitoring-mode quantity is the **model-averaged correctness
> estimate `Q̂`** (Proposition 5.2), which mixes the "majority is correct"
> (`H_c`) and "majority is incorrect" (`H_i`) hypotheses — a symmetric
> prior leaves the vote split uninformative, so the asymmetry of
> `Beta(k, 1)` is what makes `Q̂` meaningful. `bayes.py` must implement
> `Q̂` (Prop. 5.2), the per-hypothesis posteriors, the terminal-state
> rising-factorial closed form (Eq. terminal_Hc_closed_form), optional
> non-uniform class priors (base rate `π`), and the item-level
> method-of-moments prior fit **with the binomial-noise denoising
> correction** (`ν = p̄(1−p̄)/Var̂(pᵢ) − 1`). See `TASKS.md` §E.

### 1. Posterior-predictive functions: population-level, item-level, or both?

`TASKS.md` section E lists posterior functions that take a prior on `p`
and a state `(n1, n2)`, but the signature does not disambiguate two
distinct use cases:

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

> **Resolution: support BOTH.** Provide population-level
> posterior-predictive functions (the from-scratch / `s = 0` integral,
> i.e. the paper's `Q^fresh`) *and* item-level mid-process functions (the
> remaining quality/cost from state `(n1, n2)`, i.e. `Q^rem`/`T^rem` from
> Proposition 5.1). Keep them under distinct names so the two questions —
> "what quality should a fresh item from this pool reach?" vs. "given the
> votes on *this* item, what is the remaining quality/cost?" — are never
> conflated. The deployment estimate `Q̂` (Prop. 5.2) is the item-level
> answer once `H_c`/`H_i` averaging is applied.

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

> **Resolution: IN scope — bring it in.** This is no longer "beyond the
> paper": §5 *monitoring mode* (Propositions 5.1–5.2) updates the
> `Beta(α + n1, β + n2)` posterior on `p` from the current item's running
> counts as votes arrive, and the `H_c`/`H_i` model-averaging in `Q̂`
> (Prop. 5.2) is precisely what keeps "what is the pool's accuracy?"
> separate from "which label is correct on this item?". Implement the
> online per-item update as the monitoring-mode estimator (`Q̂`, expected
> remaining votes, and the posterior-predictive variance via the law of
> total variance). The `Q^fresh` footnote in §5.1 documents the
> alternative reading (treating `(n1, n2)` as external calibration that
> updates `p` but does not advance the walk); expose both, clearly named.

## Notebooks and documentation are in scope

A common mistake worth flagging up front: do **not** treat notebooks or
the documentation site as out of scope just because the *installed
package* keeps a lean dependency footprint. They are different things.

- **Installed package** (`src/deltavote/`): numpy + scipy only, no
  plotting, no I/O. Strict.
- **Examples** (`examples/`): Jupyter notebooks demonstrating realistic
  practitioner workflows. May use `matplotlib`, `pandas`, or load
  datasets from local paths. Committed to the repo, executed in CI,
  rendered into the docs site.
- **Documentation** (`docs/`): narrative pages + API reference +
  rendered notebooks. Built with `mkdocs` + `mkdocs-jupyter`
  (recommended) and deployed to GitHub Pages.

`TASKS.md` section H lists concrete notebook deliverables. Treat them
with the same seriousness as the source code — a brilliant API that
nobody can find or learn is worth little.

## Bootstrap checklist

Scaffolding (TASKS §A) and the `core.py` (§B), `inflight.py` (§C),
`equivalence.py` (§D), and `design.py` (§F) modules are **done and green**
(203 tests). License is MIT. The two `bayes.py` decisions are **resolved**
(above), and paper read-access is available via `gh` (see `CLAUDE.md`).
The remaining v0.1 work, in rough order:

1. Read `AGENTS.md` and `TASKS.md` in full (note the §5 evolution recorded
   in the resolved decisions above).
2. Implement **`bayes.py`** (TASKS §E) against the *current* §5: canonical
   `Beta(k, 1)` prior, mixture-of-Betas (Prop. 5.3), model-averaged `Q̂`
   (Prop. 5.2) with `H_c`/`H_i` averaging and the terminal rising-factorial
   closed form, optional base-rate `π`, both population-level (`Q^fresh`)
   and item-level (`Q^rem`, Prop. 5.1) predictive functions, and the
   item-level method-of-moments fit with binomial-noise denoising.
3. Grow the test suite (TASKS §G): port `scripts/verify_numerics.py` and
   the §5 worked examples (e.g. the `Beta(2,1)`, `δ=2`, `(2,1)` example
   gives `Q̂ ≈ 0.644`; terminal `(2,0)` at `k=2,δ=2` gives
   `P(incorrect)=1/4`).
4. Confirm the docs toolchain (default: `mkdocs` + `mkdocs-material` +
   `mkdocs-jupyter`) and build out `examples/` and `docs/` (TASKS §H).

## Working conventions

- **Do not invent functions.** If a function does not correspond to a
  numbered result in the paper, do not add it. Propose it to the user
  for a follow-up paper instead.
- **Cite the paper in every public docstring.** Include the
  section-numbered theorem/equation label, e.g. *"Implements Theorem 4.3
  (§4.3.1)."* (See the theorem-numbering note in `AGENTS.md`.)
- **Vectorize.** Functions accept Python scalars or NumPy arrays;
  follow NumPy broadcasting rules.
- **Validate at module boundaries.** Raise `ValueError` for invalid
  `phi`, `delta`, or `(n1, n2)` rather than coercing silently.
- **No backward-compatibility shims.** This is pre-v0.1. If you change
  a signature, change it everywhere — including notebooks.
- **Keep notebooks runnable.** If you change an API, update the
  notebooks in the same PR. CI executes them; broken notebooks block
  merge.
- **No CHANGELOG entries yet.** Start a `CHANGELOG.md` only once v0.1
  is tagged on PyPI.

## Definition of done for v0.1

- All tasks in `TASKS.md` sections A–H are checked off.
- `pytest` passes on Python 3.9 through 3.12 in CI.
- Every worked numerical example in the paper has a corresponding test.
- Every notebook in `examples/` executes end-to-end in CI.
- The README quickstart runs end-to-end without modification.
- The docs site is built and deployed to GitHub Pages.
- The package is published to PyPI as `deltavote==0.1.0`.
- `TASKS.md` section I (paper-repo integration) is queued as a separate
  PR against `ipeirotis/margin-voting` — not blocked on, but tracked.
