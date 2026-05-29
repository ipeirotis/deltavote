# CLAUDE.md — `deltavote`

Operational notes for Claude Code sessions in this repo. Read alongside
[`AGENTS.md`](AGENTS.md) (canonical design spec), [`HANDOVER.md`](HANDOVER.md)
(session briefing), and [`TASKS.md`](TASKS.md) (v0.1 task list).

## What this is

A lean Python package (`numpy` + `scipy` only) implementing the formulas and
estimators from the paper:

> Boyarskaya, M. and Ipeirotis, P. *A Design Calculus for δ-margin Majority
> Voting: Quality, Cost, and Payment.* Submitted to JAIR.

Every public function maps one-to-one to a numbered result in the paper.
**If the code and the paper disagree, the paper wins.**

## Paper reference access (gh)

The authoritative source is the **private** companion repo
`ipeirotis/margin-voting`. In Claude Code on the web sessions you **can read
it**:

- The `gh` CLI is installed and authenticated via the `GITHUB_TOKEN`
  environment variable that the web environment injects (account
  `ipeirotis`, `repo` scope). The GitHub **MCP** tools are scoped to
  `ipeirotis/deltavote` only — use `gh` (not MCP) for the paper repo.
- Clone it read-only for reference, e.g.:
  ```bash
  gh repo clone ipeirotis/margin-voting -- --depth 1   # into /tmp, outside this tree
  ```
- Key files: `tex_files/4-model-properties.tex` (Theorems 4.1–4.5),
  `tex_files/5-unknown_p.tex` (Props. 5.1–5.3), `tex_files/6-payment.tex`
  (Theorems 6.1–6.2), `tex_files/8-example.tex` (AML case study, the test
  constants), `scripts/verify_numerics.py`, and the compiled PDF under
  `jair-submission/`.
- **Do not** print or commit the token value, and **do not push** to
  `ipeirotis/margin-voting` from a `deltavote` session (read-only).

## Theorem numbering (read before citing the paper)

The compiled paper numbers results **by section**, so docstrings cite:

| Result | Paper label | Code |
|---|---|---|
| Consensus quality `Q` | **Theorem 4.1** (§4.2) | `core.consensus_quality` |
| Weak jury theorem | Corollary 4.2 | — |
| Expected votes `E[n]` | **Theorem 4.3** (§4.3.1) | `core.expected_votes` |
| Variance `Var[n]` | **Theorem 4.4** (§4.3.2) | `core.var_votes` |
| Distribution `pmf(m)` | **Theorem 4.5** (§4.3.3) | `core.votes_pmf` |
| Shifted-start `H_s`/`T_s`, `V_s`, pmf | Eqs. H_s/T_s + **Prop. 5.1** (§5.1) | `inflight.*` |
| Bayesian remaining estimates | **Prop. 5.1** | `bayes.*` (planned) |
| Model-averaged `Q̂` | **Prop. 5.2** | `bayes.*` (planned) |
| Mixture-of-Betas conjugacy | **Prop. 5.3** | `bayes.*` (planned) |
| δ-equivalence | **Theorem 6.1** (§6.1) | `equivalence.equivalent_delta` |
| Payment equivalence | **Theorem 6.2** (§6.2) | `equivalence.equivalent_payment` |

Note `E[n]` is **4.3, not 4.2** (Corollary 4.2 sits between). The paper's own
`CLAUDE.md` still uses the older flat "Theorem 1–6" scheme — the compiled PDF
is authoritative. Anchor citations by the result's *name* so they survive any
camera-ready renumbering.

## `bayes.py` (§5) is the next module

Resolved decisions (2026-05-29): support **both** population-level (`Q^fresh`,
`s=0`) and item-level (`Q^rem`, Prop. 5.1) predictive functions, and **do**
implement per-item online updating of the posterior on `p` (it is the paper's
monitoring mode). Canonical prior is **`Beta(k, 1)`** (the truncated symmetric
prior is legacy). The headline quantity is the model-averaged `Q̂` (Prop. 5.2)
with `H_c`/`H_i` averaging and the terminal rising-factorial closed form;
`method_of_moments` is item-level with binomial-noise denoising. See `TASKS.md`
§E and `HANDOVER.md`.

## Dev workflow

```bash
pip install -e .            # runtime: numpy, scipy
pip install pytest          # (dev) test runner
python -m pytest -q         # full suite (203 passing as of this writing)
```

- Develop on the branch assigned for the session; do not push to `main`
  without explicit permission. Open a PR only when asked.
- Validate inputs at module boundaries (raise `ValueError`); vectorize over
  `phi`, `delta`, `n1`, `n2` with NumPy broadcasting; no I/O in the package.
- Add a worked-example test for every paper number you rely on.
