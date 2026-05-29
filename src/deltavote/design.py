"""Practitioner design helpers built on the §4 closed forms.

These compose the core results into the questions a requester actually
asks: "how large must δ be to hit a quality target?" and "what will that
cost?" They add no new theory — every value derives from
:func:`deltavote.core.consensus_quality` and
:func:`deltavote.core.expected_votes`.

All functions accept Python scalars or NumPy arrays and follow NumPy
broadcasting rules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import logit

from deltavote.core import _validate_phi, consensus_quality, expected_votes


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _validate_target_quality(target_quality: ArrayLike) -> np.ndarray:
    q = np.asarray(target_quality, dtype=float)
    if np.any(~np.isfinite(q)) or np.any(q <= 0.0) or np.any(q >= 1.0):
        raise ValueError("target_quality must lie strictly in (0, 1)")
    return q


# ---------------------------------------------------------------------------
# Choosing delta for a quality target
# ---------------------------------------------------------------------------

def recommend_delta(phi: ArrayLike, target_quality: ArrayLike) -> np.ndarray:
    """Smallest integer δ with ``Q(phi, delta) >= target_quality`` (§4).

    Since ``Q(φ, δ) = σ(δ · ln φ)`` is increasing in δ for a competent
    pool (``φ > 1``), the requirement ``Q >= q`` solves to

        δ >= logit(q) / ln(φ),

    and the smallest admissible threshold is ``max(1, ceil(...))``.

    Requires ``phi > 1``: a random or below-chance pool (``phi <= 1``)
    cannot reach a quality target above its single-vote accuracy by
    adding votes, so no finite δ exists and a ``ValueError`` is raised.
    """
    phi = _validate_phi(phi)
    q = _validate_target_quality(target_quality)

    if np.any(phi <= 1.0):
        raise ValueError(
            "recommend_delta requires phi > 1 (a competent pool); adding "
            "votes cannot raise the quality of a random or below-chance pool"
        )

    needed = logit(q) / np.log(phi)
    delta = np.maximum(np.ceil(needed).astype(int), 1)

    # Guard against floating-point over-rounding at exact quality
    # boundaries: e.g. Q(2, 2) == 0.8 exactly, yet logit(0.8) / ln(2)
    # evaluates to 2.0000000000000004, so ceil would return 3. If the
    # next-smaller threshold already meets the target, it is the true
    # answer. (The float error is far below 1, so a single step suffices.)
    lower = np.maximum(delta - 1, 1)
    step_down = (lower < delta) & (consensus_quality(phi, lower) >= q)
    delta = np.where(step_down, lower, delta)
    return delta


# ---------------------------------------------------------------------------
# Cost helpers
# ---------------------------------------------------------------------------

def expected_cost(
    phi: ArrayLike, delta: ArrayLike, cost_per_vote: ArrayLike = 1.0
) -> np.ndarray:
    """Expected cost to label one item (Theorem 2, §4).

    ``cost_per_vote · E[n_votes | phi, delta]``. Validation of ``phi`` and
    ``delta`` is delegated to :func:`deltavote.core.expected_votes`.
    """
    cost_per_vote = np.asarray(cost_per_vote, dtype=float)
    if np.any(cost_per_vote < 0.0):
        raise ValueError("cost_per_vote must be non-negative")
    return cost_per_vote * expected_votes(phi, delta)


def cost_for_target_quality(
    phi: ArrayLike, target_quality: ArrayLike, cost_per_vote: ArrayLike = 1.0
) -> np.ndarray:
    """Expected cost to reach a quality target (§4 + design helpers).

    Composes :func:`recommend_delta` (smallest δ achieving
    ``target_quality``) with :func:`expected_cost` at that δ.
    """
    delta = recommend_delta(phi, target_quality)
    return expected_cost(phi, delta, cost_per_vote)
