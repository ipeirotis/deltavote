"""Equivalence across worker pools of different accuracy (paper §6).

Given two worker pools with odds ``phi1 = p1/(1-p1)`` and
``phi2 = p2/(1-p2)``, these functions answer two design questions:

* What threshold ``delta2`` makes pool 2 deliver the same consensus
  quality as pool 1 running at ``delta1``?  (Theorem 5)
* What per-vote payment ratio equalises the requester's total expected
  cost per item across the two pools, once their qualities are matched?
  (Theorem 6)

Both theorems assume the two pools lie on the *same side* of chance —
either both ``p > 1/2`` or both ``p < 1/2`` — so that ``ln(phi1)`` and
``ln(phi2)`` share a sign and the matched threshold is positive. The
opposite-side case is handled by inverting the labels of the below-chance
pool, which replaces ``phi`` with ``1/phi`` (see the §6 remark).

All functions accept Python scalars or NumPy arrays and follow NumPy
broadcasting rules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from deltavote.core import _validate_delta, _validate_phi


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _reject_random(phi: np.ndarray, name: str) -> None:
    """A random pool (phi == 1, p == 1/2) has no quality/payment equivalent."""
    if np.any(phi == 1.0):
        raise ValueError(
            f"{name} == 1 corresponds to a random pool (p = 1/2); "
            "consensus quality is 1/2 regardless of delta, so no "
            "quality-matched threshold or payment ratio is defined"
        )


def _require_same_side(phi1: np.ndarray, phi2: np.ndarray) -> None:
    """Both pools must sit on the same side of chance (Theorem 5 hypothesis)."""
    opposite = (phi1 - 1.0) * (phi2 - 1.0) < 0.0
    if np.any(opposite):
        raise ValueError(
            "phi1 and phi2 must lie on the same side of chance "
            "(both > 1 or both < 1) so that ln(phi1) and ln(phi2) share a "
            "sign and delta2 > 0; for an opposite-side pool, invert its "
            "labels (replace phi with 1/phi) before matching"
        )


# ---------------------------------------------------------------------------
# Theorem 5 — quality-equivalent threshold
# ---------------------------------------------------------------------------

def equivalent_delta(phi1: ArrayLike, delta1: ArrayLike, phi2: ArrayLike) -> np.ndarray:
    """Threshold for pool 2 that matches pool 1's quality (Theorem 5, §6).

    δ₂ = δ₁ · ln(φ₁) / ln(φ₂)

    Setting ``Q(phi1, delta1) == Q(phi2, delta2)`` (Theorem 1) forces
    ``phi1**delta1 == phi2**delta2``; taking logs and solving for
    ``delta2`` gives the formula above.

    The returned value is the *continuous* matched threshold. Because the
    process requires an integer δ, exact matching is generally impossible:
    rounding up to ``ceil(delta2)`` guarantees quality at least as high as
    pool 1's target (the conservative choice), while rounding down yields
    slightly lower quality.

    Both pools must lie on the same side of chance (both ``phi > 1`` or
    both ``phi < 1``); otherwise a ``ValueError`` is raised. For an
    opposite-side pool, invert its labels (use ``1/phi``) first.
    """
    phi1 = _validate_phi(phi1)
    phi2 = _validate_phi(phi2)
    delta1 = _validate_delta(delta1)
    _reject_random(phi1, "phi1")
    _reject_random(phi2, "phi2")
    _require_same_side(phi1, phi2)

    return delta1.astype(float) * np.log(phi1) / np.log(phi2)


# ---------------------------------------------------------------------------
# Theorem 6 — cost-equivalent payment
# ---------------------------------------------------------------------------

def equivalent_payment(phi: ArrayLike) -> np.ndarray:
    """Relative per-vote payment for a pool with odds ``phi`` (Theorem 6, §6).

    pay(φ) ∝ ln(φ) · (φ − 1) / (φ + 1)

    This is defined only up to a proportionality constant: it is the
    payment *ratio* between pools that is meaningful, not the absolute
    value. For two quality-matched pools, Theorem 6 gives

        pay(φ₁) / pay(φ₂)
            = ln(φ₁)/ln(φ₂) · (φ₂+1)/(φ₁+1) · (φ₁−1)/(φ₂−1),

    which equals ``equivalent_payment(phi1) / equivalent_payment(phi2)``.

    The function is non-negative and symmetric under ``phi ↔ 1/phi``
    (a below-chance pool is as informative as its label-inverted mirror).
    A random pool (``phi == 1``) yields ``pay == 0``: it carries no
    information and warrants no per-vote payment.
    """
    phi = _validate_phi(phi)
    return np.log(phi) * (phi - 1.0) / (phi + 1.0)


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

def quality_matched_pools(
    phi1: ArrayLike, delta1: ArrayLike, phi_list: ArrayLike
) -> np.ndarray:
    """Matched threshold for each pool in ``phi_list`` (Theorem 5, §6).

    Convenience wrapper around :func:`equivalent_delta`: given a reference
    pool ``(phi1, delta1)``, return the continuous threshold ``delta2``
    that makes each pool in ``phi_list`` deliver the same consensus
    quality.

    Every pool in ``phi_list`` must lie on the same side of chance as
    ``phi1`` (see :func:`equivalent_delta`).
    """
    phi_list = _validate_phi(phi_list)
    return equivalent_delta(phi1, delta1, phi_list)
