"""In-flight (mid-process) estimators for the δ-margin voting process.

These are the Gambler's-Ruin counterparts of the from-scratch quantities in
:mod:`deltavote.core`. Given the current vote counts ``(n1, n2)`` for an
item that has not yet absorbed (``|n1 - n2| < delta``), they return the
remaining quality, expected remaining votes, variance of remaining votes,
and the pmf of remaining votes.

All formulas reduce exactly to the from-scratch :mod:`deltavote.core`
expressions when ``n1 == n2`` (start state ``d = 0``). See paper §4
(Theorems 1–4) for the closed forms; the in-flight generalization is the
standard shifted Gambler's-Ruin on states ``d ∈ {-δ, ..., +δ}`` with
absorbing barriers at ``±δ`` and drift ``p = φ / (1 + φ)``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike

from deltavote.core import _validate_delta, _validate_phi


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _validate_counts(n1: ArrayLike, n2: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    n1_a = np.asarray(n1)
    n2_a = np.asarray(n2)
    for name, arr in (("n1", n1_a), ("n2", n2_a)):
        if np.issubdtype(arr.dtype, np.floating):
            if not np.all(arr == np.floor(arr)):
                raise ValueError(f"{name} must be a non-negative integer")
        elif not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(f"{name} must be a non-negative integer")
        if np.any(arr < 0):
            raise ValueError(f"{name} must be non-negative")
    return n1_a.astype(int), n2_a.astype(int)


def _prepare(
    n1: ArrayLike,
    n2: ArrayLike,
    phi: ArrayLike,
    delta: ArrayLike,
) -> tuple[bool, tuple[int, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Validate, broadcast, and check the not-yet-absorbed invariant."""
    n1_a, n2_a = _validate_counts(n1, n2)
    phi_a = _validate_phi(phi)
    delta_a = _validate_delta(delta)

    scalar = (
        n1_a.ndim == 0 and n2_a.ndim == 0
        and phi_a.ndim == 0 and delta_a.ndim == 0
    )
    n1_a = np.atleast_1d(n1_a)
    n2_a = np.atleast_1d(n2_a)
    phi_a = np.atleast_1d(phi_a)
    delta_a = np.atleast_1d(delta_a)

    shape = np.broadcast_shapes(n1_a.shape, n2_a.shape, phi_a.shape, delta_a.shape)
    d = np.broadcast_to(n1_a - n2_a, shape)
    phi_b = np.broadcast_to(phi_a, shape)
    delta_b = np.broadcast_to(delta_a, shape)

    if np.any(np.abs(d) >= delta_b):
        raise ValueError(
            "Process has already absorbed: |n1 - n2| must be strictly less than delta"
        )
    return scalar, shape, d, phi_b, delta_b


# ---------------------------------------------------------------------------
# Remaining quality (shifted Gambler's-Ruin absorption probability)
# ---------------------------------------------------------------------------

def remaining_quality(
    n1: ArrayLike,
    n2: ArrayLike,
    phi: ArrayLike,
    delta: ArrayLike,
) -> np.ndarray:
    """Probability the process absorbs at +δ given current ``(n1, n2)``.

    Standard asymmetric Gambler's-Ruin on states ``d ∈ {-δ, ..., +δ}``
    starting at ``d = n1 - n2``:

    * ``φ = 1``:  ``(d + δ) / (2δ)``
    * ``φ ≠ 1``:  ``(1 − φ^{-(d+δ)}) / (1 − φ^{-2δ})``

    Reduces to the from-scratch :func:`deltavote.core.consensus_quality`
    when ``n1 == n2``. See paper §4 (Theorem 1, shifted start).
    """
    scalar, shape, d, phi, delta = _prepare(n1, n2, phi, delta)
    result = np.empty(shape, dtype=float)

    random = phi == 1.0
    big = (~random) & (phi >= 1.0)
    small = (~random) & (phi < 1.0)

    if np.any(random):
        d_r = d[random].astype(float)
        delta_r = delta[random].astype(float)
        result[random] = (d_r + delta_r) / (2.0 * delta_r)

    # φ ≥ 1: r = 1/φ ∈ (0, 1], so r**k stays bounded.
    if np.any(big):
        r = 1.0 / phi[big]
        exp_x = d[big] + delta[big]
        exp_n = 2 * delta[big]
        result[big] = (1.0 - r ** exp_x) / (1.0 - r ** exp_n)

    # φ < 1: rewrite numerator/denominator times φ^{2δ} so all powers ≤ 1.
    if np.any(small):
        p = phi[small]
        num = p ** (2 * delta[small]) - p ** (delta[small] - d[small])
        den = p ** (2 * delta[small]) - 1.0
        result[small] = num / den

    if scalar:
        return result.squeeze()
    return result


# ---------------------------------------------------------------------------
# Remaining expected votes (Gambler's-Ruin expected absorption time)
# ---------------------------------------------------------------------------

def remaining_expected_votes(
    n1: ArrayLike,
    n2: ArrayLike,
    phi: ArrayLike,
    delta: ArrayLike,
) -> np.ndarray:
    """Expected number of remaining votes until absorption.

    From the standard Gambler's-Ruin recursion, starting at ``d = n1 − n2``:

    * ``φ = 1``:  ``δ² − d²``
    * ``φ ≠ 1``:  ``((φ+1)/(φ−1)) · (2δ · Q_d − (d + δ))``

      where ``Q_d`` is :func:`remaining_quality`.

    Reduces to :func:`deltavote.core.expected_votes` when ``n1 == n2``.
    See paper §4 (Theorem 2, shifted start).
    """
    scalar, shape, d, phi, delta = _prepare(n1, n2, phi, delta)
    result = np.empty(shape, dtype=float)

    random = phi == 1.0
    nr = ~random

    if np.any(random):
        d_r = d[random].astype(float)
        delta_r = delta[random].astype(float)
        result[random] = delta_r ** 2 - d_r ** 2

    if np.any(nr):
        p = phi[nr]
        d_nr = d[nr]
        delta_nr = delta[nr]
        # Near-random: the closed form is a 0/0 amplified by 1/(φ−1).
        # Fall back to the absorbing-chain fundamental matrix in that regime.
        near_random_nr = np.abs(p - 1.0) < _NEAR_RANDOM_TOL
        closed_nr = ~near_random_nr

        if np.any(near_random_nr):
            out = np.empty(int(np.sum(near_random_nr)), dtype=float)
            d_n = d_nr[near_random_nr]
            phi_n = p[near_random_nr]
            delta_n = delta_nr[near_random_nr]
            for k in range(out.size):
                mu, _ = _chain_mean_var(
                    int(d_n[k]), float(phi_n[k]), int(delta_n[k])
                )
                out[k] = mu
            result_nr_block = np.empty(p.shape, dtype=float)
            result_nr_block[near_random_nr] = out
        else:
            result_nr_block = np.empty(p.shape, dtype=float)

        if np.any(closed_nr):
            p_c = p[closed_nr]
            d_c = d_nr[closed_nr]
            delta_c = delta_nr[closed_nr]
            q = _remaining_quality_raw(d_c, p_c, delta_c)
            result_nr_block[closed_nr] = (
                ((p_c + 1.0) / (p_c - 1.0))
                * (2.0 * delta_c * q - (d_c + delta_c))
            )

        result[nr] = result_nr_block

    if scalar:
        return result.squeeze()
    return result


def _remaining_quality_raw(d: np.ndarray, phi: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """Vectorized quality on already-validated, equally-shaped arrays (φ ≠ 1)."""
    result = np.empty(d.shape, dtype=float)
    big = phi >= 1.0
    small = ~big
    if np.any(big):
        r = 1.0 / phi[big]
        result[big] = (1.0 - r ** (d[big] + delta[big])) / (1.0 - r ** (2 * delta[big]))
    if np.any(small):
        p = phi[small]
        num = p ** (2 * delta[small]) - p ** (delta[small] - d[small])
        den = p ** (2 * delta[small]) - 1.0
        result[small] = num / den
    return result


# ---------------------------------------------------------------------------
# Absorbing Markov chain — fundamental matrix utilities
# ---------------------------------------------------------------------------

# Tolerance for the near-random branch in `remaining_expected_votes`. When
# |φ − 1| is below this, the closed form's 0/0 amplified by 1/(φ−1) loses
# precision and we fall back to the (exact) fundamental-matrix computation.
_NEAR_RANDOM_TOL = 1e-6


def _build_Q_matrix(p: float, delta: int) -> np.ndarray:
    """Tridiagonal transient transition matrix on states {-δ+1, …, δ-1}."""
    n = 2 * delta - 1
    Q = np.zeros((n, n), dtype=float)
    q = 1.0 - p
    for row in range(n):
        if row > 0:
            Q[row, row - 1] = q
        if row < n - 1:
            Q[row, row + 1] = p
    return Q


def _build_R_matrix(p: float, delta: int) -> np.ndarray:
    """Absorption matrix: column 0 is the −δ barrier, column 1 is the +δ barrier."""
    n = 2 * delta - 1
    R = np.zeros((n, 2), dtype=float)
    R[0, 0] = 1.0 - p
    R[n - 1, 1] = p
    return R


def _chain_mean_var(d: int, phi: float, delta: int) -> tuple[float, float]:
    """Return (E[T], Var[T]) for absorption time starting at state ``d``.

    Uses the fundamental matrix ``N = (I − Q)^{-1}``:
    ``E[T] = (z N 1)``, ``E[T²] = 2 (z N² 1) − (z N 1)``.
    """
    if delta == 1:
        # The only allowed start is d = 0; absorption after exactly one vote.
        return 1.0, 0.0
    p = phi / (1.0 + phi)
    Q = _build_Q_matrix(p, delta)
    n = 2 * delta - 1
    I = np.eye(n)
    N = np.linalg.solve(I - Q, I)
    z_idx = d + delta - 1
    N1 = N @ np.ones(n)
    mu = float(N1[z_idx])
    N2_1 = N @ N1
    mu2 = 2.0 * float(N2_1[z_idx]) - mu
    return mu, mu2 - mu * mu


def remaining_var_votes(
    n1: ArrayLike,
    n2: ArrayLike,
    phi: ArrayLike,
    delta: ArrayLike,
) -> np.ndarray:
    """Variance of the number of remaining votes until absorption.

    Computed from the fundamental matrix ``N = (I − Q)^{-1}`` of the
    absorbing Markov chain on transient states ``{−δ+1, …, δ−1}``:

    ``E[T]  = z · N · 1``,
    ``E[T²] = 2 · z · N² · 1 − z · N · 1``,
    ``Var[T] = E[T²] − E[T]²``

    where ``z`` is the indicator of the starting state ``d = n1 − n2``.

    Reduces to :func:`deltavote.core.var_votes` when ``n1 == n2``. See
    paper §4 (Theorem 3, shifted start).
    """
    scalar, shape, d, phi, delta = _prepare(n1, n2, phi, delta)
    flat_d = d.ravel()
    flat_phi = phi.ravel()
    flat_delta = delta.ravel()
    out = np.empty(flat_d.shape, dtype=float)

    for idx in range(out.size):
        _, var = _chain_mean_var(
            int(flat_d[idx]), float(flat_phi[idx]), int(flat_delta[idx])
        )
        out[idx] = var

    result = out.reshape(shape)
    if scalar:
        return result.squeeze()
    return result


# ---------------------------------------------------------------------------
# Remaining-votes pmf
# ---------------------------------------------------------------------------

def remaining_votes_pmf(
    m: ArrayLike,
    n1: ArrayLike,
    n2: ArrayLike,
    phi: ArrayLike,
    delta: ArrayLike,
) -> np.ndarray:
    """P(process absorbs after exactly ``m`` more votes).

    ``pmf(m) = z · Q^{m−1} · R · 1``, where ``z`` is the indicator of the
    starting state ``d = n1 − n2`` in the transient-state space. Returns
    0 for ``m < δ − |d|`` and for ``m`` with wrong parity relative to
    ``δ − d``.

    Reduces to :func:`deltavote.core.votes_pmf` when ``n1 == n2``. See
    paper §4 (Theorem 4, shifted start).
    """
    scalar_state, shape_state, d_state, phi_state, delta_state = _prepare(n1, n2, phi, delta)

    m_arr = np.asarray(m)
    if np.issubdtype(m_arr.dtype, np.floating):
        if not np.all(m_arr == np.floor(m_arr)):
            raise ValueError("m must be a non-negative integer (number of votes)")
        m_arr = m_arr.astype(int)
    elif not np.issubdtype(m_arr.dtype, np.integer):
        raise ValueError("m must be a non-negative integer (number of votes)")

    scalar = scalar_state and m_arr.ndim == 0
    m_arr = np.atleast_1d(m_arr)

    shape = np.broadcast_shapes(m_arr.shape, shape_state)
    m_b = np.broadcast_to(m_arr, shape).ravel()
    d_b = np.broadcast_to(d_state, shape).ravel()
    phi_b = np.broadcast_to(phi_state, shape).ravel()
    delta_b = np.broadcast_to(delta_state, shape).ravel()

    result = np.zeros(m_b.size, dtype=float)
    for idx in range(m_b.size):
        mi = int(m_b[idx])
        di = int(d_b[idx])
        pi = float(phi_b[idx])
        de = int(delta_b[idx])
        if mi < de - abs(di) or (mi - (de - di)) % 2 != 0:
            continue
        p = pi / (1.0 + pi)
        Q = _build_Q_matrix(p, de)
        R = _build_R_matrix(p, de)
        n = 2 * de - 1
        z = np.zeros(n)
        z[di + de - 1] = 1.0
        Qpow = np.linalg.matrix_power(Q, mi - 1)
        result[idx] = z @ Qpow @ R @ np.ones(2)

    result = result.reshape(shape)
    if scalar:
        return result.squeeze()
    return result
