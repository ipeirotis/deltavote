"""Closed-form properties of the δ-margin voting process (paper §4).

All functions accept Python scalars or NumPy arrays for ``phi`` and
``delta`` and follow NumPy broadcasting rules.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import expit


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _validate_phi(phi: ArrayLike) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    if np.any(~np.isfinite(phi)) or np.any(phi <= 0):
        raise ValueError("phi must be finite and positive (phi = p / (1 - p) with 0 < p < 1)")
    return phi


def _validate_delta(delta: ArrayLike) -> np.ndarray:
    delta = np.asarray(delta)
    if not np.issubdtype(delta.dtype, np.integer):
        delta_float = np.asarray(delta, dtype=float)
        if not np.all(delta_float == np.floor(delta_float)):
            raise ValueError("delta must be a positive integer")
        delta = delta_float.astype(int)
    if np.any(delta < 1):
        raise ValueError("delta must be >= 1")
    return delta


# ---------------------------------------------------------------------------
# Theorem 4.1 — consensus quality
# ---------------------------------------------------------------------------

def consensus_quality(phi: ArrayLike, delta: ArrayLike) -> np.ndarray:
    """Probability that the consensus label is correct (Theorem 4.1, §4.2).

    Q(φ, δ) = φ^δ / (1 + φ^δ) = σ(δ · ln φ)

    Computed via the logistic sigmoid to stay overflow-safe in both
    directions (large or small φ^δ).
    """
    phi = _validate_phi(phi)
    delta = _validate_delta(delta)
    return expit(delta.astype(float) * np.log(phi))


# ---------------------------------------------------------------------------
# Theorem 4.3 — expected votes
# ---------------------------------------------------------------------------

def expected_votes(phi: ArrayLike, delta: ArrayLike) -> np.ndarray:
    """Expected number of votes to reach consensus (Theorem 4.3, §4.3.1).

    E[n | φ, δ] = δ · (φ+1)/(φ-1) · (φ^δ - 1)/(φ^δ + 1)
                = δ · (φ+1)/(φ-1) · tanh(δ · ln(φ) / 2)

    When φ = 1 (random voter), the limit is δ². ``tanh`` is overflow-safe
    in both the large-φ and small-φ directions.
    """
    phi = _validate_phi(phi)
    delta = _validate_delta(delta)

    scalar_input = phi.ndim == 0 and delta.ndim == 0
    phi = np.atleast_1d(phi)
    delta = np.atleast_1d(delta)
    shape = np.broadcast_shapes(phi.shape, delta.shape)
    result = np.empty(shape, dtype=float)

    phi_b = np.broadcast_to(phi, shape)
    delta_b = np.broadcast_to(delta, shape)

    random = phi_b == 1.0
    if np.any(random):
        d_r = delta_b[random].astype(float)
        result[random] = d_r ** 2

    nr = ~random
    if np.any(nr):
        p = phi_b[nr]
        d = delta_b[nr].astype(float)
        result[nr] = d * (p + 1.0) / (p - 1.0) * np.tanh(d * np.log(p) / 2.0)

    if scalar_input:
        return result.squeeze()
    return result


# ---------------------------------------------------------------------------
# Theorem 4.4 — variance of votes
# ---------------------------------------------------------------------------

def _quarter_square(z: int) -> int:
    """h(z) = floor(z² / 4), the quarter-squares sequence."""
    return z * z // 4


def var_votes(phi: ArrayLike, delta: ArrayLike) -> np.ndarray:
    """Variance of the number of votes to consensus (Theorem 4.4, §4.3.2).

    When φ = 1, Var = 2·δ²·(δ² − 1) / 3.

    Since Var is symmetric under φ ↔ 1/φ, the internal computation maps
    φ > 1 to its reciprocal to keep all intermediate powers bounded.
    """
    phi = _validate_phi(phi)
    delta = _validate_delta(delta)

    scalar_input = phi.ndim == 0 and delta.ndim == 0
    phi = np.atleast_1d(phi)
    delta = np.atleast_1d(delta)
    shape = np.broadcast_shapes(phi.shape, delta.shape)
    result = np.empty(shape, dtype=float)

    phi_b = np.broadcast_to(phi, shape)
    delta_b = np.broadcast_to(delta, shape)

    random = phi_b == 1.0
    if np.any(random):
        d_r = delta_b[random].astype(float)
        result[random] = 2.0 * d_r ** 2 * (d_r ** 2 - 1.0) / 3.0

    nr = ~random
    if np.any(nr):
        p_arr = phi_b[nr]
        d_arr = delta_b[nr]

        out = np.empty(p_arr.shape, dtype=float)
        for idx in range(len(p_arr)):
            p = float(p_arr[idx])
            d = int(d_arr[idx])
            if d == 1:
                out[idx] = 0.0
                continue
            if p > 1.0:
                p = 1.0 / p
            inner = _quarter_square(d) * p ** (d - 2)
            for i in range(1, d - 1):
                h = _quarter_square(d - i)
                inner += h * (p ** (d + i - 2) + p ** (d - i - 2))
            prefactor = 4.0 * d * p * ((p + 1.0) / (p ** d + 1.0)) ** 2
            out[idx] = prefactor * inner
        result[nr] = out

    if scalar_input:
        return result.squeeze()
    return result


# ---------------------------------------------------------------------------
# Theorem 4.5 — pmf of votes to consensus
# ---------------------------------------------------------------------------

def _build_Q_matrix(p: float, delta: int) -> np.ndarray:
    """Build the (2δ-1) × (2δ-1) transient-state transition matrix Q."""
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
    """Build the (2δ-1) × 2 absorption matrix R."""
    n = 2 * delta - 1
    R = np.zeros((n, 2), dtype=float)
    q = 1.0 - p
    R[0, 0] = q       # absorb at -δ
    R[n - 1, 1] = p   # absorb at +δ
    return R


def votes_pmf(m: ArrayLike, phi: ArrayLike, delta: ArrayLike) -> np.ndarray:
    """P(process terminates after exactly m votes) (Theorem 4.5, §4.3.3).

    pmf(m) = z · Q^{m-1} · R · 1

    Returns 0 for m < δ and for m with wrong parity (m ≢ δ mod 2).
    """
    phi_arr = _validate_phi(phi)
    delta_arr = _validate_delta(delta)
    m_arr = np.asarray(m)
    if np.issubdtype(m_arr.dtype, np.floating):
        if not np.all(m_arr == np.floor(m_arr)):
            raise ValueError("m must be a non-negative integer (number of votes)")
        m_arr = m_arr.astype(int)
    elif not np.issubdtype(m_arr.dtype, np.integer):
        raise ValueError("m must be a non-negative integer (number of votes)")

    scalar_input = phi_arr.ndim == 0 and delta_arr.ndim == 0 and m_arr.ndim == 0
    phi_arr = np.atleast_1d(phi_arr)
    delta_arr = np.atleast_1d(delta_arr)
    m_arr = np.atleast_1d(m_arr)

    shape = np.broadcast_shapes(m_arr.shape, phi_arr.shape, delta_arr.shape)
    m_b = np.broadcast_to(m_arr, shape).ravel()
    phi_b = np.broadcast_to(phi_arr, shape).ravel()
    delta_b = np.broadcast_to(delta_arr, shape).ravel()

    result = np.zeros(len(m_b), dtype=float)

    for idx in range(len(m_b)):
        mi = int(m_b[idx])
        pi = float(phi_b[idx])
        di = int(delta_b[idx])
        if mi < di or (mi - di) % 2 != 0:
            continue
        p = pi / (1.0 + pi)
        Q_mat = _build_Q_matrix(p, di)
        R_mat = _build_R_matrix(p, di)
        n = 2 * di - 1
        z = np.zeros(n)
        z[di - 1] = 1.0  # state 0 is at index δ-1
        Qpow = np.linalg.matrix_power(Q_mat, mi - 1)
        result[idx] = z @ Qpow @ R_mat @ np.ones(2)

    result = result.reshape(shape)
    if scalar_input:
        return result.squeeze()
    return result
