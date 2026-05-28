"""Tests for deltavote.inflight — in-flight Gambler's-Ruin estimators (§4)."""

import numpy as np
import pytest

from deltavote.core import (
    consensus_quality,
    expected_votes,
    var_votes,
    votes_pmf,
)
from deltavote.inflight import (
    remaining_expected_votes,
    remaining_quality,
    remaining_var_votes,
    remaining_votes_pmf,
)


# ── Reduction to core when (n1, n2) = (0, 0) ───────────────────────────────

class TestReductionToCore:
    """With n1 = n2 = 0 the in-flight quantities must equal the core ones."""

    @pytest.mark.parametrize("phi", [0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    @pytest.mark.parametrize("delta", [1, 2, 3, 5, 7])
    def test_quality_matches_core(self, phi, delta):
        assert remaining_quality(0, 0, phi, delta) == pytest.approx(
            float(consensus_quality(phi, delta)), rel=1e-12, abs=1e-14
        )

    @pytest.mark.parametrize("phi", [0.5, 1.0, 1.5, 2.0, 3.0, 5.0])
    @pytest.mark.parametrize("delta", [1, 2, 3, 5, 7])
    def test_expected_matches_core(self, phi, delta):
        assert remaining_expected_votes(0, 0, phi, delta) == pytest.approx(
            float(expected_votes(phi, delta)), rel=1e-12, abs=1e-14
        )

    @pytest.mark.parametrize("phi", [0.5, 1.0, 1.5, 2.0, 3.0])
    @pytest.mark.parametrize("delta", [1, 2, 3, 5])
    def test_variance_matches_core(self, phi, delta):
        assert remaining_var_votes(0, 0, phi, delta) == pytest.approx(
            float(var_votes(phi, delta)), rel=1e-10, abs=1e-12
        )

    @pytest.mark.parametrize("phi", [1.5, 2.0, 3.0])
    @pytest.mark.parametrize("delta", [2, 3, 4])
    def test_pmf_matches_core(self, phi, delta):
        m_vals = np.arange(delta, delta + 50, 2)
        for m in m_vals:
            r = float(remaining_votes_pmf(int(m), 0, 0, phi, delta))
            c = float(votes_pmf(int(m), phi, delta))
            assert r == pytest.approx(c, rel=1e-12, abs=1e-14)


# ── Sanity properties of remaining_quality ─────────────────────────────────

class TestRemainingQuality:

    def test_random_voter_linear(self):
        """φ = 1: P(absorb at +δ | d) = (d + δ) / (2δ)."""
        for delta in [2, 3, 5]:
            for d in range(-delta + 1, delta):
                # encode d = n1 - n2 with n2 = 0 when d ≥ 0, else n1 = 0.
                n1, n2 = max(d, 0), max(-d, 0)
                assert remaining_quality(n1, n2, 1.0, delta) == pytest.approx(
                    (d + delta) / (2.0 * delta)
                )

    def test_boundary_states(self):
        """At d = δ−1 P is near 1; at d = −(δ−1) it lies strictly between the
        no-drift baseline ``1/(2δ)`` and the centered ``Q(φ, δ)`` value."""
        phi, delta = 3.0, 4
        p_high = float(remaining_quality(delta - 1, 0, phi, delta))
        p_low = float(remaining_quality(0, delta - 1, phi, delta))
        assert p_high > 0.99
        assert 1.0 / (2 * delta) < p_low < float(consensus_quality(phi, delta))

    def test_monotone_in_d(self):
        """For fixed φ > 1, δ: remaining quality is increasing in d = n1 − n2."""
        phi, delta = 2.5, 6
        d_range = range(-delta + 1, delta)
        vals = [
            float(remaining_quality(max(d, 0), max(-d, 0), phi, delta))
            for d in d_range
        ]
        assert all(a < b for a, b in zip(vals, vals[1:]))

    def test_reflection_symmetry(self):
        """remaining_quality(n1, n2, φ, δ) = 1 − remaining_quality(n2, n1, 1/φ, δ)."""
        for phi in [1.5, 2.0, 3.0]:
            for delta in [3, 5]:
                for d in [-2, -1, 0, 1, 2]:
                    n1, n2 = max(d, 0), max(-d, 0)
                    a = float(remaining_quality(n1, n2, phi, delta))
                    b = float(remaining_quality(n2, n1, 1.0 / phi, delta))
                    assert a + b == pytest.approx(1.0)

    def test_large_phi_no_overflow(self):
        """Far-from-random pool with a tied state still yields a finite Q ≈ 1."""
        assert remaining_quality(0, 0, 1e200, 4) == pytest.approx(1.0)
        # A tied state with extreme φ should still be ≈ 1.
        assert remaining_quality(1, 1, 1e50, 5) == pytest.approx(1.0)

    def test_small_phi_no_overflow(self):
        assert remaining_quality(0, 0, 1e-200, 4) == pytest.approx(0.0, abs=1e-10)

    def test_already_absorbed_raises(self):
        with pytest.raises(ValueError):
            remaining_quality(3, 0, 2.0, 3)  # d = 3 = δ
        with pytest.raises(ValueError):
            remaining_quality(0, 4, 2.0, 3)  # d = -4, |d| > δ

    def test_invalid_counts(self):
        with pytest.raises(ValueError):
            remaining_quality(-1, 0, 2.0, 3)
        with pytest.raises(ValueError):
            remaining_quality(0, -1, 2.0, 3)
        with pytest.raises(ValueError):
            remaining_quality(1.5, 0, 2.0, 3)

    def test_vectorized_over_counts(self):
        n1 = np.array([0, 1, 2])
        n2 = np.array([0, 0, 0])
        result = remaining_quality(n1, n2, 2.0, 4)
        assert result.shape == (3,)
        for i in range(3):
            assert result[i] == pytest.approx(
                float(remaining_quality(int(n1[i]), int(n2[i]), 2.0, 4))
            )


# ── Sanity properties of remaining_expected_votes ──────────────────────────

class TestRemainingExpectedVotes:

    def test_random_voter_parabolic(self):
        """φ = 1: E[remaining] = δ² − d²."""
        for delta in [2, 4, 6]:
            for d in range(-delta + 1, delta):
                n1, n2 = max(d, 0), max(-d, 0)
                assert remaining_expected_votes(n1, n2, 1.0, delta) == pytest.approx(
                    delta ** 2 - d ** 2
                )

    def test_one_step_to_absorption(self):
        """At d = δ − 1, the remaining time has E ≥ 1 and equals 1 exactly when φ = ∞."""
        for phi in [1.5, 2.0, 5.0]:
            e = float(remaining_expected_votes(2, 0, phi, 3))  # d = 2, δ = 3
            assert e >= 1.0

    def test_symmetric_in_phi_at_zero_d(self):
        """At d = 0 the expected remaining votes are symmetric under φ ↔ 1/φ."""
        for delta in [2, 3, 5]:
            for phi in [1.5, 2.0, 3.0]:
                a = float(remaining_expected_votes(0, 0, phi, delta))
                b = float(remaining_expected_votes(0, 0, 1.0 / phi, delta))
                assert a == pytest.approx(b, rel=1e-10)

    def test_decreasing_toward_boundary(self):
        """E[remaining] is maximised at d = 0 and decreases toward ±δ."""
        phi, delta = 2.0, 6
        e_center = float(remaining_expected_votes(0, 0, phi, delta))
        for off in [1, 2, 3, 4, 5]:
            n1, n2 = off, 0
            e_off = float(remaining_expected_votes(n1, n2, phi, delta))
            assert e_off < e_center

    def test_nonnegative(self):
        for phi in [0.5, 1.0, 2.0]:
            for delta in [2, 4]:
                for d in range(-delta + 1, delta):
                    n1, n2 = max(d, 0), max(-d, 0)
                    assert float(remaining_expected_votes(n1, n2, phi, delta)) >= 0


# ── Sanity properties of remaining_var_votes ───────────────────────────────

class TestRemainingVarVotes:

    def test_delta_1_zero(self):
        """δ = 1: only the start state d = 0 is allowed, and 1 vote suffices."""
        for phi in [1.0, 2.0, 5.0]:
            assert remaining_var_votes(0, 0, phi, 1) == pytest.approx(0.0, abs=1e-12)

    def test_one_step_to_absorption_zero_variance(self):
        """At d = δ − 1 with φ = ∞: deterministic single step, Var → 0."""
        v = float(remaining_var_votes(4, 0, 1e200, 5))  # d = 4, δ = 5
        assert v == pytest.approx(0.0, abs=1e-8)

    def test_nonnegative(self):
        for phi in [0.5, 1.5, 3.0]:
            for delta in [2, 3, 5]:
                for d in range(-delta + 1, delta):
                    n1, n2 = max(d, 0), max(-d, 0)
                    v = float(remaining_var_votes(n1, n2, phi, delta))
                    assert v >= -1e-12

    def test_matches_pmf_second_moment(self):
        """Variance from the closed form ≈ variance computed from the pmf."""
        phi, delta = 2.0, 3
        for d in [-1, 0, 1]:
            n1, n2 = max(d, 0), max(-d, 0)
            m_vals = np.arange(delta - abs(d), delta - abs(d) + 200, 2)
            # ensure parity is right for delta - d
            m_vals = np.array([m for m in m_vals if (m - (delta - d)) % 2 == 0])
            pmf_vals = np.array(
                [float(remaining_votes_pmf(int(m), n1, n2, phi, delta)) for m in m_vals]
            )
            mu = float((m_vals * pmf_vals).sum())
            mu2 = float((m_vals ** 2 * pmf_vals).sum())
            var_pmf = mu2 - mu ** 2
            var_closed = float(remaining_var_votes(n1, n2, phi, delta))
            assert var_pmf == pytest.approx(var_closed, rel=1e-3, abs=1e-6)


# ── Sanity properties of remaining_votes_pmf ───────────────────────────────

class TestRemainingVotesPmf:

    def test_min_steps(self):
        """pmf(m) = 0 for m < δ − |d|."""
        # d = 2, δ = 4 ⇒ minimum 2 more votes.
        assert remaining_votes_pmf(1, 2, 0, 3.0, 4) == pytest.approx(0.0)
        assert remaining_votes_pmf(0, 2, 0, 3.0, 4) == pytest.approx(0.0)

    def test_parity(self):
        """pmf(m) = 0 when (m − (δ − d)) is odd."""
        # d = 1, δ = 4 ⇒ δ − d = 3, so pmf at even m must be 0.
        assert remaining_votes_pmf(4, 1, 0, 3.0, 4) == pytest.approx(0.0)
        assert remaining_votes_pmf(6, 1, 0, 3.0, 4) == pytest.approx(0.0)

    def test_sums_to_one(self):
        for phi in [1.5, 3.0]:
            for delta in [3, 4]:
                for d in [-1, 0, 1, 2]:
                    n1, n2 = max(d, 0), max(-d, 0)
                    m_vals = np.arange(0, 400)
                    total = float(
                        np.sum(
                            [
                                remaining_votes_pmf(int(m), n1, n2, phi, delta)
                                for m in m_vals
                            ]
                        )
                    )
                    assert total == pytest.approx(1.0, abs=1e-6)

    def test_expectation_matches_closed_form(self):
        phi, delta = 2.5, 4
        for d in [-1, 0, 1, 2]:
            n1, n2 = max(d, 0), max(-d, 0)
            m_vals = np.arange(0, 300)
            pmf_vals = np.array(
                [float(remaining_votes_pmf(int(m), n1, n2, phi, delta)) for m in m_vals]
            )
            e_pmf = float((m_vals * pmf_vals).sum())
            e_closed = float(remaining_expected_votes(n1, n2, phi, delta))
            assert e_pmf == pytest.approx(e_closed, rel=1e-4)

    def test_one_step_absorbing(self):
        """d = δ − 1, m = 1: pmf = p = φ / (1 + φ); other side has m = 1 ⇒ 0 (parity)."""
        phi, delta = 3.0, 4
        # d = 3 ⇒ δ − d = 1, m = 1 hits +δ with prob p; parity even so m=1 fine.
        p = phi / (1.0 + phi)
        assert remaining_votes_pmf(1, 3, 0, phi, delta) == pytest.approx(p)

    def test_invalid_m_float(self):
        with pytest.raises(ValueError):
            remaining_votes_pmf(2.5, 0, 0, 3.0, 2)


# ── Vectorization / broadcast ──────────────────────────────────────────────

class TestBroadcast:

    def test_broadcast_counts_and_phi(self):
        n1 = np.array([0, 1, 2])
        n2 = np.array([0, 0, 0])
        phis = np.array([2.0, 3.0, 4.0])
        result = remaining_quality(n1, n2, phis, 5)
        assert result.shape == (3,)

    def test_preserves_broadcast_shape(self):
        n1 = np.array([[0], [1]])
        n2 = np.array([0])
        result = remaining_expected_votes(n1, n2, 2.0, 4)
        assert result.shape == (2, 1)
