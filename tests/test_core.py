"""Tests for deltavote.core — Theorems 1–4 (§4)."""

import numpy as np
import pytest

from deltavote.core import (
    consensus_quality,
    expected_votes,
    var_votes,
    votes_pmf,
)


# ── Theorem 1: consensus_quality ────────────────────────────────────────────

class TestConsensusQuality:

    def test_random_voter(self):
        """φ = 1 → Q = 0.5 for any δ."""
        for d in [1, 2, 5, 10]:
            assert consensus_quality(1.0, d) == pytest.approx(0.5)

    def test_delta_1(self):
        """δ = 1 → Q = φ / (1 + φ) = p."""
        for p in [0.6, 0.75, 0.9]:
            phi = p / (1 - p)
            assert consensus_quality(phi, 1) == pytest.approx(p)

    def test_paper_discussion_p075(self):
        """Worked example from §4 discussion: p = 0.75 (φ = 3)."""
        phi = 3.0
        assert consensus_quality(phi, 2) == pytest.approx(0.9, abs=1e-4)
        assert consensus_quality(phi, 3) == pytest.approx(0.964, abs=1e-3)
        assert consensus_quality(phi, 4) == pytest.approx(0.9878, abs=1e-4)

    def test_aml_case_study_junior(self):
        """Table 4 (§8): junior pool φ = 2.95."""
        phi = 2.95
        assert consensus_quality(phi, 2) == pytest.approx(0.897, abs=2e-3)
        assert consensus_quality(phi, 3) == pytest.approx(0.963, abs=2e-3)
        assert consensus_quality(phi, 4) == pytest.approx(0.987, abs=2e-3)
        assert consensus_quality(phi, 5) == pytest.approx(0.996, abs=2e-3)

    def test_aml_case_study_senior(self):
        """Table 4 (§8): senior pool φ = 5.25."""
        phi = 5.25
        assert consensus_quality(phi, 2) == pytest.approx(0.965, abs=2e-3)
        assert consensus_quality(phi, 3) == pytest.approx(0.993, abs=2e-3)
        assert consensus_quality(phi, 4) == pytest.approx(0.999, abs=2e-3)

    def test_monotone_in_delta(self):
        """Q is strictly increasing in δ when φ > 1."""
        phi = 2.0
        vals = [float(consensus_quality(phi, d)) for d in range(1, 10)]
        assert all(a < b for a, b in zip(vals, vals[1:]))

    def test_monotone_in_phi(self):
        """Q is strictly increasing in φ for fixed δ > 1."""
        delta = 3
        phis = np.linspace(1.01, 10.0, 20)
        vals = consensus_quality(phis, delta)
        assert np.all(np.diff(vals) > 0)

    def test_vectorized(self):
        phis = np.array([2.0, 3.0, 4.0])
        result = consensus_quality(phis, 2)
        assert result.shape == (3,)
        for i, phi in enumerate(phis):
            assert result[i] == pytest.approx(float(consensus_quality(phi, 2)))

    def test_large_phi_no_overflow(self):
        """φ^δ overflow should not produce NaN — Q should approach 1."""
        assert consensus_quality(1e200, 2) == pytest.approx(1.0)
        assert consensus_quality(1e50, 10) == pytest.approx(1.0)

    def test_small_phi_no_overflow(self):
        """φ^δ → 0 should not produce NaN — Q should approach 0."""
        assert consensus_quality(1e-200, 2) == pytest.approx(0.0, abs=1e-10)
        assert consensus_quality(1e-50, 10) == pytest.approx(0.0, abs=1e-10)

    def test_invalid_phi(self):
        with pytest.raises(ValueError):
            consensus_quality(0.0, 2)
        with pytest.raises(ValueError):
            consensus_quality(-1.0, 2)
        with pytest.raises(ValueError):
            consensus_quality(np.nan, 2)
        with pytest.raises(ValueError):
            consensus_quality(np.inf, 2)

    def test_invalid_delta(self):
        with pytest.raises(ValueError):
            consensus_quality(2.0, 0)
        with pytest.raises(ValueError):
            consensus_quality(2.0, -1)


# ── Theorem 2: expected_votes ───────────────────────────────────────────────

class TestExpectedVotes:

    def test_random_voter(self):
        """φ = 1 → E[n] = δ²."""
        for d in [1, 2, 3, 5]:
            assert expected_votes(1.0, d) == pytest.approx(d ** 2)

    def test_aml_junior(self):
        """Table 4 (§8): junior pool φ = 2.95."""
        phi = 2.95
        assert expected_votes(phi, 2) == pytest.approx(3.22, abs=0.05)
        assert expected_votes(phi, 3) == pytest.approx(5.62, abs=0.05)
        assert expected_votes(phi, 4) == pytest.approx(7.89, abs=0.05)
        assert expected_votes(phi, 5) == pytest.approx(10.04, abs=0.05)

    def test_aml_senior(self):
        """Table 4 (§8): senior pool φ = 5.25."""
        phi = 5.25
        assert expected_votes(phi, 2) == pytest.approx(2.74, abs=0.05)
        assert expected_votes(phi, 3) == pytest.approx(4.35, abs=0.05)
        assert expected_votes(phi, 4) == pytest.approx(5.87, abs=0.05)

    def test_delta_1(self):
        """δ = 1 → exactly 1 vote needed."""
        for phi in [1.5, 3.0, 10.0]:
            assert expected_votes(phi, 1) == pytest.approx(1.0)

    def test_vectorized(self):
        phis = np.array([2.0, 3.0])
        deltas = np.array([2, 3])
        result = expected_votes(phis, deltas)
        assert result.shape == (2,)

    def test_broadcast_scalar_phi_array_delta(self):
        """Scalar phi with array delta should broadcast correctly."""
        deltas = np.array([1, 2, 3])
        result = expected_votes(2.0, deltas)
        assert result.shape == (3,)
        for i, d in enumerate(deltas):
            assert result[i] == pytest.approx(float(expected_votes(2.0, d)))

    def test_large_phi_no_overflow(self):
        """Large φ should not overflow — E[n] approaches δ."""
        assert np.isfinite(expected_votes(1e200, 2))
        assert expected_votes(1e200, 2) == pytest.approx(2.0, abs=0.01)

    def test_small_phi_no_overflow(self):
        """Small φ should not overflow — E[n] approaches δ by symmetry."""
        assert np.isfinite(expected_votes(1e-200, 2))
        assert expected_votes(1e-200, 2) == pytest.approx(2.0, abs=0.01)

    def test_near_random_not_treated_as_random(self):
        """φ = 1.00001 should use the closed form, not the random-voter limit."""
        result = float(expected_votes(1.00001, 100000))
        # Closed form: 100000·(2/1e-5)·tanh(0.5) ≈ 9.24e9, ~7.6% below δ² = 1e10.
        assert result == pytest.approx(9.24e9, rel=0.02)
        assert result < 1e10  # strictly below the random-voter limit

    def test_preserves_broadcast_shape(self):
        """Singleton broadcast axes should be preserved."""
        phis = np.array([[2.0], [3.0]])
        deltas = np.array([2])
        result = expected_votes(phis, deltas)
        assert result.shape == (2, 1)


# ── Theorem 3: var_votes ────────────────────────────────────────────────────

class TestVarVotes:

    def test_random_voter(self):
        """φ = 1 → Var = 2·δ²·(δ²−1)/3."""
        for d in [2, 3, 4, 5]:
            expected = 2.0 * d ** 2 * (d ** 2 - 1) / 3.0
            assert var_votes(1.0, d) == pytest.approx(expected)

    def test_delta_1(self):
        """δ = 1 → 0 variance (always exactly 1 vote)."""
        for phi in [1.0, 2.0, 5.0]:
            assert var_votes(phi, 1) == pytest.approx(0.0, abs=1e-12)

    def test_paper_table_delta2(self):
        """Table 1 (§4): Var for δ = 2 is 8φ·((φ+1)/(φ²+1))²."""
        for phi in [1.5, 2.0, 3.0, 5.0]:
            expected = 8.0 * phi * ((phi + 1) / (phi ** 2 + 1)) ** 2
            assert var_votes(phi, 2) == pytest.approx(expected, rel=1e-10)

    def test_paper_table_delta3(self):
        """Table 1 (§4): Var for δ = 3."""
        for phi in [1.5, 2.0, 3.0, 5.0]:
            expected = (12.0 * phi * ((phi + 1) / (phi ** 3 + 1)) ** 2
                        * (phi ** 2 + 2 * phi + 1))
            assert var_votes(phi, 3) == pytest.approx(expected, rel=1e-10)

    def test_paper_table_delta4(self):
        """Table 1 (§4): Var for δ = 4."""
        for phi in [1.5, 2.0, 3.0]:
            expected = (16.0 * phi * ((phi + 1) / (phi ** 4 + 1)) ** 2
                        * (phi ** 4 + 2 * phi ** 3 + 4 * phi ** 2
                           + 2 * phi + 1))
            assert var_votes(phi, 4) == pytest.approx(expected, rel=1e-10)

    def test_paper_table_delta5(self):
        """Table 1 (§4): Var for δ = 5."""
        for phi in [2.0, 3.0]:
            expected = (20.0 * phi * ((phi + 1) / (phi ** 5 + 1)) ** 2
                        * (phi ** 6 + 2 * phi ** 5 + 4 * phi ** 4
                           + 6 * phi ** 3 + 4 * phi ** 2 + 2 * phi + 1))
            assert var_votes(phi, 5) == pytest.approx(expected, rel=1e-10)

    def test_paper_table_delta7(self):
        """Table 1 (§4): Var for δ = 7."""
        phi = 2.0
        poly = (phi ** 10 + 2 * phi ** 9 + 4 * phi ** 8 + 6 * phi ** 7
                + 9 * phi ** 6 + 12 * phi ** 5 + 9 * phi ** 4
                + 6 * phi ** 3 + 4 * phi ** 2 + 2 * phi + 1)
        expected = 28.0 * phi * ((phi + 1) / (phi ** 7 + 1)) ** 2 * poly
        assert var_votes(phi, 7) == pytest.approx(expected, rel=1e-10)

    def test_nonnegative(self):
        for phi in [0.5, 1.0, 2.0, 5.0]:
            for d in [1, 2, 3, 5]:
                assert float(var_votes(phi, d)) >= -1e-15

    def test_broadcast_scalar_phi_array_delta(self):
        """Scalar phi with array delta should broadcast correctly."""
        deltas = np.array([1, 2, 3])
        result = var_votes(2.0, deltas)
        assert result.shape == (3,)
        for i, d in enumerate(deltas):
            assert result[i] == pytest.approx(float(var_votes(2.0, d)))

    def test_large_phi_no_overflow(self):
        """Large φ should not overflow — Var approaches 0."""
        result = float(var_votes(1e200, 3))
        assert np.isfinite(result)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_small_phi_no_overflow(self):
        """Small φ should not overflow — Var approaches 0."""
        result = float(var_votes(1e-200, 3))
        assert np.isfinite(result)
        assert result == pytest.approx(0.0, abs=1e-10)

    def test_preserves_broadcast_shape(self):
        """Singleton broadcast axes should be preserved."""
        phis = np.array([[2.0], [3.0]])
        deltas = np.array([2])
        result = var_votes(phis, deltas)
        assert result.shape == (2, 1)


# ── Theorem 4: votes_pmf ───────────────────────────────────────────────────

class TestVotesPmf:

    def test_zero_below_delta(self):
        """pmf(m) = 0 for m < δ."""
        assert votes_pmf(1, 3.0, 3) == pytest.approx(0.0)
        assert votes_pmf(2, 3.0, 3) == pytest.approx(0.0)

    def test_zero_wrong_parity(self):
        """pmf(m) = 0 when m ≢ δ (mod 2)."""
        assert votes_pmf(4, 3.0, 3) == pytest.approx(0.0)
        assert votes_pmf(3, 3.0, 2) == pytest.approx(0.0)

    def test_sums_to_one(self):
        """Sum of pmf over enough terms ≈ 1."""
        for phi in [1.5, 3.0, 5.0]:
            for delta in [2, 3, 4]:
                m_vals = np.arange(delta, delta + 200, 2)
                total = np.sum(votes_pmf(m_vals, phi, delta))
                assert total == pytest.approx(1.0, abs=1e-6)

    def test_delta_1(self):
        """δ = 1 → exactly 1 vote, so pmf(1) = 1, pmf(k > 1) = 0."""
        for phi in [1.5, 3.0]:
            assert votes_pmf(1, phi, 1) == pytest.approx(1.0)
            assert votes_pmf(3, phi, 1) == pytest.approx(0.0)

    def test_expectation_matches_theorem2(self):
        """E[m] from pmf should match expected_votes."""
        for phi in [2.0, 3.0]:
            for delta in [2, 3, 4]:
                m_vals = np.arange(delta, delta + 300, 2)
                pmf_vals = votes_pmf(m_vals, phi, delta)
                e_from_pmf = np.sum(m_vals * pmf_vals)
                e_formula = float(expected_votes(phi, delta))
                assert e_from_pmf == pytest.approx(e_formula, rel=1e-4)

    def test_variance_matches_theorem3(self):
        """Var[m] from pmf should match var_votes."""
        for phi in [2.0, 3.0]:
            for delta in [2, 3]:
                m_vals = np.arange(delta, delta + 400, 2)
                pmf_vals = votes_pmf(m_vals, phi, delta)
                e1 = np.sum(m_vals * pmf_vals)
                e2 = np.sum(m_vals ** 2 * pmf_vals)
                var_from_pmf = e2 - e1 ** 2
                var_formula = float(var_votes(phi, delta))
                assert var_from_pmf == pytest.approx(var_formula, rel=1e-3)

    def test_invalid_m_float(self):
        """Non-integer m should raise ValueError."""
        with pytest.raises(ValueError):
            votes_pmf(2.9, 3.0, 2)

    def test_preserves_shape(self):
        """Array m should preserve its shape in the output."""
        m_vals = np.array([2, 4, 6])
        result = votes_pmf(m_vals, 3.0, 2)
        assert result.shape == (3,)


# ── Cross-checks ────────────────────────────────────────────────────────────

class TestSymmetry:

    def test_phi_reciprocal_quality(self):
        """Q(φ, δ) + Q(1/φ, δ) = 1."""
        for phi in [1.5, 2.0, 3.0, 5.0]:
            for d in [1, 2, 3, 5]:
                q1 = float(consensus_quality(phi, d))
                q2 = float(consensus_quality(1.0 / phi, d))
                assert q1 + q2 == pytest.approx(1.0)

    def test_expected_votes_symmetric_in_phi(self):
        """E[n](φ, δ) = E[n](1/φ, δ)."""
        for phi in [1.5, 2.0, 3.0]:
            for d in [2, 3, 5]:
                e1 = float(expected_votes(phi, d))
                e2 = float(expected_votes(1.0 / phi, d))
                assert e1 == pytest.approx(e2)

    def test_var_votes_symmetric_in_phi(self):
        """Var[n](φ, δ) = Var[n](1/φ, δ)."""
        for phi in [1.5, 2.0, 3.0]:
            for d in [2, 3, 5]:
                v1 = float(var_votes(phi, d))
                v2 = float(var_votes(1.0 / phi, d))
                assert v1 == pytest.approx(v2, rel=1e-8)
