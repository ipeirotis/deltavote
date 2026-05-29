"""Tests for deltavote.equivalence — Theorems 6.1–6.2 (§6)."""

import numpy as np
import pytest

from deltavote.core import consensus_quality
from deltavote.equivalence import (
    equivalent_delta,
    equivalent_payment,
    quality_matched_pools,
)


# ── Theorem 6.1: equivalent_delta ───────────────────────────────────────────

class TestEquivalentDelta:

    def test_same_pool_is_identity(self):
        """Matching a pool to itself returns the same threshold."""
        for phi in [1.5, 3.0, 7.0]:
            for d in [1, 3, 5]:
                assert equivalent_delta(phi, d, phi) == pytest.approx(d)

    def test_matches_quality(self):
        """δ₂ makes Q(φ₂, δ₂) equal Q(φ₁, δ₁).

        Verified through the log-odds identity δ₁·ln φ₁ = δ₂·ln φ₂, which
        is exactly the condition for Q = σ(δ·ln φ) to agree (δ₂ is in
        general non-integer, so we cannot call consensus_quality on it).
        """
        phi1, delta1, phi2 = 3.0, 4.0, 2.0
        d2 = equivalent_delta(phi1, delta1, phi2)
        assert delta1 * np.log(phi1) == pytest.approx(float(d2) * np.log(phi2))

    def test_formula(self):
        """δ₂ = δ₁ · ln(φ₁) / ln(φ₂)."""
        phi1, delta1, phi2 = 5.0, 3.0, 2.0
        expected = delta1 * np.log(phi1) / np.log(phi2)
        assert equivalent_delta(phi1, delta1, phi2) == pytest.approx(expected)

    def test_higher_accuracy_needs_fewer_votes(self):
        """A more accurate pool 2 (φ₂ > φ₁) matches with δ₂ < δ₁."""
        d2 = equivalent_delta(2.0, 6, 8.0)
        assert d2 < 6

    def test_lower_accuracy_needs_more_votes(self):
        """A less accurate pool 2 (φ₂ < φ₁) matches with δ₂ > δ₁."""
        d2 = equivalent_delta(8.0, 3, 2.0)
        assert d2 > 3

    def test_below_chance_same_side(self):
        """Two below-chance pools (both φ < 1) are a valid same-side pair."""
        d2 = equivalent_delta(0.5, 4, 0.25)
        assert float(d2) == pytest.approx(4 * np.log(0.5) / np.log(0.25))
        assert d2 > 0

    def test_opposite_side_raises(self):
        """φ₁ > 1 and φ₂ < 1 have no positive matched threshold."""
        with pytest.raises(ValueError, match="same side of chance"):
            equivalent_delta(3.0, 2, 0.5)

    def test_random_pool_raises(self):
        with pytest.raises(ValueError, match="random pool"):
            equivalent_delta(1.0, 2, 3.0)
        with pytest.raises(ValueError, match="random pool"):
            equivalent_delta(3.0, 2, 1.0)

    def test_invalid_delta(self):
        with pytest.raises(ValueError):
            equivalent_delta(3.0, 0, 2.0)
        with pytest.raises(ValueError):
            equivalent_delta(3.0, 2.5, 2.0)

    def test_vectorized_over_phi2(self):
        phi2 = np.array([2.0, 4.0, 8.0])
        out = equivalent_delta(3.0, 5, phi2)
        expected = 5 * np.log(3.0) / np.log(phi2)
        np.testing.assert_allclose(out, expected)


# ── Theorem 6.2: equivalent_payment ─────────────────────────────────────────

class TestEquivalentPayment:

    def test_formula(self):
        """pay(φ) ∝ ln(φ)·(φ−1)/(φ+1)."""
        phi = 3.0
        expected = np.log(phi) * (phi - 1.0) / (phi + 1.0)
        assert equivalent_payment(phi) == pytest.approx(expected)

    def test_random_pool_pays_zero(self):
        """A random pool (φ = 1) carries no information → pay = 0."""
        assert equivalent_payment(1.0) == pytest.approx(0.0)

    def test_symmetric_under_reciprocal(self):
        """pay(φ) = pay(1/φ): a pool and its label-inverted mirror match."""
        for phi in [1.5, 3.0, 9.0]:
            assert equivalent_payment(phi) == pytest.approx(
                equivalent_payment(1.0 / phi)
            )

    def test_non_negative(self):
        phis = np.linspace(0.05, 20.0, 50)
        assert np.all(equivalent_payment(phis) >= 0.0)

    def test_increasing_above_chance(self):
        """More accurate pools (larger φ > 1) warrant higher pay."""
        phis = np.array([1.5, 2.0, 4.0, 8.0])
        pays = equivalent_payment(phis)
        assert np.all(np.diff(pays) > 0)

    def test_ratio_matches_theorem6(self):
        """pay(φ₁)/pay(φ₂) equals the explicit Theorem 6.2 ratio."""
        phi1, phi2 = 5.0, 2.0
        ratio = equivalent_payment(phi1) / equivalent_payment(phi2)
        explicit = (
            np.log(phi1) / np.log(phi2)
            * (phi2 + 1.0) / (phi1 + 1.0)
            * (phi1 - 1.0) / (phi2 - 1.0)
        )
        assert ratio == pytest.approx(explicit)

    def test_invalid_phi(self):
        with pytest.raises(ValueError):
            equivalent_payment(0.0)
        with pytest.raises(ValueError):
            equivalent_payment(-1.0)


# ── quality_matched_pools ───────────────────────────────────────────────────

class TestQualityMatchedPools:

    def test_matches_equivalent_delta(self):
        phi_list = [2.0, 4.0, 6.0]
        out = quality_matched_pools(3.0, 5, phi_list)
        expected = [float(equivalent_delta(3.0, 5, p)) for p in phi_list]
        np.testing.assert_allclose(out, expected)

    def test_reference_pool_in_list_is_identity(self):
        """The reference pool itself maps back to delta1."""
        out = quality_matched_pools(3.0, 4, [3.0, 9.0])
        assert out[0] == pytest.approx(4.0)

    def test_returns_array(self):
        out = quality_matched_pools(3.0, 4, [2.0, 5.0])
        assert isinstance(out, np.ndarray)
        assert out.shape == (2,)

    def test_opposite_side_in_list_raises(self):
        with pytest.raises(ValueError, match="same side of chance"):
            quality_matched_pools(3.0, 4, [2.0, 0.5])


# ── Cross-checks against core ───────────────────────────────────────────────

class TestConsistency:

    def test_rounded_up_delta_meets_or_beats_target(self):
        """Above chance: ceil(δ₂) delivers quality ≥ pool 1's target."""
        phi1, delta1, phi2 = 4.0, 3, 2.0
        target = float(consensus_quality(phi1, delta1))
        d2 = int(np.ceil(float(equivalent_delta(phi1, delta1, phi2))))
        assert float(consensus_quality(phi2, d2)) >= target

    def test_below_chance_rounding_direction_reverses(self):
        """Below chance: floor(δ₂) — not ceil — is the conservative round.

        Q decreases in δ when φ < 1, so rounding up undershoots the target
        while rounding down preserves quality ≥ pool 1's target.
        """
        phi1, delta1, phi2 = 0.5, 3, 0.4
        target = float(consensus_quality(phi1, delta1))
        d2 = float(equivalent_delta(phi1, delta1, phi2))
        assert float(consensus_quality(phi2, int(np.floor(d2)))) >= target
        assert float(consensus_quality(phi2, int(np.ceil(d2)))) < target
