"""Tests for deltavote.design — practitioner helpers (§4)."""

import numpy as np
import pytest

from deltavote.core import consensus_quality, expected_votes
from deltavote.design import (
    cost_for_target_quality,
    expected_cost,
    recommend_delta,
)


# ── recommend_delta ─────────────────────────────────────────────────────────

class TestRecommendDelta:

    def test_smallest_delta_meets_target(self):
        """Returned δ achieves the target; δ−1 does not (when δ > 1)."""
        phi = 3.0
        for target in [0.85, 0.92, 0.97, 0.99]:
            d = int(recommend_delta(phi, target))
            assert float(consensus_quality(phi, d)) >= target
            if d > 1:
                assert float(consensus_quality(phi, d - 1)) < target

    def test_paper_example_p075(self):
        """§4: φ = 3 gives Q(2)=0.9, Q(3)=0.964, Q(4)=0.988."""
        assert recommend_delta(3.0, 0.89) == 2
        assert recommend_delta(3.0, 0.95) == 3
        assert recommend_delta(3.0, 0.98) == 4

    def test_target_below_single_vote_accuracy(self):
        """If Q(φ,1)=p already clears the target, δ = 1."""
        # φ = 3 → p = 0.75
        assert recommend_delta(3.0, 0.7) == 1
        assert recommend_delta(3.0, 0.5) == 1

    def test_exact_boundary_no_over_rounding(self):
        """Q(2,2)=0.8 exactly: target 0.8 must give δ=2, not 3.

        Regression for the float boundary logit(0.8)/ln(2)=2.0000000000004,
        which naively rounds up to 3 (one extra, costlier vote).
        """
        assert recommend_delta(2.0, 0.8) == 2
        # A few more exact Q(phi, delta) = phi^d/(1+phi^d) boundaries.
        assert recommend_delta(3.0, 0.9) == 2          # Q(3,2) = 0.9
        assert recommend_delta(2.0, 8.0 / 9.0) == 3    # Q(2,3) = 8/9

    def test_high_target_needs_large_delta(self):
        d = int(recommend_delta(1.5, 0.999))
        assert float(consensus_quality(1.5, d)) >= 0.999
        assert float(consensus_quality(1.5, d - 1)) < 0.999

    def test_random_pool_raises(self):
        with pytest.raises(ValueError, match="phi > 1"):
            recommend_delta(1.0, 0.9)

    def test_below_chance_raises(self):
        with pytest.raises(ValueError, match="phi > 1"):
            recommend_delta(0.5, 0.9)

    def test_invalid_target(self):
        for bad in [0.0, 1.0, -0.1, 1.5]:
            with pytest.raises(ValueError, match="target_quality"):
                recommend_delta(3.0, bad)

    def test_vectorized(self):
        phi = np.array([2.0, 3.0, 5.0])
        out = recommend_delta(phi, 0.95)
        assert out.shape == (3,)
        for p, d in zip(phi, out):
            assert float(consensus_quality(p, int(d))) >= 0.95


# ── expected_cost ───────────────────────────────────────────────────────────

class TestExpectedCost:

    def test_default_cost_equals_expected_votes(self):
        phi, delta = 3.0, 4
        assert expected_cost(phi, delta) == pytest.approx(
            float(expected_votes(phi, delta))
        )

    def test_scales_with_cost_per_vote(self):
        phi, delta = 2.5, 3
        base = float(expected_votes(phi, delta))
        assert expected_cost(phi, delta, 0.05) == pytest.approx(0.05 * base)

    def test_negative_cost_raises(self):
        with pytest.raises(ValueError, match="cost_per_vote"):
            expected_cost(3.0, 2, -1.0)

    def test_non_finite_cost_raises(self):
        """nan/inf slip past a bare < 0 check; reject them explicitly."""
        for bad in [np.nan, np.inf, -np.inf]:
            with pytest.raises(ValueError, match="cost_per_vote"):
                expected_cost(3.0, 2, bad)

    def test_vectorized(self):
        phi = np.array([2.0, 3.0])
        out = expected_cost(phi, 3, 2.0)
        np.testing.assert_allclose(out, 2.0 * expected_votes(phi, 3))


# ── cost_for_target_quality ─────────────────────────────────────────────────

class TestCostForTargetQuality:

    def test_composition(self):
        """Equals expected_cost at the recommended δ."""
        phi, target, cpv = 3.0, 0.97, 0.10
        d = recommend_delta(phi, target)
        assert cost_for_target_quality(phi, target, cpv) == pytest.approx(
            float(expected_cost(phi, d, cpv))
        )

    def test_higher_target_costs_more(self):
        phi = 2.0
        c_low = float(cost_for_target_quality(phi, 0.9))
        c_high = float(cost_for_target_quality(phi, 0.99))
        assert c_high > c_low

    def test_more_accurate_pool_costs_less(self):
        """For the same target, a more accurate pool is cheaper."""
        target = 0.97
        c_weak = float(cost_for_target_quality(2.0, target))
        c_strong = float(cost_for_target_quality(8.0, target))
        assert c_strong < c_weak
