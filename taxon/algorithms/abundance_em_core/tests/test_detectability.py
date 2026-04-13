"""Unit tests for peptide detectability weights.

Run from the repository root with::

    python -m pytest taxon/algorithms/abundance_em_core/tests/test_detectability.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from taxon.algorithms.abundance_em_core.detectability import (
    DetectabilityWeights,
    build_weighted_emission_matrix,
)
from taxon.algorithms.abundance_em_core.model import AbundanceEM
from taxon.algorithms.abundance_em_core.synthetic import (
    evaluate_recovery,
    generate_synthetic_community,
)


# ============================================================ build_weighted_emission_matrix


class TestBuildWeightedEmissionMatrix:
    """Tests for the weighted emission matrix builder."""

    def test_uniform_weights_match_original(self):
        """Uniform weights (all ones) should reproduce A/n_t."""
        A = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float64)
        d = np.ones(3)
        M = build_weighted_emission_matrix(A, d)
        # Column 0: n_t=2, so M[:,0] = [0.5, 0.5, 0]
        # Column 1: n_t=2, so M[:,1] = [0, 0.5, 0.5]
        expected_col0 = np.array([0.5, 0.5, 0.0])
        expected_col1 = np.array([0.0, 0.5, 0.5])
        np.testing.assert_allclose(M[:, 0], expected_col0)
        np.testing.assert_allclose(M[:, 1], expected_col1)

    def test_none_weights_match_uniform(self):
        """None weights should be identical to all-ones."""
        A = np.array([[1, 0], [1, 1], [0, 1]], dtype=np.float64)
        M_none = build_weighted_emission_matrix(A, None)
        M_ones = build_weighted_emission_matrix(A, np.ones(3))
        np.testing.assert_allclose(M_none, M_ones)

    def test_nonuniform_weights(self):
        """Non-uniform weights should change the emission distribution."""
        A = np.array([[1, 0], [1, 0]], dtype=np.float64)
        # Peptide 0 has weight 3, peptide 1 has weight 1.
        d = np.array([3.0, 1.0])
        M = build_weighted_emission_matrix(A, d)
        # Column 0: weighted sum = 3 + 1 = 4
        np.testing.assert_allclose(M[0, 0], 3.0 / 4.0)
        np.testing.assert_allclose(M[1, 0], 1.0 / 4.0)

    def test_columns_sum_to_one(self):
        """Each column of M should sum to 1 (for non-empty columns)."""
        rng = np.random.default_rng(42)
        A = (rng.random((20, 5)) > 0.6).astype(np.float64)
        d = rng.exponential(1.0, size=20)
        M = build_weighted_emission_matrix(A, d)
        col_sums = M.sum(axis=0)
        for t in range(5):
            if A[:, t].sum() > 0:
                assert abs(col_sums[t] - 1.0) < 1e-12

    def test_empty_column_stays_zero(self):
        """A taxon with no peptides should have an all-zero column."""
        A = np.array([[1, 0], [0, 0]], dtype=np.float64)
        d = np.array([1.0, 1.0])
        M = build_weighted_emission_matrix(A, d)
        assert M[:, 1].sum() == 0.0

    def test_shape_mismatch_raises(self):
        A = np.array([[1, 0], [0, 1]], dtype=np.float64)
        d = np.array([1.0, 2.0, 3.0])  # wrong size
        with pytest.raises(ValueError, match="shape"):
            build_weighted_emission_matrix(A, d)

    def test_negative_weights_raise(self):
        A = np.array([[1, 0], [0, 1]], dtype=np.float64)
        d = np.array([1.0, -1.0])
        with pytest.raises(ValueError, match="non-negative"):
            build_weighted_emission_matrix(A, d)


# ============================================================ DetectabilityWeights class


class TestDetectabilityWeights:
    """Tests for the DetectabilityWeights container."""

    def test_get_weight_vector_known_peptides(self):
        dw = DetectabilityWeights(
            weights={"AAA": 2.0, "BBB": 0.5},
            default_weight=1.0,
        )
        vec = dw.get_weight_vector(["AAA", "CCC", "BBB"])
        np.testing.assert_allclose(vec, [2.0, 1.0, 0.5])

    def test_get_weight_vector_all_default(self):
        dw = DetectabilityWeights(weights={}, default_weight=0.7)
        vec = dw.get_weight_vector(["X", "Y", "Z"])
        np.testing.assert_allclose(vec, [0.7, 0.7, 0.7])

    def test_uniform_factory(self):
        dw = DetectabilityWeights.uniform()
        assert dw.source == "uniform"
        vec = dw.get_weight_vector(["A", "B"])
        np.testing.assert_allclose(vec, [1.0, 1.0])

    def test_negative_default_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            DetectabilityWeights(weights={}, default_weight=-1.0)


# ============================================================ from_pure_cultures


class TestFromPureCultures:
    """Tests for pure culture weight estimation."""

    def test_single_culture(self):
        counts = {"culture_A": {"PEP1": 100, "PEP2": 50, "PEP3": 10}}
        dw = DetectabilityWeights.from_pure_cultures(
            counts, pseudocount=0.0, min_total_psms=10
        )
        # With pseudocount=0: d_p = count / total = count / 160
        assert dw.source == "pure_culture"
        assert abs(dw.weights["PEP1"] - 100 / 160) < 1e-10
        assert abs(dw.weights["PEP2"] - 50 / 160) < 1e-10
        assert abs(dw.weights["PEP3"] - 10 / 160) < 1e-10

    def test_multiple_cultures_averaged(self):
        """Peptide seen in two cultures should have averaged weight."""
        counts = {
            "A": {"PEP1": 100, "PEP2": 0},
            "B": {"PEP1": 50, "PEP3": 50},
        }
        dw = DetectabilityWeights.from_pure_cultures(
            counts, pseudocount=0.0, min_total_psms=10
        )
        # PEP1: culture A gives 100/100=1.0, culture B gives 50/100=0.5
        # average = 0.75
        assert abs(dw.weights["PEP1"] - 0.75) < 1e-10
        # PEP2 only in culture A: 0/100 = 0.0
        assert abs(dw.weights["PEP2"] - 0.0) < 1e-10

    def test_min_total_psms_filter(self):
        counts = {
            "good": {"PEP1": 100, "PEP2": 50},
            "bad": {"PEP1": 5},  # only 5 PSMs
        }
        dw = DetectabilityWeights.from_pure_cultures(
            counts, pseudocount=0.0, min_total_psms=50
        )
        # "bad" culture should be skipped; only "good" contributes
        assert dw.n_peptides_estimated == 2

    def test_all_cultures_below_threshold_raises(self):
        counts = {"tiny": {"PEP1": 5}}
        with pytest.raises(ValueError, match="No pure cultures"):
            DetectabilityWeights.from_pure_cultures(
                counts, min_total_psms=100
            )

    def test_default_weight_is_median(self):
        """Default weight for unseen peptides should be the median."""
        counts = {"A": {"PEP1": 100, "PEP2": 50, "PEP3": 10}}
        dw = DetectabilityWeights.from_pure_cultures(
            counts, pseudocount=0.0, min_total_psms=10
        )
        # Weights: 100/160, 50/160, 10/160 = 0.625, 0.3125, 0.0625
        expected_median = 50 / 160  # 0.3125
        assert abs(dw.default_weight - expected_median) < 1e-10

    def test_pseudocount_smoothing(self):
        """Pseudocount should prevent zero weights."""
        counts = {"A": {"PEP1": 100, "PEP2": 0}}
        dw = DetectabilityWeights.from_pure_cultures(
            counts, pseudocount=1.0, min_total_psms=10
        )
        # PEP2 with pseudocount: (0 + 1) / (100 + 2*1) = 1/102
        assert dw.weights["PEP2"] > 0


# ============================================================ EM with detectability weights


class TestEMWithDetectability:
    """Tests for the EM algorithm with detectability weights."""

    def test_uniform_weights_same_as_no_weights(self):
        """Passing all-ones weights should give identical results to None."""
        data = generate_synthetic_community(
            n_taxa=4,
            n_peptides_per_taxon=100,
            shared_fraction=0.10,
            total_psms=8000,
            seed=42,
        )
        m1 = AbundanceEM(alpha=0.5, max_iter=300, tol=1e-8, seed=0)
        m1.fit(data["A"], data["y"], detectability_weights=None)

        m2 = AbundanceEM(alpha=0.5, max_iter=300, tol=1e-8, seed=0)
        P = data["A"].shape[0]
        m2.fit(data["A"], data["y"], detectability_weights=np.ones(P))

        np.testing.assert_allclose(m1.pi_, m2.pi_, atol=1e-10)

    def test_known_weights_improve_recovery(self):
        """When data is generated with non-uniform detectability, passing
        the true weights to the EM should improve recovery vs. ignoring them.

        We test across multiple seeds to ensure robust improvement, and use
        a high detectability spread (shape=2.0) where the uniform model is
        clearly misspecified.
        """
        n_better = 0
        n_trials = 5
        for trial_seed in [77, 123, 456, 789, 1024]:
            data = generate_synthetic_community(
                n_taxa=5,
                n_peptides_per_taxon=150,
                shared_fraction=0.15,
                total_psms=12000,
                seed=trial_seed,
                detectability="lognormal",
                detectability_shape=2.0,  # very high variability
            )

            m_uniform = AbundanceEM(alpha=0.5, max_iter=500, tol=1e-8, seed=0)
            m_uniform.fit(data["A"], data["y"])
            err_uniform = evaluate_recovery(data["true_pi"], m_uniform.pi_)

            m_weighted = AbundanceEM(alpha=0.5, max_iter=500, tol=1e-8, seed=0)
            m_weighted.fit(
                data["A"],
                data["y"],
                detectability_weights=data["detectability_weights"],
            )
            err_weighted = evaluate_recovery(data["true_pi"], m_weighted.pi_)

            if err_weighted["l1_error"] < err_uniform["l1_error"]:
                n_better += 1

        # Weighted model should be better in majority of trials.
        assert n_better >= 3, (
            f"Weighted model was better in only {n_better}/{n_trials} trials"
        )

    def test_detectability_gamma(self):
        """Gamma detectability distribution should also work."""
        data = generate_synthetic_community(
            n_taxa=4,
            n_peptides_per_taxon=150,
            shared_fraction=0.10,
            total_psms=10000,
            seed=99,
            detectability="gamma",
            detectability_shape=0.3,
        )

        m = AbundanceEM(alpha=0.5, max_iter=500, tol=1e-8, seed=0)
        m.fit(
            data["A"],
            data["y"],
            detectability_weights=data["detectability_weights"],
        )
        err = evaluate_recovery(data["true_pi"], m.pi_)
        # Should achieve reasonable recovery with true weights.
        assert err["l1_error"] < 0.10, f"L1 error too large: {err['l1_error']}"

    def test_convergence_with_weights(self):
        """EM should converge with non-uniform weights."""
        data = generate_synthetic_community(
            n_taxa=5,
            n_peptides_per_taxon=100,
            shared_fraction=0.15,
            total_psms=8000,
            seed=42,
            detectability="lognormal",
            detectability_shape=0.5,
        )

        m = AbundanceEM(alpha=0.5, max_iter=500, tol=1e-7, seed=0)
        m.fit(
            data["A"],
            data["y"],
            detectability_weights=data["detectability_weights"],
        )
        assert m.converged_

    def test_monotonicity_with_weights(self):
        """Log-posterior should be non-decreasing with detectability weights."""
        data = generate_synthetic_community(
            n_taxa=6,
            n_peptides_per_taxon=120,
            shared_fraction=0.20,
            total_psms=10000,
            seed=13,
            detectability="lognormal",
            detectability_shape=0.8,
        )

        m = AbundanceEM(alpha=0.7, max_iter=300, tol=1e-9, seed=0)
        m.fit(
            data["A"],
            data["y"],
            detectability_weights=data["detectability_weights"],
        )

        history = m.log_posterior_history_
        assert len(history) >= 2
        diffs = np.diff(history)
        assert (diffs >= -1e-7 * (np.abs(history[:-1]) + 1.0)).all(), (
            f"non-monotonic step: {diffs.min()}"
        )

    def test_weight_shape_mismatch_raises(self):
        """Wrong-length weight vector should raise ValueError."""
        data = generate_synthetic_community(
            n_taxa=3, n_peptides_per_taxon=50, total_psms=1000, seed=0
        )
        m = AbundanceEM(alpha=0.5)
        with pytest.raises(ValueError, match="shape"):
            m.fit(data["A"], data["y"], detectability_weights=np.ones(5))

    def test_negative_weight_raises(self):
        """Negative weights should raise ValueError."""
        data = generate_synthetic_community(
            n_taxa=3, n_peptides_per_taxon=50, total_psms=1000, seed=0
        )
        m = AbundanceEM(alpha=0.5)
        P = data["A"].shape[0]
        d = np.ones(P)
        d[0] = -1.0
        with pytest.raises(ValueError, match="non-negative"):
            m.fit(data["A"], data["y"], detectability_weights=d)

    def test_sparse_community_with_weights(self):
        """Sparse prior + detectability weights should still detect present taxa."""
        n_taxa = 15
        true_pi = np.zeros(n_taxa)
        present = [3, 8, 12]
        true_pi[present] = [0.5, 0.3, 0.2]

        data = generate_synthetic_community(
            n_taxa=n_taxa,
            n_peptides_per_taxon=80,
            shared_fraction=0.10,
            true_pi=true_pi,
            total_psms=15000,
            seed=42,
            detectability="lognormal",
            detectability_shape=0.7,
        )

        m = AbundanceEM(
            alpha=0.5,
            max_iter=1000,
            tol=1e-8,
            n_restarts=3,
            min_abundance=1e-3,
            seed=0,
        )
        m.fit(
            data["A"],
            data["y"],
            detectability_weights=data["detectability_weights"],
        )

        detected = set(np.where(m.pi_ > 1e-3)[0].tolist())
        assert detected == set(present), (
            f"detected={detected}, expected={set(present)}"
        )


# ============================================================ synthetic community extensions


class TestSyntheticDetectability:
    """Tests for the extended synthetic data generator."""

    def test_uniform_detectability_default(self):
        """No detectability arg should give all-ones weights."""
        data = generate_synthetic_community(n_taxa=3, seed=0)
        np.testing.assert_allclose(
            data["detectability_weights"], np.ones(data["A"].shape[0])
        )

    def test_lognormal_detectability(self):
        """Lognormal weights should be positive and variable."""
        data = generate_synthetic_community(
            n_taxa=3,
            n_peptides_per_taxon=100,
            seed=0,
            detectability="lognormal",
            detectability_shape=1.0,
        )
        d = data["detectability_weights"]
        assert (d > 0).all()
        assert d.std() > 0.1  # should have meaningful variation

    def test_gamma_detectability(self):
        """Gamma weights should be non-negative."""
        data = generate_synthetic_community(
            n_taxa=3,
            n_peptides_per_taxon=100,
            seed=0,
            detectability="gamma",
            detectability_shape=0.5,
        )
        d = data["detectability_weights"]
        assert (d >= 0).all()

    def test_invalid_detectability_raises(self):
        with pytest.raises(ValueError, match="detectability"):
            generate_synthetic_community(
                n_taxa=3, seed=0, detectability="invalid"
            )
