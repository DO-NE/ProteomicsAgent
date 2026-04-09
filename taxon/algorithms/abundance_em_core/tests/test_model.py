"""Unit tests for the AbundanceEM core algorithm.

Run from the repository root with::

    python -m pytest taxon/algorithms/abundance_em_core/tests/ -v
"""

from __future__ import annotations

import numpy as np
import pytest

from taxon.algorithms.abundance_em_core.identifiability import identifiability_report
from taxon.algorithms.abundance_em_core.model import AbundanceEM
from taxon.algorithms.abundance_em_core.synthetic import (
    evaluate_recovery,
    generate_synthetic_community,
)


def _build_disjoint_matrix(true_pi: np.ndarray, peptides_per_taxon: int = 80):
    """Build a perfectly disjoint mapping matrix and a noiseless count vector."""
    T = true_pi.shape[0]
    P = T * peptides_per_taxon
    A = np.zeros((P, T), dtype=np.int8)
    for t in range(T):
        A[t * peptides_per_taxon : (t + 1) * peptides_per_taxon, t] = 1
    n_t = A.sum(axis=0)
    M = A.astype(float) / n_t
    phi = M @ true_pi
    rng = np.random.default_rng(0)
    y = rng.multinomial(50000, phi)
    return A, y


# ----------------------------------------------------------------------- tests


def test_perfect_recovery_no_sharing():
    """Disjoint peptide sets -> recovery should be near-exact."""
    true_pi = np.array([0.4, 0.25, 0.2, 0.1, 0.05])
    A, y = _build_disjoint_matrix(true_pi, peptides_per_taxon=80)

    model = AbundanceEM(alpha=1.0, max_iter=500, tol=1e-9, seed=0)
    model.fit(A, y)

    err = float(np.abs(model.pi_ - true_pi).sum())
    assert err < 0.01, f"L1 error too large: {err}"
    assert model.converged_


def test_recovery_with_sharing():
    """15% peptide sharing -> L1 error should be < 0.05."""
    data = generate_synthetic_community(
        n_taxa=5,
        n_peptides_per_taxon=200,
        shared_fraction=0.15,
        total_psms=15000,
        seed=11,
    )

    model = AbundanceEM(alpha=0.5, max_iter=500, tol=1e-7, seed=0)
    model.fit(data["A"], data["y"])

    metrics = evaluate_recovery(data["true_pi"], model.pi_)
    assert metrics["l1_error"] < 0.05, metrics


def test_sparse_community():
    """20 taxa in DB, 3 truly present -> EM with sparse prior should detect them."""
    n_taxa = 20
    rng = np.random.default_rng(3)

    # Build a community where only 3 taxa have nonzero true abundance.
    true_pi = np.zeros(n_taxa)
    present = [2, 9, 17]
    true_pi[present] = [0.5, 0.3, 0.2]

    data = generate_synthetic_community(
        n_taxa=n_taxa,
        n_peptides_per_taxon=80,
        shared_fraction=0.10,
        true_pi=true_pi,
        total_psms=20000,
        seed=42,
    )

    model = AbundanceEM(
        alpha=0.5,
        max_iter=1000,
        tol=1e-8,
        n_restarts=3,
        min_abundance=1e-3,
        seed=0,
    )
    model.fit(data["A"], data["y"])

    detected = np.where(model.pi_ > 1e-3)[0].tolist()
    assert set(detected) == set(present), f"detected={detected}, expected={present}"
    # Mass on the present taxa should be close to 1.
    assert model.pi_[present].sum() > 0.99


def test_em_monotonicity():
    """Log-posterior should be non-decreasing across EM iterations."""
    data = generate_synthetic_community(
        n_taxa=6,
        n_peptides_per_taxon=150,
        shared_fraction=0.20,
        total_psms=8000,
        seed=5,
    )

    model = AbundanceEM(alpha=0.7, max_iter=300, tol=1e-9, seed=0)
    model.fit(data["A"], data["y"])

    history = model.log_posterior_history_
    assert len(history) >= 2
    diffs = np.diff(history)
    # Allow tiny floating-point regressions on the order of EPS * |lp|.
    assert (diffs >= -1e-7 * (np.abs(history[:-1]) + 1.0)).all(), (
        f"non-monotonic step: {diffs.min()}"
    )


def test_standard_errors():
    """Well-identified -> small SEs; poorly identified -> larger SEs."""
    # Well-identified: disjoint peptide sets.
    true_pi = np.array([0.5, 0.3, 0.2])
    A_clean, y_clean = _build_disjoint_matrix(true_pi, peptides_per_taxon=100)
    model_clean = AbundanceEM(alpha=1.0, max_iter=500, tol=1e-8, seed=0)
    model_clean.fit(A_clean, y_clean)

    # Poorly identified: heavy sharing.
    data_shared = generate_synthetic_community(
        n_taxa=3,
        n_peptides_per_taxon=120,
        shared_fraction=0.85,
        total_psms=5000,
        seed=21,
    )
    model_shared = AbundanceEM(alpha=1.0, max_iter=500, tol=1e-8, seed=0)
    model_shared.fit(data_shared["A"], data_shared["y"])

    assert model_clean.standard_errors_.max() < 0.05
    assert model_shared.standard_errors_.max() > model_clean.standard_errors_.max()


def test_identifiability_report():
    """Collinear taxa and unique-peptide counts should be reported correctly."""
    # 3 taxa: A and B share all peptides, C is disjoint.
    A = np.zeros((10, 3), dtype=np.int8)
    A[:5, 0] = 1
    A[:5, 1] = 1
    A[5:, 2] = 1

    report = identifiability_report(A, taxon_names=["A", "B", "C"])

    assert report["unique_peptide_counts"] == {"A": 0, "B": 0, "C": 5}
    assert "A" in report["at_risk_taxa"]
    assert "B" in report["at_risk_taxa"]
    assert "C" not in report["at_risk_taxa"]
    assert any(set(group) == {"A", "B"} for group in report["collinear_groups"])
    assert report["rank"] == 2
    assert report["identifiable"] is False


def test_dirichlet_prior_effect():
    """Sparse prior (alpha < 1) should suppress absent taxa more than alpha=1."""
    # Build a database with 10 taxa, only 3 truly present.
    n_taxa = 10
    true_pi = np.zeros(n_taxa)
    true_pi[[1, 4, 7]] = [0.5, 0.3, 0.2]
    data = generate_synthetic_community(
        n_taxa=n_taxa,
        n_peptides_per_taxon=100,
        shared_fraction=0.15,
        true_pi=true_pi,
        total_psms=8000,
        seed=99,
    )

    absent = [i for i in range(n_taxa) if i not in (1, 4, 7)]

    sparse_model = AbundanceEM(alpha=0.5, max_iter=500, tol=1e-8, min_abundance=0.0, seed=0)
    sparse_model.fit(data["A"], data["y"])

    uniform_model = AbundanceEM(alpha=1.0, max_iter=500, tol=1e-8, min_abundance=0.0, seed=0)
    uniform_model.fit(data["A"], data["y"])

    sparse_absent = sparse_model.pi_[absent].sum()
    uniform_absent = uniform_model.pi_[absent].sum()
    assert sparse_absent <= uniform_absent + 1e-9, (
        f"sparse prior didn't suppress absent taxa: sparse={sparse_absent}, "
        f"uniform={uniform_absent}"
    )


def test_single_taxon():
    """Edge case: only one taxon -> pi == [1.0]."""
    A = np.ones((5, 1), dtype=np.int8)
    y = np.array([3, 1, 2, 0, 4])
    model = AbundanceEM(alpha=0.5, max_iter=10, tol=1e-8)
    model.fit(A, y)
    assert model.pi_.shape == (1,)
    assert pytest.approx(1.0, abs=1e-12) == float(model.pi_[0])
    assert model.converged_


def test_reproducibility_with_seed():
    """Same seed -> identical results."""
    data = generate_synthetic_community(
        n_taxa=4,
        n_peptides_per_taxon=120,
        shared_fraction=0.15,
        total_psms=6000,
        seed=33,
    )

    m1 = AbundanceEM(alpha=0.5, max_iter=500, tol=1e-7, n_restarts=3, seed=2024)
    m1.fit(data["A"], data["y"])
    m2 = AbundanceEM(alpha=0.5, max_iter=500, tol=1e-7, n_restarts=3, seed=2024)
    m2.fit(data["A"], data["y"])

    np.testing.assert_allclose(m1.pi_, m2.pi_, atol=1e-12)
    assert m1.n_iter_ == m2.n_iter_
