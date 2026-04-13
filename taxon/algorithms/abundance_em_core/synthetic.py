"""Synthetic community generators and evaluation metrics.

Used to validate :class:`AbundanceEM` against ground-truth abundances drawn
from a controlled generative process.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


def generate_synthetic_community(
    n_taxa: int = 5,
    n_peptides_per_taxon: int = 200,
    shared_fraction: float = 0.15,
    true_pi: Optional[np.ndarray] = None,
    total_psms: int = 10000,
    seed: int = 42,
) -> dict:
    """Generate a synthetic metaproteomics community with known ground truth.

    The generator first draws a private peptide set per taxon, then promotes a
    fraction of those peptides to be "shared" by reassigning them to a random
    pair of taxa. Spectral counts are sampled from the implied multinomial.

    Parameters
    ----------
    n_taxa : int, optional
        Number of taxa in the community (default ``5``).
    n_peptides_per_taxon : int, optional
        Number of peptides each taxon contributes before sharing reassignment
        (default ``200``). The realized total ``P`` will be slightly smaller
        than ``n_taxa * n_peptides_per_taxon`` because shared peptides count
        once in the row dimension.
    shared_fraction : float, optional
        Fraction of peptides that should map to at least two taxa
        (default ``0.15``). Must lie in ``[0, 1]``.
    true_pi : np.ndarray or None, optional
        True abundance vector. If ``None``, sampled from
        ``Dirichlet(2, ..., 2)``.
    total_psms : int, optional
        Total number of peptide-spectrum matches to simulate (default
        ``10000``).
    seed : int, optional
        Random seed (default ``42``).

    Returns
    -------
    dict
        Keys: ``A``, ``y``, ``true_pi``, ``peptide_names``, ``taxon_names``,
        ``phi``.

    Notes
    -----
    The returned ``A`` is a dense ``np.int8`` matrix; for synthetic sizes
    used in tests this is comfortably small.
    """
    if not 0.0 <= shared_fraction <= 1.0:
        raise ValueError("shared_fraction must lie in [0, 1]")
    if n_taxa < 1:
        raise ValueError("n_taxa must be >= 1")
    if n_peptides_per_taxon < 1:
        raise ValueError("n_peptides_per_taxon must be >= 1")

    rng = np.random.default_rng(seed)

    if true_pi is None:
        true_pi = rng.dirichlet(np.full(n_taxa, 2.0))
    else:
        true_pi = np.asarray(true_pi, dtype=np.float64)
        if true_pi.shape != (n_taxa,):
            raise ValueError(f"true_pi must have shape ({n_taxa},)")
        if not np.isclose(true_pi.sum(), 1.0):
            true_pi = true_pi / true_pi.sum()

    # Step 1: each taxon gets its own block of private peptides. Distinct
    # peptides are kept distinct even if their eventual membership signature
    # ends up identical to another peptide's — this preserves the row
    # dimension at ``n_taxa * n_peptides_per_taxon``.
    P = n_taxa * n_peptides_per_taxon
    A = np.zeros((P, n_taxa), dtype=np.int8)
    for t in range(n_taxa):
        start = t * n_peptides_per_taxon
        A[start : start + n_peptides_per_taxon, t] = 1

    # Step 2: promote a random subset of peptide rows to shared status by
    # adding one extra random taxon to their membership.
    n_shared = int(round(shared_fraction * P))
    if n_shared > 0 and n_taxa > 1:
        share_idx = rng.choice(P, size=n_shared, replace=False)
        for p in share_idx:
            current = int(np.where(A[p] > 0)[0][0])
            choices = [t for t in range(n_taxa) if t != current]
            extra = int(rng.choice(choices))
            A[p, extra] = 1

    # Step 4: simulate spectral counts y ~ Multinomial(N, phi(true_pi))
    n_t = A.sum(axis=0).astype(np.float64)
    if (n_t == 0).any():
        # Should not occur because each taxon receives n_peptides_per_taxon
        # private peptides, but guard anyway.
        raise RuntimeError("internal: a synthetic taxon has zero peptides")
    M = A.astype(np.float64) / n_t[np.newaxis, :]
    phi = M @ true_pi
    phi = phi / phi.sum()  # numerical safety
    y = rng.multinomial(total_psms, phi)

    peptide_names = [f"PEP_{i:05d}" for i in range(P)]
    taxon_names = [f"taxon_{i}" for i in range(n_taxa)]

    return {
        "A": A,
        "y": y,
        "true_pi": true_pi,
        "peptide_names": peptide_names,
        "taxon_names": taxon_names,
        "phi": phi,
    }


def evaluate_recovery(
    true_pi: np.ndarray,
    estimated_pi: np.ndarray,
    taxon_names: Optional[list] = None,
    presence_threshold: float = 1e-4,
) -> dict:
    """Compute discrepancy metrics between true and estimated abundances.

    Parameters
    ----------
    true_pi : np.ndarray
        Ground-truth abundance vector.
    estimated_pi : np.ndarray
        Inferred abundance vector. Must be the same length as ``true_pi``.
    taxon_names : list of str or None, optional
        Optional names for human-readable per-taxon output.
    presence_threshold : float, optional
        Threshold above which a taxon is considered "present" for the purpose
        of computing precision/recall metrics (default ``1e-4``).

    Returns
    -------
    dict
        Keys: ``l1_error``, ``l2_error``, ``max_error``, ``kl_divergence``,
        ``cosine_similarity``, ``per_taxon``, ``presence_detection``.
    """
    true_pi = np.asarray(true_pi, dtype=np.float64)
    estimated_pi = np.asarray(estimated_pi, dtype=np.float64)
    if true_pi.shape != estimated_pi.shape:
        raise ValueError(
            f"shape mismatch: true_pi {true_pi.shape}, "
            f"estimated_pi {estimated_pi.shape}"
        )

    diff = true_pi - estimated_pi
    l1_error = float(np.abs(diff).sum())
    l2_error = float(np.sqrt(np.sum(diff ** 2)))
    max_error = float(np.max(np.abs(diff)))

    # KL(true || estimated). Treat strict zeros in true_pi as contributing 0.
    eps = 1e-12
    mask = true_pi > 0
    kl = float(
        np.sum(true_pi[mask] * (np.log(true_pi[mask]) - np.log(np.maximum(estimated_pi[mask], eps))))
    )

    cos_num = float(np.dot(true_pi, estimated_pi))
    cos_den = float(np.linalg.norm(true_pi) * np.linalg.norm(estimated_pi)) + eps
    cosine_similarity = cos_num / cos_den

    true_present = true_pi > presence_threshold
    est_present = estimated_pi > presence_threshold
    tp = int(np.sum(true_present & est_present))
    fp = int(np.sum(~true_present & est_present))
    fn = int(np.sum(true_present & ~est_present))

    if taxon_names is None:
        taxon_names = [f"taxon_{i}" for i in range(true_pi.shape[0])]

    per_taxon = []
    for i, name in enumerate(taxon_names):
        per_taxon.append(
            {
                "taxon_name": name,
                "true": float(true_pi[i]),
                "estimated": float(estimated_pi[i]),
                "abs_error": float(abs(true_pi[i] - estimated_pi[i])),
            }
        )

    return {
        "l1_error": l1_error,
        "l2_error": l2_error,
        "max_error": max_error,
        "kl_divergence": kl,
        "cosine_similarity": cosine_similarity,
        "per_taxon": per_taxon,
        "presence_detection": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
        },
    }


def generate_detectability_test(
    n_taxa: int = 3,
    n_peptides_per_taxon: int = 100,
    shared_fraction: float = 0.10,
    true_pi: Optional[np.ndarray] = None,
    total_psms: int = 10000,
    seed: int = 42,
) -> dict:
    """Generate a synthetic community with non-uniform peptide detectability.

    The data is generated using a *weighted* emission model so that peptides
    with higher detectability scores contribute more spectral counts.  This
    simulates the real-world scenario where some peptides are inherently
    easier to detect than others.

    Running the EM algorithm **with** the correct detectability weights should
    recover the true abundances well, while running **without** (uniform mode)
    should produce biased estimates.

    Parameters
    ----------
    n_taxa : int, optional
        Number of taxa (default ``3``).
    n_peptides_per_taxon : int, optional
        Peptides per taxon before sharing (default ``100``).
    shared_fraction : float, optional
        Fraction of peptides shared between taxa (default ``0.10``).
    true_pi : np.ndarray or None, optional
        True abundance vector.  If ``None``, uses ``[0.5, 0.3, 0.2]``.
    total_psms : int, optional
        Total spectral counts to simulate (default ``10000``).
    seed : int, optional
        Random seed (default ``42``).

    Returns
    -------
    dict
        Keys: ``A``, ``y``, ``true_pi``, ``detectability_weights``,
        ``W``, ``phi``, ``peptide_names``, ``taxon_names``.
    """
    rng = np.random.default_rng(seed)

    if true_pi is None:
        true_pi = np.array([0.5, 0.3, 0.2])
    true_pi = np.asarray(true_pi, dtype=np.float64)
    n_taxa = true_pi.shape[0]
    if not np.isclose(true_pi.sum(), 1.0):
        true_pi = true_pi / true_pi.sum()

    P = n_taxa * n_peptides_per_taxon
    A = np.zeros((P, n_taxa), dtype=np.int8)
    for t in range(n_taxa):
        start = t * n_peptides_per_taxon
        A[start : start + n_peptides_per_taxon, t] = 1

    # Promote a fraction of peptides to shared status.
    n_shared = int(round(shared_fraction * P))
    if n_shared > 0 and n_taxa > 1:
        share_idx = rng.choice(P, size=n_shared, replace=False)
        for p in share_idx:
            current = int(np.where(A[p] > 0)[0][0])
            choices = [t for t in range(n_taxa) if t != current]
            extra = int(rng.choice(choices))
            A[p, extra] = 1

    # Assign per-taxon biased detectability: taxon 0's peptides are highly
    # detectable, taxon (n_taxa-1)'s are poorly detectable.  This creates
    # a scenario where the uniform model systematically overestimates taxa
    # with highly detectable peptides and underestimates the rest.
    d = np.empty(P, dtype=np.float64)
    for t in range(n_taxa):
        start = t * n_peptides_per_taxon
        end = start + n_peptides_per_taxon
        # Mean detectability decreases linearly from 0.8 to 0.2 across taxa.
        centre = 0.8 - 0.6 * t / max(n_taxa - 1, 1)
        d[start:end] = np.clip(
            rng.normal(loc=centre, scale=0.1, size=n_peptides_per_taxon),
            0.01, 1.0,
        )

    # Build weighted emission matrix.
    A_float = A.astype(np.float64)
    dA = A_float * d[:, np.newaxis]
    col_sums = dA.sum(axis=0)
    col_sums_safe = np.where(col_sums == 0, 1.0, col_sums)
    W = dA / col_sums_safe[np.newaxis, :]

    # Generate spectral counts from the weighted emission.
    phi = W @ true_pi
    phi = phi / phi.sum()  # numerical safety
    y = rng.multinomial(total_psms, phi)

    peptide_names = [f"PEP_{i:05d}" for i in range(P)]
    taxon_names = [f"taxon_{i}" for i in range(n_taxa)]

    return {
        "A": A,
        "y": y,
        "true_pi": true_pi,
        "detectability_weights": d,
        "W": W,
        "phi": phi,
        "peptide_names": peptide_names,
        "taxon_names": taxon_names,
    }


def run_detectability_validation(seed: int = 42) -> dict:
    """Run a comparative validation of detectability-corrected vs uniform EM.

    Generates synthetic data with non-uniform detectability, then fits both
    a corrected model (with the true weights) and a uniform model, and
    prints a comparison table.

    Returns
    -------
    dict
        Keys: ``corrected_metrics``, ``uniform_metrics``,
        ``corrected_pi``, ``uniform_pi``, ``true_pi``.
    """
    from .model import AbundanceEM

    data = generate_detectability_test(seed=seed)
    true_pi = data["true_pi"]
    taxon_names = data["taxon_names"]

    # --- Corrected model: inject the true detectability weights. ---
    model_corrected = AbundanceEM(
        alpha=0.5, max_iter=500, tol=1e-8, seed=0,
        detectability_weights=data["detectability_weights"],
    )
    model_corrected.fit(data["A"], data["y"])

    # --- Uniform model: no detectability correction. ---
    model_uniform = AbundanceEM(
        alpha=0.5, max_iter=500, tol=1e-8, seed=0,
        detectability_mode="uniform",
    )
    model_uniform.fit(data["A"], data["y"])

    metrics_corrected = evaluate_recovery(
        true_pi, model_corrected.pi_, taxon_names=taxon_names,
    )
    metrics_uniform = evaluate_recovery(
        true_pi, model_uniform.pi_, taxon_names=taxon_names,
    )

    # Print comparison table.
    print("\n" + "=" * 72)
    print("  Detectability Correction Validation")
    print("=" * 72)
    header = f"{'Taxon':<12} {'True':>8} {'Corrected':>10} {'Uniform':>10}"
    print(header)
    print("-" * len(header))
    for i, name in enumerate(taxon_names):
        print(
            f"{name:<12} {true_pi[i]:>8.4f} "
            f"{model_corrected.pi_[i]:>10.4f} "
            f"{model_uniform.pi_[i]:>10.4f}"
        )
    print("-" * len(header))
    print(
        f"{'L1 error':<12} {'':>8} "
        f"{metrics_corrected['l1_error']:>10.4f} "
        f"{metrics_uniform['l1_error']:>10.4f}"
    )
    print(
        f"{'Cosine sim':<12} {'':>8} "
        f"{metrics_corrected['cosine_similarity']:>10.4f} "
        f"{metrics_uniform['cosine_similarity']:>10.4f}"
    )
    print(
        f"{'KL div':<12} {'':>8} "
        f"{metrics_corrected['kl_divergence']:>10.4f} "
        f"{metrics_uniform['kl_divergence']:>10.4f}"
    )
    print("=" * 72)

    improvement = metrics_uniform["l1_error"] - metrics_corrected["l1_error"]
    if improvement > 0:
        print(
            f"\nDetectability correction reduced L1 error by "
            f"{improvement:.4f} ({improvement / metrics_uniform['l1_error'] * 100:.1f}%)"
        )
    else:
        print("\nNote: uniform model performed comparably on this seed.")
    print()

    return {
        "corrected_metrics": metrics_corrected,
        "uniform_metrics": metrics_uniform,
        "corrected_pi": model_corrected.pi_.copy(),
        "uniform_pi": model_uniform.pi_.copy(),
        "true_pi": true_pi.copy(),
    }
