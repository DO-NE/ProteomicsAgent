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


# -- Biomass-correction validation -----------------------------------------


# Standard 20-AA alphabet. Uniform sampling over this set yields
# frequency(K) = frequency(R) = 1/20 = 5%, i.e. a combined tryptic
# cleavage-site rate of ~10%, matching the realistic target.
_AA_ALPHABET = np.array(list("ACDEFGHIKLMNPQRSTVWY"), dtype="<U1")
_BIOMASS_MIN_PEPTIDE_LEN = 7
_BIOMASS_MAX_PEPTIDE_LEN = 50


def _digest_random_proteome(
    n_proteins: int,
    avg_protein_length: int,
    rng: np.random.Generator,
) -> tuple:
    """Sample a random proteome and return ``(unique_peptides, total_aa)``.

    A simple trypsin rule is used: peptides end immediately after every
    ``K`` or ``R``, and the final protein-terminal fragment is also
    emitted. Only peptides in
    ``[_BIOMASS_MIN_PEPTIDE_LEN, _BIOMASS_MAX_PEPTIDE_LEN]`` are kept.
    """
    total_aa = int(n_proteins * avg_protein_length)
    if total_aa == 0:
        return set(), 0

    # Draw the entire proteome in one call and slice into proteins.
    codes = rng.integers(0, _AA_ALPHABET.shape[0], size=total_aa, dtype=np.int64)
    chars = _AA_ALPHABET[codes].reshape(n_proteins, avg_protein_length)

    peptide_set: set = set()
    for i in range(n_proteins):
        row = chars[i]
        # Vectorised cut-site lookup; the " + 1" pushes the boundary to the
        # index AFTER each K/R so that the K/R itself is the last residue
        # of the peptide it terminates.
        is_cut = (row == "K") | (row == "R")
        cut_positions = np.nonzero(is_cut)[0] + 1
        seq = "".join(row.tolist())
        start = 0
        for end in cut_positions:
            length = int(end) - start
            if _BIOMASS_MIN_PEPTIDE_LEN <= length <= _BIOMASS_MAX_PEPTIDE_LEN:
                peptide_set.add(seq[start:int(end)])
            start = int(end)
        # Trailing fragment (protein C-terminus without a final K/R).
        tail_length = avg_protein_length - start
        if _BIOMASS_MIN_PEPTIDE_LEN <= tail_length <= _BIOMASS_MAX_PEPTIDE_LEN:
            peptide_set.add(seq[start:])

    return peptide_set, total_aa


def generate_biomass_correction_test(
    n_taxa: int = 4,
    proteome_sizes: Optional[list] = None,
    avg_protein_lengths: Optional[list] = None,
    true_biomass: Optional[np.ndarray] = None,
    total_psms: int = 10000,
    shared_fraction: float = 0.15,
    seed: int = 42,
) -> dict:
    """Generate a synthetic community with taxon-specific PSM yield.

    Mimics a bacteria-to-eukaryote range by giving each taxon a different
    proteome size and average protein length. The per-taxon tryptic-yield
    factor ``g_t^(length) = n_tryptic / total_aa`` is computed empirically
    from the generated proteome, and expected PSM shares are drawn
    proportional to ``true_biomass[t] * g_t``. A model fit without
    corrections therefore recovers ``true_psm_proportions``, while a model
    fit with ``biomass_corrections=g_t`` recovers ``true_biomass``.

    Parameters
    ----------
    n_taxa : int, optional
        Number of taxa (default ``4``).
    proteome_sizes : list of int or None, optional
        Number of proteins per taxon. Defaults to
        ``[500, 2000, 5000, 14000]``.
    avg_protein_lengths : list of int or None, optional
        Average protein length (aa) per taxon. Defaults to
        ``[280, 300, 320, 420]``.
    true_biomass : np.ndarray or None, optional
        Ground-truth biomass proportions. Defaults to equal biomass
        (``[0.25, 0.25, 0.25, 0.25]`` when ``n_taxa == 4``).
    total_psms : int, optional
        Total spectral counts to simulate (default ``10000``).
    shared_fraction : float, optional
        Fraction of each taxon's peptides to promote to shared status by
        copying them to one other random taxon (default ``0.15``).
    seed : int, optional
        Random seed (default ``42``).

    Returns
    -------
    dict
        Keys: ``A``, ``y``, ``true_biomass``, ``true_psm_proportions``,
        ``g_t``, ``peptide_list``, ``taxon_labels``, ``proteome_sizes``,
        ``avg_protein_lengths``.
    """
    if proteome_sizes is None:
        proteome_sizes = [500, 2000, 5000, 14000]
    if avg_protein_lengths is None:
        avg_protein_lengths = [280, 300, 320, 420]
    if true_biomass is None:
        true_biomass = np.full(n_taxa, 1.0 / n_taxa)

    proteome_sizes = list(proteome_sizes)
    avg_protein_lengths = list(avg_protein_lengths)

    if len(proteome_sizes) != n_taxa:
        raise ValueError(
            f"proteome_sizes must have length {n_taxa}, got {len(proteome_sizes)}"
        )
    if len(avg_protein_lengths) != n_taxa:
        raise ValueError(
            f"avg_protein_lengths must have length {n_taxa}, got "
            f"{len(avg_protein_lengths)}"
        )
    if not 0.0 <= shared_fraction <= 1.0:
        raise ValueError("shared_fraction must lie in [0, 1]")

    true_biomass = np.asarray(true_biomass, dtype=np.float64)
    if true_biomass.shape != (n_taxa,):
        raise ValueError(f"true_biomass must have shape ({n_taxa},)")
    if not np.isclose(true_biomass.sum(), 1.0):
        true_biomass = true_biomass / true_biomass.sum()

    rng = np.random.default_rng(seed)

    # Step (a)-(c): generate proteomes, digest, compute g_length.
    taxon_peptide_sets: list = []
    total_aa: list = []
    for t in range(n_taxa):
        pep_set, aa_count = _digest_random_proteome(
            n_proteins=proteome_sizes[t],
            avg_protein_length=avg_protein_lengths[t],
            rng=rng,
        )
        taxon_peptide_sets.append(pep_set)
        total_aa.append(aa_count)

    g_t = np.array(
        [
            (len(taxon_peptide_sets[t]) / total_aa[t]) if total_aa[t] > 0 else 0.0
            for t in range(n_taxa)
        ],
        dtype=np.float64,
    )
    if (g_t <= 0).any():
        raise RuntimeError(
            "At least one synthetic taxon produced no detectable peptides; "
            "increase the proteome size or protein length."
        )

    # Seed the peptide-to-taxa map with origin memberships.
    peptide_to_taxa: dict = {}
    for t in range(n_taxa):
        for pep in taxon_peptide_sets[t]:
            peptide_to_taxa.setdefault(pep, set()).add(t)

    # Step (d): cross-contaminate by copying a fraction of each taxon's
    # peptides onto a random other taxon.
    if n_taxa > 1:
        for t in range(n_taxa):
            peps = list(taxon_peptide_sets[t])
            n_share = int(round(shared_fraction * len(peps)))
            if n_share <= 0:
                continue
            idx = rng.choice(len(peps), size=n_share, replace=False)
            others_template = [u for u in range(n_taxa) if u != t]
            for k in idx:
                pep = peps[int(k)]
                other = int(rng.choice(others_template))
                peptide_to_taxa[pep].add(other)

    # Step (f): build the binary peptide-taxon matrix. Sorting keeps the
    # row order deterministic across runs at the same seed.
    peptide_list = sorted(peptide_to_taxa.keys())
    P = len(peptide_list)
    A = np.zeros((P, n_taxa), dtype=np.int8)
    for i, pep in enumerate(peptide_list):
        for t in peptide_to_taxa[pep]:
            A[i, t] = 1

    # Step (e): expected PSM proportions are proportional to biomass * g.
    psm_num = true_biomass * g_t
    psm_sum = float(psm_num.sum())
    if psm_sum <= 0:
        raise RuntimeError("biomass * g_t is identically zero; cannot sample.")
    true_psm_proportions = psm_num / psm_sum

    # Step (g): sample y from Multinomial(total_psms, phi), where phi is
    # built by distributing each taxon's PSM share uniformly across its
    # peptides (post-sharing). phi_p = sum_t (psm_prop[t] * A[p,t] / n_t[t]).
    n_t = A.sum(axis=0).astype(np.float64)
    if (n_t == 0).any():
        raise RuntimeError("A synthetic taxon has no peptides after sharing.")
    M = A.astype(np.float64) / n_t[np.newaxis, :]
    phi = M @ true_psm_proportions
    phi_total = float(phi.sum())
    if phi_total <= 0:
        raise RuntimeError("phi vector is identically zero; cannot sample.")
    phi = phi / phi_total
    y = rng.multinomial(total_psms, phi)

    taxon_labels = [f"taxon_{i}" for i in range(n_taxa)]

    return {
        "A": A,
        "y": y,
        "true_biomass": true_biomass,
        "true_psm_proportions": true_psm_proportions,
        "g_t": g_t,
        "peptide_list": peptide_list,
        "taxon_labels": taxon_labels,
        "proteome_sizes": proteome_sizes,
        "avg_protein_lengths": avg_protein_lengths,
    }


def run_biomass_correction_validation(seed: int = 42) -> dict:
    """Run a comparative validation of biomass-corrected vs uncorrected EM.

    Generates a synthetic community in which taxa have markedly different
    proteome sizes and protein lengths. Fits two models on the same data:
    one without corrections (should recover ``true_psm_proportions``) and
    one with ``biomass_corrections=g_t`` (should recover ``true_biomass``).
    Prints a per-taxon comparison table and returns both recovery metric
    dicts.

    Returns
    -------
    dict
        Keys: ``uncorrected_metrics``, ``corrected_metrics``,
        ``uncorrected_pi``, ``corrected_pi``, ``true_biomass``,
        ``true_psm_proportions``, ``g_t``.
    """
    from .model import AbundanceEM

    data = generate_biomass_correction_test(seed=seed)
    true_biomass = data["true_biomass"]
    true_psm = data["true_psm_proportions"]
    g_t = data["g_t"]
    taxon_names = data["taxon_labels"]
    A = data["A"]
    y = data["y"]

    model_uncorr = AbundanceEM(
        alpha=0.5, max_iter=500, tol=1e-8, seed=0,
    )
    model_uncorr.fit(A, y)

    model_corr = AbundanceEM(
        alpha=0.5, max_iter=500, tol=1e-8, seed=0,
        biomass_corrections=g_t,
    )
    model_corr.fit(A, y)

    # Uncorrected pi should match the PSM simplex; corrected pi should
    # match the biomass simplex.
    uncorrected_metrics = evaluate_recovery(
        true_psm, model_uncorr.pi_, taxon_names=taxon_names,
    )
    corrected_metrics = evaluate_recovery(
        true_biomass, model_corr.pi_, taxon_names=taxon_names,
    )

    header = (
        f"{'Taxon':<10} {'TrueBio':>10} {'TruePSM':>10} "
        f"{'Uncorr':>10} {'Corr':>10} "
        f"{'Err(Unc)':>12} {'Err(Cor)':>12}"
    )
    divider = "=" * len(header)
    sub = "-" * len(header)

    print("\n" + divider)
    print("  Biomass Correction Validation")
    print(divider)
    print(header)
    print(sub)
    for i, name in enumerate(taxon_names):
        err_u = abs(float(model_uncorr.pi_[i]) - float(true_psm[i]))
        err_c = abs(float(model_corr.pi_[i]) - float(true_biomass[i]))
        print(
            f"{name:<10} "
            f"{true_biomass[i]:>10.4f} "
            f"{true_psm[i]:>10.4f} "
            f"{model_uncorr.pi_[i]:>10.4f} "
            f"{model_corr.pi_[i]:>10.4f} "
            f"{err_u:>12.4f} "
            f"{err_c:>12.4f}"
        )
    print(sub)
    print(
        f"{'L1 total':<10} "
        f"{'':>10} {'':>10} "
        f"{uncorrected_metrics['l1_error']:>10.4f} "
        f"{corrected_metrics['l1_error']:>10.4f}"
    )
    print(
        f"{'Cosine':<10} "
        f"{'':>10} {'':>10} "
        f"{uncorrected_metrics['cosine_similarity']:>10.4f} "
        f"{corrected_metrics['cosine_similarity']:>10.4f}"
    )
    print(
        f"{'KL div':<10} "
        f"{'':>10} {'':>10} "
        f"{uncorrected_metrics['kl_divergence']:>10.4f} "
        f"{corrected_metrics['kl_divergence']:>10.4f}"
    )
    print(divider)
    print(
        "\nInterpretation:"
        "\n  Uncorrected pi is expected to match True PSM%."
        "\n  Corrected pi is expected to match True Biomass."
    )
    print(
        f"  Uncorrected -> true_psm  L1 error : {uncorrected_metrics['l1_error']:.4f}"
    )
    print(
        f"  Corrected   -> true_bio  L1 error : {corrected_metrics['l1_error']:.4f}"
    )
    print()

    return {
        "uncorrected_metrics": uncorrected_metrics,
        "corrected_metrics": corrected_metrics,
        "uncorrected_pi": model_uncorr.pi_.copy(),
        "corrected_pi": model_corr.pi_.copy(),
        "true_biomass": true_biomass.copy(),
        "true_psm_proportions": true_psm.copy(),
        "g_t": g_t.copy(),
    }
