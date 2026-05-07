"""
Genome-normalized cell-number abundance correction for taxon abundance estimation.

Converts PSM-level relative abundance (π_t) from AbundanceEM into a
genome-normalized cell-number abundance (b_t) by dividing each taxon's
PSM fraction by a power of its proteome capacity W_t.

Theory
------
Under the Total Protein Approach (TPA), π_t is already directly
proportional to taxon t's protein-biomass contribution — no upward
correction is needed at the biomass level.  However, larger genomes
correspond to larger cells with proportionally more protein per cell,
so dividing biomass by a per-cell-protein proxy yields a cell-number
estimate.  The full formula generalises the original linear form

    b_t ∝ π_t / W_t                                   (α = 1)

to a configurable scaling exponent α ≥ 0:

    b_t = (π_t / W_t^α) / Σ_{t'} (π_{t'} / W_{t'}^α)

Choice of α
-----------
* α = 0 disables the correction (b_t collapses to π_t after renormalization).
* α = 1 is the original linear form (Pible et al. 2020 Microbiome,
  Kleiner et al. 2017 Nat. Commun. TPA baseline).
* α = 4.8 (the new default) is motivated by Kempes et al. 2016
  (ISME J), which fits a log-log slope of 0.21 between bacterial cell
  volume and genome size (so cell volume ∝ G^4.8).  Per Milo 2013
  (BioEssays 35:1050) the per-cell protein concentration c_p is a
  near-universal bacterial constant (~0.2–0.3 g/mL), hence per-cell
  protein biomass ∝ cell volume ∝ G^4.8.  The taxon-invariant
  constants cancel under normalization, leaving W_t^α as the
  appropriate denominator.

W_t is computed directly from the MappingMatrixResult, which already
parsed the FASTA and knows which proteins belong to which taxon.
No external database, no genome size lookup, no network access needed.

Reference: Pible et al. 2020 Microbiome (cell-volume normalization);
           Kleiner et al. 2017 Nat. Commun. (TPA baseline);
           Milo 2013 BioEssays 35:1050 (constant c_p across bacteria);
           Kempes et al. 2016 ISME J (cell-volume vs genome size scaling).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ProteomeMassCorrectionResult:
    """Results of proteome-mass correction."""

    # Protein-biomass relative abundance (sums to 1), shape (T,)
    biomass_abundance: np.ndarray

    # PSM-level abundance from EM (for comparison), shape (T,)
    psm_abundance: np.ndarray

    # Per-taxon proteome size (protein count from FASTA), shape (T,)
    proteome_sizes: np.ndarray

    # Taxon labels matching column order
    taxon_labels: list[str]

    # Unnormalized weighted values (π_t / W_t^α) before normalization
    weighted_signal: np.ndarray

    # Diagnostics
    min_proteome_size: int
    max_proteome_size: int
    median_proteome_size: float
    n_taxa: int

    # Scaling exponent used (b_t ∝ π_t / W_t^α).  α=1 is the legacy
    # linear form; α=4.8 (default) is Kempes-2016 bacterial scaling;
    # α=0 disables the correction.
    alpha: float = 1.0


def compute_proteome_sizes(
    taxon_total_protein_counts: dict[str, int],
    taxon_labels: list[str],
) -> np.ndarray:
    """
    Compute W_t for each taxon: number of proteins in the FASTA for that taxon.

    Parameters
    ----------
    taxon_total_protein_counts : dict
        From MappingMatrixResult.taxon_total_protein_counts:
        taxon_label -> total protein count from FASTA (all entries, including
        those with no observed peptides).
    taxon_labels : list[str]
        Ordered list of taxon labels matching the EM's column ordering.

    Returns
    -------
    np.ndarray, shape (T,)
        W_t values in the same order as taxon_labels.

    Notes
    -----
    W_t counts the number of PROTEIN ENTRIES per taxon in the FASTA,
    not the number of unique peptides.  It serves as a genome-derived
    proxy for per-cell proteome capacity: taxa whose reference FASTA
    contains more proteins are assumed to have proportionally more
    protein per cell.  Dividing π_t by W_t^α therefore converts
    protein-biomass abundance into a genome-normalized cell-number
    estimate (see module docstring for the choice of α).

    For taxa with zero proteins (should not happen but handle gracefully),
    W_t is set to 1 to avoid division by zero.
    """
    sizes = np.zeros(len(taxon_labels), dtype=np.float64)
    for t, label in enumerate(taxon_labels):
        n_proteins = taxon_total_protein_counts.get(label, 0)
        sizes[t] = max(n_proteins, 1)
        if n_proteins == 0:
            logger.warning(
                "Taxon %r has zero proteins in taxon_total_protein_counts; "
                "setting W_t = 1 as fallback",
                label,
            )
    return sizes


def compute_biomass_abundance(
    pi: np.ndarray,
    proteome_sizes: np.ndarray,
    taxon_labels: list[str],
    alpha: float = 4.8,
) -> ProteomeMassCorrectionResult:
    """
    Compute genome-normalized cell-number abundance via proteome-size weighting.

    Implements

        b_t = (π_t / W_t^α) / Σ_{t'} (π_{t'} / W_{t'}^α)

    Parameters
    ----------
    pi : np.ndarray, shape (T,)
        PSM-level abundance from AbundanceEM (sums to 1).
    proteome_sizes : np.ndarray, shape (T,)
        W_t values from :func:`compute_proteome_sizes`.  Any zero entries
        are replaced with 1 locally to prevent division by zero.
    taxon_labels : list[str]
        Taxon labels in the same order as ``pi`` and ``proteome_sizes``.
    alpha : float, default 4.8
        Genome-scaling exponent.  Must be ≥ 0.
        - ``alpha = 0`` disables the correction (b_t collapses to π_t).
        - ``alpha = 1`` reproduces the original linear form
          (Pible 2020 / Kleiner 2017 TPA) bit-identically — the
          ``W_t^α`` step is short-circuited so there is no rounding
          drift relative to the pre-α implementation.
        - ``alpha = 4.8`` (the default) follows Kempes-2016 bacterial
          genome-volume scaling: cell volume ∝ G^4.8, and Milo-2013
          fixes c_p so per-cell protein biomass tracks cell volume.

    Returns
    -------
    ProteomeMassCorrectionResult
        With ``alpha`` recorded on the result for downstream provenance.

    Algorithm
    ---------
    1. Validate ``alpha >= 0`` (raise ``ValueError`` otherwise).
    2. Guard against zero proteome sizes (W_t = 0 → W_t = 1).
    3. Compute weighted signal:
         - ``alpha == 1``: ``w_t = π_t / W_t`` (legacy code path).
         - else:           ``w_t = π_t * exp(-α * log(W_t))``
                           (log-space form that avoids overflow at
                           large W_t and α).
    4. Normalize: ``b_t = w_t / Σ w_t``.
    5. Return ``ProteomeMassCorrectionResult`` with all fields populated.

    Raises
    ------
    ValueError
        If ``alpha`` is negative.
    """
    if alpha < 0:
        raise ValueError(
            f"genome_scaling_exponent (alpha) must be >= 0, got {alpha!r}"
        )

    pi = np.asarray(pi, dtype=np.float64)
    proteome_sizes = np.asarray(proteome_sizes, dtype=np.float64)
    alpha_f = float(alpha)

    logger.info(
        "Proteome-mass correction: alpha=%.4f (W_t^alpha denominator). "
        "alpha=0 disables, alpha=1 is the linear TPA form, "
        "alpha=4.8 is Kempes-2016 bacterial genome-volume scaling.",
        alpha_f,
    )

    # Guard: zero proteome size → fall back to W_t = 1 (taxa with no FASTA
    # entries are treated as if they have a single-protein genome so that
    # π_t is returned unchanged for those taxa after normalization).
    sizes_safe = np.where(proteome_sizes == 0, 1.0, proteome_sizes)

    # α=1 must be byte-for-byte identical to the previous behaviour: keep
    # the original division so a regression test against the old
    # implementation passes exactly.
    if alpha_f == 1.0:
        weighted = pi / sizes_safe
    elif alpha_f == 0.0:
        # W_t^0 = 1 for every taxon → b_t reduces to π_t after renormalization.
        weighted = pi.copy()
    else:
        # Log-space exponentiation.  All `sizes_safe` entries are >= 1, so
        # `log` is finite and we never underflow on `exp(-alpha * log W)`.
        weighted = pi * np.exp(-alpha_f * np.log(sizes_safe))

    total = float(weighted.sum())
    if total > 0:
        biomass = weighted / total
    else:
        logger.warning(
            "Proteome-mass correction: weighted signal sums to zero; "
            "returning uniform cell-number abundance"
        )
        T = len(pi)
        biomass = np.full(T, 1.0 / T) if T > 0 else np.zeros(0)

    sizes_int = proteome_sizes.astype(np.int64)
    return ProteomeMassCorrectionResult(
        biomass_abundance=biomass,   # column name kept for backwards compatibility
        psm_abundance=pi.copy(),
        proteome_sizes=proteome_sizes.copy(),
        taxon_labels=list(taxon_labels),
        weighted_signal=weighted,
        min_proteome_size=int(sizes_int.min()) if len(sizes_int) > 0 else 0,
        max_proteome_size=int(sizes_int.max()) if len(sizes_int) > 0 else 0,
        median_proteome_size=float(np.median(sizes_int)) if len(sizes_int) > 0 else 0.0,
        n_taxa=len(taxon_labels),
        alpha=alpha_f,
    )


def log_proteome_mass_diagnostics(
    result: ProteomeMassCorrectionResult,
    logger=None,
) -> str:
    """
    Generate diagnostic report for genome-normalized cell-number correction.

    Reports:
    - Scaling exponent α used for this run
    - Proteome size range (min, max, median across taxa)
    - Top taxa by cell-number abundance vs PSM abundance (to highlight corrections)
    - Largest absolute shifts: taxa where b_t differs most from π_t

    Returns report as string, also logs if logger provided.
    """
    lines: list[str] = []
    lines.append("=== Genome-Normalized Cell-Number Correction ===")
    lines.append(
        f"alpha (genome-scaling exponent): {result.alpha:.4f}  |  "
        f"Taxa: {result.n_taxa}  |  "
        f"Proteome sizes — min: {result.min_proteome_size}, "
        f"max: {result.max_proteome_size}, "
        f"median: {result.median_proteome_size:.1f}"
    )

    # One-line append for downstream tooling (Task 4): pick out the
    # b_t / pi_hat ratio range over taxa with pi_hat > 0.
    pi = np.asarray(result.psm_abundance, dtype=np.float64)
    bt = np.asarray(result.biomass_abundance, dtype=np.float64)
    nz = pi > 0
    if nz.any():
        ratios = bt[nz] / pi[nz]
        ratio_min = float(ratios.min())
        ratio_max = float(ratios.max())
    else:
        ratio_min = float("nan")
        ratio_max = float("nan")
    lines.append(
        f"Proteome-mass correction: alpha={result.alpha:.4f}. "
        f"W_t range: [{result.min_proteome_size}, {result.max_proteome_size}]. "
        f"b_t / pi_hat ratio range: [{ratio_min:.4g}, {ratio_max:.4g}]."
    )

    order = np.argsort(-result.biomass_abundance)
    lines.append("")
    lines.append(
        f"{'Taxon':<40} {'PSM-pi':>9} {'Biomass-b':>10} {'W_t':>8} {'Shift':>9}"
    )
    top_n = min(25, result.n_taxa)
    for rank, t in enumerate(order):
        if rank >= top_n:
            break
        lbl = result.taxon_labels[t]
        name = lbl.split("|", 1)[-1][:40]
        shift = result.biomass_abundance[t] - result.psm_abundance[t]
        lines.append(
            f"{name:<40} "
            f"{result.psm_abundance[t]:>9.4f} "
            f"{result.biomass_abundance[t]:>10.4f} "
            f"{int(result.proteome_sizes[t]):>8d} "
            f"{shift:>+9.4f}"
        )

    # Largest absolute shifts
    shifts = np.abs(result.biomass_abundance - result.psm_abundance)
    shift_order = np.argsort(-shifts)
    lines.append("")
    lines.append("Largest cell-number corrections (|b_t - π_t|):")
    for rank, t in enumerate(shift_order[:10]):
        if shifts[t] < 1e-6:
            break
        lbl = result.taxon_labels[t]
        name = lbl.split("|", 1)[-1][:40]
        sign = "+" if result.biomass_abundance[t] >= result.psm_abundance[t] else "-"
        lines.append(
            f"  {name}: π={result.psm_abundance[t]:.4f} → b={result.biomass_abundance[t]:.4f} "
            f"({sign}{shifts[t]:.4f})"
        )

    lines.append("=== END ===")
    report = "\n".join(lines)
    if logger is not None:
        for ln in lines:
            logger.info(ln)
    return report
