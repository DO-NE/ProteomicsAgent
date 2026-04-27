"""
Proteome-mass correction for taxon abundance estimation.

Converts PSM-level relative abundance (π_t) from AbundanceEM into
protein-biomass-level relative abundance (b_t) by weighting each
taxon's PSM fraction by its proteome size W_t.

Theory
------
PSM-level abundance overestimates taxa with large proteomes because
more proteins → more tryptic peptides → more PSMs per cell.
Multiplying π_t by W_t (number of proteins in the reference FASTA
for taxon t) approximately corrects for this bias:

    b_t ∝ π_t × W_t

For multi-strain taxa sharing a FASTA prefix (e.g. three Salmonella
strains all prefixed LT2_), W_t automatically reflects the combined
proteome size, which is a key advantage over external database lookups.

W_t is computed directly from the MappingMatrixResult, which already
parsed the FASTA and knows which proteins belong to which taxon.
No external database, no genome size lookup, no network access needed.

Reference: Kleiner et al. 2017 Nat. Commun. (iBAQ baseline);
           Wiśniewski & Mann 2014 (Proteomic Ruler concept).
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

    # Unnormalized weighted values (π_t × W_t) before normalization
    weighted_signal: np.ndarray

    # Diagnostics
    min_proteome_size: int
    max_proteome_size: int
    median_proteome_size: float
    n_taxa: int


def compute_proteome_sizes(
    taxon_protein_peptides: dict[str, dict[str, list[str]]],
    taxon_labels: list[str],
) -> np.ndarray:
    """
    Compute W_t for each taxon: number of proteins in the FASTA for that taxon.

    Parameters
    ----------
    taxon_protein_peptides : dict
        From MappingMatrixResult.taxon_protein_peptides:
        taxon_label -> {protein_accession -> [peptide_sequences]}
        This already contains the complete mapping of proteins to taxa
        as parsed from the FASTA.
    taxon_labels : list[str]
        Ordered list of taxon labels matching the EM's column ordering.

    Returns
    -------
    np.ndarray, shape (T,)
        W_t values in the same order as taxon_labels.

    Notes
    -----
    W_t counts the number of PROTEIN ENTRIES per taxon in the FASTA,
    not the number of unique peptides. This is the correct denominator
    because it reflects proteome size (total number of expressed gene
    products), analogous to the iBAQ denominator at the protein level.

    For taxa with zero proteins (should not happen but handle gracefully),
    W_t is set to 1 to avoid division by zero.
    """
    sizes = np.zeros(len(taxon_labels), dtype=np.float64)
    for t, label in enumerate(taxon_labels):
        prot_map = taxon_protein_peptides.get(label, {})
        n_proteins = len(prot_map)
        sizes[t] = max(n_proteins, 1)
        if n_proteins == 0:
            logger.warning(
                "Taxon %r has zero proteins in taxon_protein_peptides; "
                "setting W_t = 1 as fallback",
                label,
            )
    return sizes


def compute_biomass_abundance(
    pi: np.ndarray,
    proteome_sizes: np.ndarray,
    taxon_labels: list[str],
) -> ProteomeMassCorrectionResult:
    """
    Compute protein-biomass relative abundance via proteome-size weighting.

    Parameters
    ----------
    pi : np.ndarray, shape (T,)
        PSM-level abundance from AbundanceEM (sums to 1).
    proteome_sizes : np.ndarray, shape (T,)
        W_t values from compute_proteome_sizes().
    taxon_labels : list[str]
        Taxon labels in the same order as pi and proteome_sizes.

    Returns
    -------
    ProteomeMassCorrectionResult

    Algorithm
    ---------
    1. Compute weighted signal: w_t = π_t × W_t
    2. Normalize: b_t = w_t / Σ w_t
    3. Return ProteomeMassCorrectionResult with all fields populated.
    """
    pi = np.asarray(pi, dtype=np.float64)
    proteome_sizes = np.asarray(proteome_sizes, dtype=np.float64)

    weighted = pi * proteome_sizes
    total = float(weighted.sum())
    if total > 0:
        biomass = weighted / total
    else:
        logger.warning(
            "Proteome-mass correction: weighted signal sums to zero; "
            "returning uniform biomass abundance"
        )
        T = len(pi)
        biomass = np.full(T, 1.0 / T) if T > 0 else np.zeros(0)

    sizes_int = proteome_sizes.astype(np.int64)
    return ProteomeMassCorrectionResult(
        biomass_abundance=biomass,
        psm_abundance=pi.copy(),
        proteome_sizes=proteome_sizes.copy(),
        taxon_labels=list(taxon_labels),
        weighted_signal=weighted,
        min_proteome_size=int(sizes_int.min()) if len(sizes_int) > 0 else 0,
        max_proteome_size=int(sizes_int.max()) if len(sizes_int) > 0 else 0,
        median_proteome_size=float(np.median(sizes_int)) if len(sizes_int) > 0 else 0.0,
        n_taxa=len(taxon_labels),
    )


def log_proteome_mass_diagnostics(
    result: ProteomeMassCorrectionResult,
    logger=None,
) -> str:
    """
    Generate diagnostic report for proteome-mass correction.

    Reports:
    - Proteome size range (min, max, median across taxa)
    - Top taxa by biomass abundance vs PSM abundance (to highlight corrections)
    - Largest absolute shifts: taxa where b_t differs most from π_t

    Returns report as string, also logs if logger provided.
    """
    lines: list[str] = []
    lines.append("=== Proteome-Mass Correction ===")
    lines.append(
        f"Taxa: {result.n_taxa}  |  "
        f"Proteome sizes — min: {result.min_proteome_size}, "
        f"max: {result.max_proteome_size}, "
        f"median: {result.median_proteome_size:.1f}"
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
    lines.append("Largest corrections (|b_t - π_t|):")
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
