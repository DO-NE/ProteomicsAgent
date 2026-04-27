"""Post-EM marker-based cell-equivalent abundance correction.

The EM in :mod:`taxon.algorithms.abundance_em_core.model` estimates a
PSM-level relative abundance vector ``pi`` (the fraction of observed
spectra attributable to each taxon).  PSM signal is biased by proteome
size and per-organism MS detectability: a taxon with twice as many
proteins, or twice the MS-friendly chemistry, will look twice as
abundant even at equal cell counts.

This module implements a *cell-equivalent* correction that runs after
EM convergence.  The idea, following GTDB-Tk / Parks-2018 marker
phylogenetics, is to restrict signal to a small set of universal
single-copy proteins (the bac120 / ar53 marker set) so that each
contributing taxon contributes ~one gene-copy worth of marker peptide
mass per cell.  Cell-equivalent relative abundance is then

    c_t = s_t / Σ_{t'} s_{t'},     s_t = Σ_p y_p * r_{pt}  for marker p

where ``y_p`` is the observed PSM count of peptide ``p`` and
``r_{pt}`` is the EM responsibility (so shared marker peptides are
fractionally assigned by the EM rather than discarded).  Taxa that
fail minimum-evidence thresholds fall back to their ``pi_t`` value.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MarkerCorrectionResult:
    """Output of :func:`compute_cell_equivalent_abundance`.

    All vector attributes are aligned with the column order of the EM
    mapping matrix: ``cell_abundance[t]`` corresponds to
    ``taxon_labels[t]``.
    """

    cell_abundance: np.ndarray            # (T,) cell-equivalent rel. abundance
    psm_abundance: np.ndarray             # (T,) original pi from the EM
    marker_signal: np.ndarray             # (T,) Σ_p y_p * r_{pt} over marker p
    marker_families_per_taxon: np.ndarray  # (T,) int, distinct families w/ signal > 0.5
    marker_psm_count: np.ndarray          # (T,) fractional marker PSM count
    marker_peptides_per_taxon: np.ndarray  # (T,) unique marker peptides w/ r > 0
    taxon_labels: list                    # column-aligned label strings
    has_marker_estimate: np.ndarray       # (T,) bool — passed thresholds
    total_marker_psms: float              # global Σ marker_psm_count
    total_marker_peptides: int            # # unique marker peptide rows used
    fraction_psms_from_markers: float     # total_marker_psms / Σ y
    # Per-taxon family hit map: ``label -> {family: signal}``.  Useful
    # for diagnostics ("which markers fired in *Bacillus subtilis*?").
    family_signal_per_taxon: dict = field(default_factory=dict)


def compute_cell_equivalent_abundance(
    pi: np.ndarray,
    responsibilities: np.ndarray,
    spectral_counts: dict,
    mapping_matrix: np.ndarray,
    taxon_labels: list,
    peptide_index: dict,
    marker_proteins: dict,
    taxon_protein_peptides: dict,
    min_marker_families: int = 3,
    min_marker_psms: float = 1.0,
) -> MarkerCorrectionResult:
    """Convert PSM-level ``pi`` to a cell-equivalent relative abundance.

    Parameters
    ----------
    pi : np.ndarray, shape ``(T,)``
        PSM-level abundance vector returned by :class:`AbundanceEM`.
    responsibilities : np.ndarray, shape ``(P, T)``
        EM responsibility matrix ``r_{pt}``.
    spectral_counts : dict[str, int]
        ``peptide_sequence -> observed PSM count``.
    mapping_matrix : np.ndarray, shape ``(P, T)``
        Peptide-to-taxon binary mapping ``A``.  Currently used only for
        the diagnostic ``total_psms`` denominator; the correction itself
        depends on ``responsibilities`` and ``spectral_counts``.
    taxon_labels : list[str]
        Column labels of ``A`` in ``"taxid|name"`` form.
    peptide_index : dict[str, int]
        ``peptide_sequence -> row index in A / responsibilities``.
    marker_proteins : dict
        ``protein_accession -> (taxon_label, marker_family, evalue, score)``.
        Typically obtained from
        :func:`taxon.algorithms.abundance_em_core.hmm_marker_search.run_hmmsearch`.
        The ``taxon_label`` in this tuple is informational; cross-referencing
        against :paramref:`taxon_protein_peptides` is what actually drives
        the calculation, so taxa whose accessions never produced an
        observed peptide simply contribute zero signal.
    taxon_protein_peptides : dict
        ``taxon_label -> {protein_accession -> [observed peptides]}``.
        Provided by :class:`MappingMatrixResult`.
    min_marker_families : int, default ``3``
        Minimum number of distinct marker families with nontrivial signal
        (per-taxon family signal > 0.5) required for a taxon to receive a
        marker-based estimate.
    min_marker_psms : float, default ``1.0``
        Minimum total fractional marker PSM count for a taxon.

    Returns
    -------
    MarkerCorrectionResult
        Cell-equivalent abundance plus diagnostics.  Taxa that fail the
        thresholds keep their PSM-level value (``pi_t``); the combined
        vector is renormalised to sum to 1.

    Notes
    -----
    The signal model treats every observed marker peptide as contributing
    proportionally to cell count.  Differential MS detectability between
    organisms is *not* corrected here — the assumption is that, averaged
    over the bac120 set, marker-peptide detectability is roughly comparable
    between bacteria in the same sample.  For organisms whose marker set
    is systematically harder or easier to detect, the cell-equivalent
    estimate inherits that bias.
    """
    pi = np.asarray(pi, dtype=np.float64).ravel()
    responsibilities = np.asarray(responsibilities, dtype=np.float64)
    T = pi.shape[0]
    if responsibilities.shape[1] != T:
        raise ValueError(
            f"responsibilities columns ({responsibilities.shape[1]}) must "
            f"match length of pi ({T})"
        )
    if len(taxon_labels) != T:
        raise ValueError(
            f"taxon_labels length ({len(taxon_labels)}) must match length "
            f"of pi ({T})"
        )

    label_to_idx = {lbl: i for i, lbl in enumerate(taxon_labels)}

    # ------------------------------------------------------------------ step 1
    # Marker peptide -> set of marker families.  A peptide may come from
    # several markers (e.g. ribosomal RpL2 and RpL14 both map to the same
    # short conserved tryptic fragment), in which case it counts toward
    # both family memberships.
    peptide_to_families: dict = defaultdict(set)
    # Also remember which taxon-buckets each marker peptide came from, so
    # the diagnostic counters reflect actually-observed taxa.
    peptide_to_taxa: dict = defaultdict(set)

    n_marker_proteins_seen = 0
    for accession, payload in marker_proteins.items():
        try:
            _hmm_taxon_label, family, _evalue, _score = payload
        except (TypeError, ValueError):
            logger.debug(
                "skipping malformed marker_proteins entry for %r: %r",
                accession, payload,
            )
            continue
        # Find this accession in any taxon bucket.  Marker proteins not
        # represented in taxon_protein_peptides (e.g. unclassified
        # proteins, or ones whose digest produced no observed peptides)
        # are silently skipped — they cannot contribute signal.
        appears_in = []
        for taxon_label, prot_map in taxon_protein_peptides.items():
            peps = prot_map.get(accession)
            if peps:
                appears_in.append((taxon_label, peps))
        if not appears_in:
            continue
        n_marker_proteins_seen += 1
        for taxon_label, peps in appears_in:
            for pep in peps:
                peptide_to_families[pep].add(family)
                peptide_to_taxa[pep].add(taxon_label)

    marker_peptides = sorted(peptide_to_families.keys())
    n_marker_peptide_rows = sum(1 for p in marker_peptides if p in peptide_index)

    logger.info(
        "Marker correction: %d marker proteins matched into %d taxa, "
        "%d unique marker peptides observed (%d with non-zero rows in A)",
        n_marker_proteins_seen,
        len({tl for tls in peptide_to_taxa.values() for tl in tls}),
        len(marker_peptides),
        n_marker_peptide_rows,
    )

    # ------------------------------------------------------------------ step 3
    # Per-taxon marker signal s_t and per-(family, t) breakdown.
    marker_signal = np.zeros(T, dtype=np.float64)
    marker_psm_count = np.zeros(T, dtype=np.float64)
    marker_peptides_per_taxon = np.zeros(T, dtype=np.int64)
    family_signal: dict = defaultdict(lambda: defaultdict(float))
    # Per-taxon, set of marker peptides with non-zero responsibility.
    used_peps_per_taxon: dict = defaultdict(set)

    total_marker_psms = 0.0
    for pep, families in peptide_to_families.items():
        p_idx = peptide_index.get(pep)
        if p_idx is None:
            continue
        y_p = float(spectral_counts.get(pep, 0))
        if y_p <= 0:
            continue
        r_row = responsibilities[p_idx]  # (T,)
        if not np.any(r_row > 0):
            continue
        contributions = y_p * r_row  # (T,)
        marker_signal += contributions
        marker_psm_count += contributions
        total_marker_psms += float(contributions.sum())
        for t in np.nonzero(r_row > 0)[0]:
            used_peps_per_taxon[int(t)].add(pep)
            for f in families:
                family_signal[f][int(t)] += float(contributions[t])

    # ------------------------------------------------------------------ step 4
    # Distinct families with non-trivial signal per taxon.  Threshold of
    # 0.5 fractional PSMs aligns with the marker_psm_count threshold —
    # families contributing less than half a PSM are below noise.
    marker_families_per_taxon = np.zeros(T, dtype=np.int64)
    family_signal_per_taxon: dict = defaultdict(dict)
    for f, t_sig in family_signal.items():
        for t, sig in t_sig.items():
            if sig > 0.5:
                marker_families_per_taxon[t] += 1
            family_signal_per_taxon[taxon_labels[t]][f] = sig

    for t, peps in used_peps_per_taxon.items():
        marker_peptides_per_taxon[t] = len(peps)

    # ------------------------------------------------------------------ step 5
    has_marker_estimate = (
        (marker_families_per_taxon >= min_marker_families)
        & (marker_psm_count >= min_marker_psms)
    )

    # ------------------------------------------------------------------ step 6
    # Compose c_t: marker-derived for qualifying taxa, pi-fallback for
    # the rest, then renormalise.
    cell_abundance = np.zeros(T, dtype=np.float64)
    if has_marker_estimate.any():
        sel_total = float(marker_signal[has_marker_estimate].sum())
        if sel_total > 0:
            cell_abundance[has_marker_estimate] = (
                marker_signal[has_marker_estimate] / sel_total
            )
        else:
            # All "qualifying" taxa actually have zero signal — pathological
            # combination of thresholds and zero numerator; fall back to pi.
            cell_abundance[has_marker_estimate] = pi[has_marker_estimate]
    cell_abundance[~has_marker_estimate] = pi[~has_marker_estimate]

    s = float(cell_abundance.sum())
    if s > 0:
        cell_abundance = cell_abundance / s
    else:
        # No qualifying taxa, no fallback signal — degenerate.  Keep pi
        # exactly to avoid handing back a NaN vector.
        cell_abundance = pi.copy()

    # ------------------------------------------------------------------ stats
    total_y = float(sum(spectral_counts.values()))
    fraction_psms_from_markers = (
        total_marker_psms / total_y if total_y > 0 else 0.0
    )

    return MarkerCorrectionResult(
        cell_abundance=cell_abundance,
        psm_abundance=pi.copy(),
        marker_signal=marker_signal,
        marker_families_per_taxon=marker_families_per_taxon,
        marker_psm_count=marker_psm_count,
        marker_peptides_per_taxon=marker_peptides_per_taxon,
        taxon_labels=list(taxon_labels),
        has_marker_estimate=has_marker_estimate,
        total_marker_psms=total_marker_psms,
        total_marker_peptides=n_marker_peptide_rows,
        fraction_psms_from_markers=fraction_psms_from_markers,
        family_signal_per_taxon=dict(family_signal_per_taxon),
    )


def log_marker_diagnostics(
    result: MarkerCorrectionResult,
    logger_obj: Optional[logging.Logger] = None,
    top_n: int = 25,
) -> str:
    """Format a human-readable diagnostic report for a correction result.

    Parameters
    ----------
    result : MarkerCorrectionResult
        Output of :func:`compute_cell_equivalent_abundance`.
    logger_obj : logging.Logger, optional
        If provided, every line of the report is also logged at INFO.
    top_n : int, default ``25``
        Number of taxa to show in the per-taxon table (sorted by
        cell-equivalent abundance descending).

    Returns
    -------
    str
        The full report as a single newline-joined string.
    """
    lines: list = []
    lines.append("=== Marker-based Cell-Equivalent Correction ===")
    lines.append(
        f"Total marker peptides used: {result.total_marker_peptides}  "
        f"|  Total marker PSMs (fractional): {result.total_marker_psms:.2f}"
    )
    lines.append(
        f"Fraction of all PSMs from markers: "
        f"{result.fraction_psms_from_markers * 100:.2f}%"
    )

    n_taxa = len(result.taxon_labels)
    n_with_marker = int(result.has_marker_estimate.sum())
    lines.append(
        f"Taxa with marker-based estimate: {n_with_marker} / {n_taxa} "
        f"(remaining {n_taxa - n_with_marker} fell back to PSM-level pi)"
    )

    order = np.argsort(-result.cell_abundance)
    lines.append("")
    lines.append(
        f"{'Taxon':<40} {'PSM-pi':>9} {'Cell-c':>9} {'Markers':>8} "
        f"{'MarkPSMs':>10} {'Source':>8}"
    )
    for rank, t in enumerate(order):
        if rank >= top_n:
            break
        lbl = result.taxon_labels[t]
        name = lbl.split("|", 1)[-1][:40]
        source = "marker" if result.has_marker_estimate[t] else "pi"
        lines.append(
            f"{name:<40} "
            f"{result.psm_abundance[t]:>9.4f} "
            f"{result.cell_abundance[t]:>9.4f} "
            f"{int(result.marker_families_per_taxon[t]):>8d} "
            f"{result.marker_psm_count[t]:>10.2f} "
            f"{source:>8}"
        )

    # Brief reasons for taxa that fell back.
    fallback = [
        t for t in range(n_taxa)
        if not result.has_marker_estimate[t] and result.psm_abundance[t] > 0
    ]
    if fallback:
        lines.append("")
        lines.append(
            f"PSM-pi fallback taxa (showing up to {min(10, len(fallback))}):"
        )
        for t in fallback[:10]:
            name = result.taxon_labels[t].split("|", 1)[-1]
            lines.append(
                f"  {name}: families={int(result.marker_families_per_taxon[t])}, "
                f"marker_psms={result.marker_psm_count[t]:.2f}, "
                f"pi={result.psm_abundance[t]:.4f}"
            )

    lines.append("=== END ===")
    report = "\n".join(lines)
    if logger_obj is not None:
        for ln in lines:
            logger_obj.info(ln)
    return report
