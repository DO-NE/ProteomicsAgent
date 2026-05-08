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
from pathlib import Path
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
    taxon_kingdom: Optional[dict] = None,
    exclude_kingdoms: frozenset = frozenset({"Eukaryota"}),
    emit_marker_peptide_table: bool = True,
    marker_peptide_table_path: Optional[str] = None,
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
        ``protein_accession -> (taxon_label, families, evalue, score)``
        where *families* is either a ``list[str]`` (current format, from
        :func:`taxon.algorithms.abundance_em_core.hmm_marker_search.run_hmmsearch`)
        or a plain ``str`` (legacy / hand-built test data — handled
        transparently).  The ``taxon_label`` field is informational;
        cross-referencing against :paramref:`taxon_protein_peptides` is
        what actually drives the calculation.
    taxon_kingdom : dict[str, str] or None, optional
        ``taxon_label -> kingdom`` where kingdom is one of
        ``"Bacteria"``, ``"Archaea"``, ``"Eukaryota"``, ``"Virus"``.
        When provided, taxa whose kingdom is in *exclude_kingdoms* are
        excluded from marker signal accumulation and always receive
        ``has_marker_estimate=False``.  When ``None`` (default), all
        taxa are eligible — backwards-compatible behaviour.
    exclude_kingdoms : frozenset, default ``frozenset({"Eukaryota"})``
        Set of kingdom strings to exclude from the cell-equivalent
        estimate.  Eukaryotes are excluded by default because bac120/ar53
        HMM profiles target prokaryotes and eukaryotic ribosomal-protein
        homologs would otherwise pass the E-value filter.
    taxon_protein_peptides : dict
        ``taxon_label -> {protein_accession -> [observed peptides]}``.
        Provided by :class:`MappingMatrixResult`.
    min_marker_families : int, default ``3``
        Minimum number of distinct marker families with nontrivial signal
        (per-taxon family signal > 0.5) required for a taxon to receive a
        marker-based estimate.
    min_marker_psms : float, default ``1.0``
        Minimum total fractional marker PSM count for a taxon.
    emit_marker_peptide_table : bool, default ``True``
        When *True* and *marker_peptide_table_path* is set, dump a
        per-(taxon, marker_family, marker_protein_accession,
        peptide_sequence) diagnostic TSV to the given path.  Pure
        diagnostic — does not affect any returned value.  A failure to
        write the file is logged at WARNING and never raises.
    marker_peptide_table_path : str or path-like, optional
        Output path for the per-marker-peptide TSV.  When *None* (the
        default), no file is written even if *emit_marker_peptide_table*
        is *True* — this keeps the function suitable for unit tests and
        in-memory callers that have no output directory.

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

    # Determine which taxon indices to exclude from marker signal.
    # Eukaryotes (and any other kingdoms in exclude_kingdoms) should never
    # receive a marker-based estimate: bac120/ar53 profiles target
    # prokaryotes, and eukaryotic ribosomal-protein homologs can pass the
    # E-value filter at 1e-10.
    excluded_taxa: set = set()
    if taxon_kingdom is not None:
        for lbl, idx in label_to_idx.items():
            kingdom = taxon_kingdom.get(lbl)
            if kingdom is not None and kingdom in exclude_kingdoms:
                excluded_taxa.add(idx)
        if excluded_taxa:
            logger.info(
                "Marker correction: excluding %d taxon/taxa from kingdom(s) %s",
                len(excluded_taxa), sorted(exclude_kingdoms),
            )

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
            _hmm_taxon_label, families_raw, _evalue, _score = payload
        except (TypeError, ValueError):
            logger.debug(
                "skipping malformed marker_proteins entry for %r: %r",
                accession, payload,
            )
            continue
        # families_raw is a list[str] in current format; plain str in old
        # cache / hand-built test data — handle both transparently.
        families_list = families_raw if isinstance(families_raw, list) else [families_raw]
        # Find this accession in any taxon bucket.  Marker proteins not
        # represented in taxon_protein_peptides (e.g. unclassified
        # proteins, or ones whose digest produced no observed peptides)
        # are silently skipped — they cannot contribute signal.
        # Excluded-kingdom taxa are also skipped so their marker hits
        # cannot inflate or contaminate the prokaryotic signal.
        appears_in = []
        for taxon_label, prot_map in taxon_protein_peptides.items():
            if label_to_idx.get(taxon_label) in excluded_taxa:
                continue
            peps = prot_map.get(accession)
            if peps:
                appears_in.append((taxon_label, peps))
        if not appears_in:
            continue
        n_marker_proteins_seen += 1
        for taxon_label, peps in appears_in:
            for pep in peps:
                pep_upper = pep.upper()
                for family in families_list:
                    peptide_to_families[pep_upper].add(family)
                peptide_to_taxa[pep_upper].add(taxon_label)

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
        p_idx = peptide_index.get(pep.upper())
        if p_idx is None:
            continue
        y_p = float(spectral_counts.get(pep.upper(), 0))
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
    # Override: excluded-kingdom taxa (e.g. Eukaryota) never receive a
    # marker-based estimate, regardless of signal strength.
    for t in excluded_taxa:
        has_marker_estimate[t] = False

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

    # ------------------------------------------------------------------ diagnostic dump
    # Per-(taxon, family, protein, peptide) TSV.  Pure diagnostic — runs
    # in a separate pass after the algorithmic accumulators are finalised
    # and is wrapped in try/except so a write error can never affect the
    # values returned to the caller.
    if emit_marker_peptide_table and marker_peptide_table_path is not None:
        try:
            n_rows = _emit_marker_peptide_table(
                output_path=Path(str(marker_peptide_table_path)),
                marker_proteins=marker_proteins,
                taxon_protein_peptides=taxon_protein_peptides,
                taxon_labels=list(taxon_labels),
                label_to_idx=label_to_idx,
                excluded_taxa=excluded_taxa,
                spectral_counts=spectral_counts,
                responsibilities=responsibilities,
                mapping_matrix=mapping_matrix,
                peptide_index=peptide_index,
                marker_psm_count=marker_psm_count,
            )
            logger.debug(
                "marker_peptides.tsv emitted with %d rows", n_rows,
            )
        except Exception as exc:  # noqa: BLE001 — diagnostic must never break the pipeline
            logger.warning("Failed to write marker_peptides.tsv: %s", exc)

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


def _emit_marker_peptide_table(
    output_path: Path,
    marker_proteins: dict,
    taxon_protein_peptides: dict,
    taxon_labels: list,
    label_to_idx: dict,
    excluded_taxa: set,
    spectral_counts: dict,
    responsibilities: np.ndarray,
    mapping_matrix: np.ndarray,
    peptide_index: dict,
    marker_psm_count: np.ndarray,
) -> int:
    """Dump per-(taxon, family, protein, peptide) marker diagnostic TSV.

    Side-effecting helper called from :func:`compute_cell_equivalent_abundance`
    after its main accumulation completes.  Performs **no** mutation of any
    argument and produces no algorithmic side effects — calling it has the
    same effect on EM-derived outputs as not calling it.

    Parameters
    ----------
    output_path : Path
        Destination TSV.  Parent directory is created if missing.
    marker_proteins, taxon_protein_peptides, taxon_labels, label_to_idx,
    excluded_taxa, spectral_counts, responsibilities, mapping_matrix,
    peptide_index :
        Same objects already in scope inside
        :func:`compute_cell_equivalent_abundance`.  See that function's
        parameter documentation for descriptions.
    marker_psm_count : np.ndarray, shape ``(T,)``
        Per-taxon ``Σ_p y_p · r_{pt}`` over marker peptides as computed
        by the main loop — used solely for the post-write sanity check.

    Returns
    -------
    int
        Number of data rows written (excluding header).

    Notes
    -----
    Granularity is one row per (consumer_taxon, marker_family,
    marker_protein, peptide).  The *consumer* taxon is any taxon that
    receives EM responsibility for the marker peptide *or* owns the
    marker protein in its FASTA digest — the union ensures both
    consistency with ``marker_psm_count`` (which sums ``y_p · r_pt`` over
    every consumer taxon, not just the owners) and that "potentially
    observable but unobserved" marker peptides still appear at least
    once with their owner taxon.

    The ``marker_protein_accession`` and ``marker_protein_hmm_evalue``
    columns refer to the FASTA hit identified by hmmsearch — i.e., the
    *origin* of the marker assignment.  For consumer taxa that do not
    own the marker protein (shared marker peptide allocated by the EM)
    the accession still points at the original marker-hit protein, since
    that is what classified the peptide as a marker in the first place.

    A peptide that appears in multiple (family, protein) origins for the
    same consumer taxon yields one row per origin, all carrying the same
    ``r_pt`` and ``weighted_psm_contribution``.  The sanity check at the
    end deduplicates by (consumer_taxon, peptide) before summing so a
    naive sum-over-rows can over-count and is documented accordingly.
    """
    A = np.asarray(mapping_matrix)
    if A.ndim == 2 and A.size > 0:
        row_sums = A.sum(axis=1).astype(np.int64)
    else:
        row_sums = np.zeros(0, dtype=np.int64)

    R = np.asarray(responsibilities, dtype=np.float64)
    T = len(taxon_labels)

    # ------------------------------------------------------------------ step 1
    # Build peptide -> list of (accession, family, evalue, owner_taxon_label).
    # Mirrors the iteration in compute_cell_equivalent_abundance step 1, but
    # preserves the (accession, family, owner) provenance per peptide.
    peptide_origins: dict = defaultdict(list)
    for accession, payload in marker_proteins.items():
        try:
            _hmm_taxon_label, families_raw, evalue, _score = payload
        except (TypeError, ValueError):
            continue
        families_list = (
            families_raw if isinstance(families_raw, list) else [families_raw]
        )
        try:
            evalue_f = float(evalue)
        except (TypeError, ValueError):
            evalue_f = float("nan")

        for taxon_label, prot_map in taxon_protein_peptides.items():
            if label_to_idx.get(taxon_label) in excluded_taxa:
                continue
            peps = prot_map.get(accession)
            if not peps:
                continue
            for pep in peps:
                pep_upper = pep.upper()
                for fam in families_list:
                    peptide_origins[pep_upper].append(
                        (accession, str(fam), evalue_f, taxon_label)
                    )

    # ------------------------------------------------------------------ step 2
    # Expand each marker peptide to the union of consumer taxa: any taxon
    # with non-zero responsibility, plus the owner taxa from the origins.
    # The owner-set ensures unobserved marker peptides (y_p == 0, r_row = 0)
    # still appear at least once.
    rows: list = []
    # (consumer_t_idx, pep_upper) -> y_p * r_pt, for the sanity check.
    dedup_contrib: dict = {}

    for pep_upper, origins in peptide_origins.items():
        p_idx = peptide_index.get(pep_upper)
        if (
            p_idx is not None
            and 0 <= p_idx < R.shape[0]
        ):
            r_row = R[p_idx]
        else:
            r_row = np.zeros(T, dtype=np.float64)
        if (
            p_idx is not None
            and 0 <= p_idx < row_sums.shape[0]
        ):
            n_sharing = int(row_sums[p_idx])
        else:
            n_sharing = 0
        is_unique = (n_sharing == 1)
        y_p = int(spectral_counts.get(pep_upper, 0))

        owner_idxs = {
            label_to_idx[lbl]
            for _a, _f, _e, lbl in origins
            if lbl in label_to_idx and label_to_idx[lbl] not in excluded_taxa
        }
        nonzero_idxs = {
            int(i) for i in np.nonzero(r_row > 0)[0]
            if int(i) not in excluded_taxa and int(i) < T
        }
        consumer_idxs = owner_idxs | nonzero_idxs
        if not consumer_idxs:
            continue

        for t_idx in consumer_idxs:
            r_pt = float(r_row[t_idx]) if t_idx < r_row.shape[0] else 0.0
            weighted = float(y_p) * r_pt
            dedup_contrib[(t_idx, pep_upper)] = weighted
            tlbl = taxon_labels[t_idx]
            tid, tname = (
                tlbl.split("|", 1) if "|" in tlbl else ("0", tlbl)
            )
            for accession, fam, evalue_f, _owner in origins:
                rows.append((
                    tid, tname, fam, accession, evalue_f,
                    pep_upper, len(pep_upper), is_unique, n_sharing,
                    y_p, r_pt, weighted,
                ))

    # Deterministic order: taxon_id, marker_family, accession, peptide.
    rows.sort(key=lambda r: (r[0], r[2], r[3], r[5]))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fh:
        fh.write(
            "taxon_id\ttaxon_name\tmarker_family\tmarker_protein_accession\t"
            "marker_protein_hmm_evalue\tpeptide_sequence\tpeptide_length\t"
            "is_unique_peptide\tn_taxa_sharing\ty_p\tr_pt\t"
            "weighted_psm_contribution\n"
        )
        for r in rows:
            tid, tname, fam, acc, ev, pep, plen, uniq, ns, yp, rpt, w = r
            ev_str = f"{ev:.6e}" if not np.isnan(ev) else "nan"
            fh.write(
                f"{tid}\t{tname}\t{fam}\t{acc}\t"
                f"{ev_str}\t{pep}\t{plen}\t"
                f"{bool(uniq)}\t{ns}\t{yp}\t{rpt:.6f}\t{w:.6f}\n"
            )

    # Sanity check: per-taxon dedup-summed weighted_psm_contribution must
    # match marker_psm_count within 1e-6.  Disagreement here indicates a
    # regression in this dump function (not in the algorithm).
    psm_per_taxon = np.zeros(T, dtype=np.float64)
    for (t_idx, _pep), contrib in dedup_contrib.items():
        psm_per_taxon[t_idx] += contrib

    target = np.asarray(marker_psm_count, dtype=np.float64)
    if target.shape[0] != T:
        # Defensive — should never happen, but avoids an obscure traceback
        # if a future caller passes a misshapen vector.
        logger.warning(
            "marker_peptides.tsv consistency check skipped: "
            "marker_psm_count shape %s does not match T=%d",
            target.shape, T,
        )
        return len(rows)

    deltas = np.abs(psm_per_taxon - target)
    max_delta = float(deltas.max()) if deltas.size else 0.0
    if max_delta < 1e-6:
        logger.info(
            "marker_peptides.tsv contains %d rows; sum of "
            "weighted_psm_contribution per taxon matches "
            "abundance_results.tsv marker_psms within 1e-6 "
            "(max delta %.3e)",
            len(rows), max_delta,
        )
    else:
        worst_t = int(deltas.argmax())
        logger.warning(
            "marker_peptides.tsv consistency check FAILED: max per-taxon "
            "delta %.6e at taxon %s (dump=%.6f vs marker_psm_count=%.6f). "
            "Total rows: %d. This indicates a regression in the dump "
            "logic, not in the marker correction algorithm.",
            max_delta, taxon_labels[worst_t],
            float(psm_per_taxon[worst_t]), float(target[worst_t]),
            len(rows),
        )

    return len(rows)


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
