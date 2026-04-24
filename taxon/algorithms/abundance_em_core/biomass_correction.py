"""Per-taxon biomass correction factors for PSM-based abundance estimation.

Converts PSM-level abundance into protein biomass-level abundance by dividing
out a taxon-specific factor ``g_t`` that represents the expected number of
PSMs generated per unit of protein biomass for taxon ``t``. A taxon with a
higher ``g_t`` over-produces PSMs relative to its biomass, so its PSM share
overstates its biomass share; scaling by ``1 / g_t`` corrects the bias.

``g_t`` decomposes multiplicatively:

- ``g_t^(length)`` — tryptic-peptide yield per amino acid. The number of
  in-silico tryptic peptides within the detectable length range divided by
  the total amino-acid count across the taxon's reference proteome. Taxa
  with higher K/R frequency or shorter average protein length produce more
  detectable peptides per unit proteome mass.

- ``g_t^(coverage)`` — proteome detection fraction. The fraction of the
  taxon's reference proteins that received at least ``min_psm_threshold``
  PSMs. Captures the "iceberg effect": a larger proteome has a smaller
  fraction above the MS detection limit and therefore produces fewer PSMs
  per unit biomass than its proteome size alone would predict.

The combined correction is ``g_t = g_t^(length) * g_t^(coverage)``.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np

from .accession_resolver import extract_uniprot_accession, resolve_accessions
from .mapping_matrix import (
    _digest,
    _extract_accession,
    _extract_prefix,
    _is_valid_taxon_name,
    _iter_fasta,
    _parse_header_detailed,
    _should_exclude,
    _slug,
)

logger = logging.getLogger(__name__)

# Floor used when a taxon has no proteins in the reference; prevents a zero
# correction factor from eliminating the taxon when its biomass-scaled share
# is later normalised.
_ZERO_PROTEINS_EPS = 1e-6

_NON_ALPHA_RE = re.compile(r"[^A-Za-z]")


def compute_biomass_corrections(
    fasta_path: str,
    taxon_labels: list,
    spectral_counts: dict,
    peptide_protein_map: dict,
    enzyme: str = "trypsin",
    missed_cleavages: int = 2,
    min_length: int = 7,
    max_length: int = 50,
    exclude_prefixes: Optional[list] = None,
    min_psm_threshold: int = 2,
    resolve_uniprot: bool = False,
) -> tuple:
    """Compute per-taxon biomass correction factors ``g_t``.

    Parameters
    ----------
    fasta_path : str
        Path to the reference FASTA database.
    taxon_labels : list of str
        ``"taxon_id|taxon_name"`` strings, in the same order produced by
        :func:`build_mapping_matrix`. The returned array aligns with this
        order.
    spectral_counts : dict[str, int]
        ``peptide_sequence -> PSM count``.
    peptide_protein_map : dict[str, set[str]]
        ``peptide_sequence -> {protein_accession, ...}`` (e.g. from pepXML
        parsing).
    enzyme, missed_cleavages, min_length, max_length : digestion parameters
        Match the values used when the mapping matrix was built so the
        detectable-peptide universe is consistent.
    exclude_prefixes : list of str or None
        FASTA header prefixes to drop (decoys, contaminants).
    min_psm_threshold : int
        A protein must receive at least this many (shared-share) PSMs to
        count as "detected" when computing ``g_t^(coverage)``.
    resolve_uniprot : bool
        Whether to resolve bare UniProt-accession headers via the REST API,
        mirroring the behaviour of :func:`build_mapping_matrix`.

    Returns
    -------
    corrections : np.ndarray of shape ``(T,)``
        Per-taxon ``g_t`` factors, aligned with ``taxon_labels``.
    diagnostics : dict
        Per-taxon detail keyed by label string. Each value is a dict with
        ``total_proteins``, ``total_amino_acids``,
        ``detectable_tryptic_peptides``, ``proteins_above_threshold``,
        ``g_length``, ``g_coverage``, and ``g_total``.
    """
    if not Path(fasta_path).exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    protein_taxon, taxon_sequences = _classify_proteins(
        fasta_path=fasta_path,
        exclude_prefixes=exclude_prefixes,
        resolve_uniprot=resolve_uniprot,
    )

    label_keys = [_label_to_key(lbl) for lbl in taxon_labels]
    T = len(taxon_labels)

    # --- Component 1: g_length -------------------------------------------
    total_proteins = np.zeros(T, dtype=np.int64)
    total_amino_acids = np.zeros(T, dtype=np.int64)
    detectable_peptides = np.zeros(T, dtype=np.int64)
    g_length = np.zeros(T, dtype=np.float64)

    for i, key in enumerate(label_keys):
        seqs = taxon_sequences.get(key, [])
        total_proteins[i] = len(seqs)
        aa_sum = 0
        peptide_set: set = set()
        for seq in seqs:
            clean = _NON_ALPHA_RE.sub("", seq)
            aa_sum += len(clean)
            for pep in _digest(
                seq,
                enzyme=enzyme,
                missed_cleavages=missed_cleavages,
                min_length=min_length,
                max_length=max_length,
            ):
                peptide_set.add(pep)
        total_amino_acids[i] = aa_sum
        detectable_peptides[i] = len(peptide_set)
        g_length[i] = (len(peptide_set) / aa_sum) if aa_sum > 0 else 0.0

    # --- Per-protein PSM counts ------------------------------------------
    # Share each peptide's PSM count equally across its mapped proteins.
    protein_psm: dict = {}
    for pep, psm in spectral_counts.items():
        proteins = peptide_protein_map.get(pep)
        if not proteins:
            continue
        share = float(psm) / len(proteins)
        for acc in proteins:
            protein_psm[acc] = protein_psm.get(acc, 0.0) + share

    # Reverse index: taxon_key -> list of protein accessions in the FASTA.
    taxon_accessions: dict = {}
    for acc, key in protein_taxon.items():
        taxon_accessions.setdefault(key, []).append(acc)

    # --- Component 2: g_coverage -----------------------------------------
    proteins_above_threshold = np.zeros(T, dtype=np.int64)
    g_coverage = np.zeros(T, dtype=np.float64)

    for i, key in enumerate(label_keys):
        accs = taxon_accessions.get(key, [])
        total = len(accs)
        above = sum(
            1 for acc in accs if protein_psm.get(acc, 0.0) >= min_psm_threshold
        )
        proteins_above_threshold[i] = above
        if total == 0:
            g_coverage[i] = _ZERO_PROTEINS_EPS
        elif above == 0:
            # At least one protein's worth of coverage to avoid zeroing g_t.
            g_coverage[i] = 1.0 / total
        else:
            g_coverage[i] = above / total

    corrections = g_length * g_coverage

    diagnostics: dict = {}
    for i, lbl in enumerate(taxon_labels):
        diagnostics[lbl] = {
            "total_proteins": int(total_proteins[i]),
            "total_amino_acids": int(total_amino_acids[i]),
            "detectable_tryptic_peptides": int(detectable_peptides[i]),
            "proteins_above_threshold": int(proteins_above_threshold[i]),
            "g_length": float(g_length[i]),
            "g_coverage": float(g_coverage[i]),
            "g_total": float(corrections[i]),
        }

    logger.info(
        "Computed biomass corrections for %d taxa (min_psm_threshold=%d)",
        T,
        min_psm_threshold,
    )

    return corrections, diagnostics


def log_biomass_diagnostics(diagnostics: dict, logger=None) -> None:
    """Log a formatted table of per-taxon biomass correction factors.

    Rows are sorted by ``g_total`` descending so the highest-correction
    (most over-represented at the PSM level) taxa appear first.
    """
    log = logger if logger is not None else logging.getLogger(__name__)
    if not diagnostics:
        log.info("No biomass diagnostics to log.")
        return

    rows = sorted(
        diagnostics.items(),
        key=lambda kv: kv[1].get("g_total", 0.0),
        reverse=True,
    )

    log.info("=== BIOMASS CORRECTION DIAGNOSTICS ===")
    header = (
        f"{'Taxon':<40} {'Proteins':>10} {'TotalAA':>12} "
        f"{'Peptides':>12} {'AboveThr':>10} "
        f"{'g_length':>12} {'g_coverage':>12} {'g_total':>12}"
    )
    log.info(header)
    for label, d in rows:
        name = label.split("|", 1)[-1]
        log.info(
            "%-40s %10d %12d %12d %10d %12.3e %12.3e %12.3e",
            name[:40],
            d["total_proteins"],
            d["total_amino_acids"],
            d["detectable_tryptic_peptides"],
            d["proteins_above_threshold"],
            d["g_length"],
            d["g_coverage"],
            d["g_total"],
        )
    log.info("=== END BIOMASS CORRECTION DIAGNOSTICS ===")


# --------------------------------------------------------------------- helpers


def _label_to_key(label: str) -> tuple:
    """Split a ``"taxon_id|taxon_name"`` label into its two components."""
    if "|" in label:
        tid, tname = label.split("|", 1)
        return (tid, tname)
    return ("0", label)


def _classify_proteins(
    fasta_path: str,
    exclude_prefixes: Optional[list] = None,
    resolve_uniprot: bool = False,
) -> tuple:
    """Classify every FASTA entry to a ``(taxon_id, taxon_name)`` key.

    Mirrors the classification pipeline used by
    :func:`abundance_em_core.mapping_matrix.build_mapping_matrix`:

    1. Header parsing via :func:`_parse_header_detailed`.
    2. Optional UniProt REST resolution for bare-accession headers.
    3. Prefix-cohort inference for residually unclassified entries.
    4. Species-level deduplication (strain variants collapsed onto a
       canonical slug key).

    The user-supplied ``prefix_map_file`` rescue step is intentionally
    omitted here; callers that rely on it should be aware that a small
    subset of proteins may be left unclassified and therefore excluded
    from the biomass calculation.

    Returns
    -------
    protein_taxon : dict[str, tuple[str, str]]
        Accession -> taxon key. Unclassified entries are excluded.
    taxon_sequences : dict[tuple[str, str], list[str]]
        Per-taxon list of raw protein sequences (classified entries only).
    """
    fasta = Path(fasta_path)

    parsed_records: list = []
    known_acc_organism: dict = {}
    organism_to_taxid: dict = {}
    unresolved_uniprot: set = set()

    for header, seq in _iter_fasta(fasta):
        if _should_exclude(header, exclude_prefixes):
            continue
        accession = _extract_accession(header)
        uniprot_acc = extract_uniprot_accession(accession)
        key, _rejected = _parse_header_detailed(header)
        if key == ("0", "unclassified"):
            if resolve_uniprot and uniprot_acc:
                unresolved_uniprot.add(uniprot_acc)
        else:
            if uniprot_acc and uniprot_acc not in known_acc_organism:
                known_acc_organism[uniprot_acc] = key[1]
            existing = organism_to_taxid.get(key[1])
            if existing is None or (not existing.isdigit() and key[0].isdigit()):
                organism_to_taxid[key[1]] = key[0]
        parsed_records.append((accession, uniprot_acc, key, seq))

    acc_to_organism: dict = {}
    if resolve_uniprot and unresolved_uniprot:
        acc_to_organism = resolve_accessions(
            fasta_path=fasta_path,
            unresolved_accessions=unresolved_uniprot,
            known_accession_organism=known_acc_organism,
            use_api=True,
        )

    resolved_parsed: list = []
    for accession, uniprot_acc, key, seq in parsed_records:
        if key == ("0", "unclassified") and uniprot_acc:
            organism = acc_to_organism.get(uniprot_acc)
            if organism and _is_valid_taxon_name(organism):
                taxid = organism_to_taxid.get(organism) or _slug(organism)
                key = (taxid, organism)
        resolved_parsed.append((accession, uniprot_acc, key, seq))

    # Prefix-cohort inference.
    prefix_totals: dict = {}
    prefix_votes: dict = {}
    for accession, _uacc, key, _seq in resolved_parsed:
        prefix = _extract_prefix(accession)
        if not prefix:
            continue
        prefix_totals[prefix] = prefix_totals.get(prefix, 0) + 1
        if key != ("0", "unclassified"):
            votes = prefix_votes.setdefault(prefix, {})
            votes[key] = votes.get(key, 0) + 1

    inferred_prefix_map: dict = {}
    for prefix, votes in prefix_votes.items():
        total = prefix_totals.get(prefix, 0)
        if total < 2:
            continue
        best_key, best_n = max(votes.items(), key=lambda kv: kv[1])
        if total > 0 and best_n / total >= 0.5 and _is_valid_taxon_name(best_key[1]):
            inferred_prefix_map[prefix] = best_key

    final_records: list = []
    for accession, uniprot_acc, key, seq in resolved_parsed:
        if key == ("0", "unclassified"):
            prefix = _extract_prefix(accession)
            rescued = inferred_prefix_map.get(prefix) if prefix else None
            if rescued is not None:
                key = rescued
        final_records.append((accession, uniprot_acc, key, seq))

    # Species-level deduplication (collapse strain-level variants).
    name_to_keys: dict = {}
    for _acc, _uacc, key, _seq in final_records:
        if key != ("0", "unclassified"):
            name_to_keys.setdefault(key[1], set()).add(key)

    key_remap: dict = {}
    for tname, keys in name_to_keys.items():
        if len(keys) > 1:
            canonical = (_slug(tname), tname)
            for k in keys:
                if k != canonical:
                    key_remap[k] = canonical

    protein_taxon: dict = {}
    taxon_sequences: dict = {}
    for accession, _uacc, key, seq in final_records:
        key = key_remap.get(key, key)
        if key == ("0", "unclassified"):
            continue
        protein_taxon[accession] = key
        taxon_sequences.setdefault(key, []).append(seq)

    return protein_taxon, taxon_sequences
