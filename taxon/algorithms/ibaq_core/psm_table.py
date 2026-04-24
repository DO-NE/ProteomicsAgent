"""Build a PSM-level DataFrame for iBAQ methods from pepXML and FASTA data."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .fasta_utils import build_protein_species_map, count_tryptic_peptides, extract_accession

logger = logging.getLogger(__name__)


def build_psm_table(
    peptides: list[str],
    spectral_counts: dict[str, int],
    protein_species_map: dict[str, str],
    peptide_protein_map: dict[str, set[str]] | None = None,
    fasta_path: str | None = None,
    exclude_prefixes: list[str] | None = None,
    min_peptide_len: int = 7,
    max_peptide_len: int = 50,
    missed_cleavages: int = 2,
) -> pd.DataFrame:
    """Build a PSM table suitable for iBAQ quantification methods.

    Returns a DataFrame with columns ``[peptide, protein_acc, species, intensity]``.
    Each (peptide, protein) pair produces one row — a peptide mapping to
    multiple proteins is exploded into multiple rows, each carrying the full
    spectral count as intensity (matching the POC convention).

    Parameters
    ----------
    peptides : list of str
        Observed peptide sequences.
    spectral_counts : dict[str, int]
        Peptide -> PSM count.  Entries absent from *peptides* are ignored.
    protein_species_map : dict[str, str]
        Protein accession -> species name (from FASTA).
    peptide_protein_map : dict[str, set[str]] or None
        Peptide -> {protein accessions} from pepXML.  When ``None``, mappings
        are derived by in-silico digestion of the FASTA at *fasta_path*.
    fasta_path : str or None
        Path to protein FASTA — used only as fallback when
        *peptide_protein_map* is ``None``.
    exclude_prefixes, min_peptide_len, max_peptide_len, missed_cleavages
        Parameters forwarded to the FASTA digestion fallback.
    """
    if peptide_protein_map is None and fasta_path:
        peptide_protein_map = _digest_fasta_to_peptide_map(
            fasta_path,
            set(peptides),
            exclude_prefixes=exclude_prefixes,
            min_len=min_peptide_len,
            max_len=max_peptide_len,
            missed_cleavages=missed_cleavages,
        )

    if not peptide_protein_map:
        logger.warning("No peptide-protein mappings available; PSM table will be empty")
        return pd.DataFrame(columns=["peptide", "protein_acc", "species", "intensity"])

    rows: list[tuple[str, str, str, float]] = []
    unmapped_peptides = 0

    for pep in peptides:
        proteins = peptide_protein_map.get(pep)
        if not proteins:
            unmapped_peptides += 1
            continue
        intensity = float(spectral_counts.get(pep, 1))
        for prot in proteins:
            # Normalize accession so it matches theoretical-count
            # and protein-species-map keys (e.g. "tr|X|Y" → "X").
            normalized = extract_accession(prot) or prot
            species = protein_species_map.get(normalized)
            if not species:
                # Fallback: try the raw pepXML accession.
                species = protein_species_map.get(prot)
            if species:
                rows.append((pep, normalized, species, intensity))

    if unmapped_peptides:
        logger.info(
            "PSM table: %d peptides had no protein mapping and were skipped",
            unmapped_peptides,
        )

    df = pd.DataFrame(rows, columns=["peptide", "protein_acc", "species", "intensity"])
    logger.info(
        "PSM table built: %d rows, %d unique peptides, %d proteins, %d species",
        len(df),
        df["peptide"].nunique(),
        df["protein_acc"].nunique(),
        df["species"].nunique(),
    )
    return df


def _digest_fasta_to_peptide_map(
    fasta_path: str,
    observed_peptides: set[str],
    exclude_prefixes: list[str] | None = None,
    min_len: int = 7,
    max_len: int = 50,
    missed_cleavages: int = 2,
) -> dict[str, set[str]]:
    """Digest all FASTA proteins and map observed peptides to accessions."""
    from pathlib import Path
    import re

    path = Path(fasta_path)
    if not path.exists():
        return {}

    if exclude_prefixes is None:
        exclude_prefixes = ["DECOY", "contag"]
    exclude_lower = [p.lower() for p in exclude_prefixes]

    peptide_proteins: dict[str, set[str]] = {}
    cur_acc: str | None = None
    seq_parts: list[str] = []

    def _flush() -> None:
        if cur_acc is None:
            return
        seq = "".join(seq_parts)
        if not seq:
            return
        for pep in _trypsin_digest(seq, min_len, max_len, missed_cleavages):
            if pep in observed_peptides:
                peptide_proteins.setdefault(pep, set()).add(cur_acc)

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.startswith(">"):
                _flush()
                header = stripped[1:]
                m = re.search(r"[a-z]{2}\|([^|]+)\|", header, flags=re.IGNORECASE)
                acc = m.group(1) if m else header.split(None, 1)[0]
                if acc and not any(acc.lower().startswith(pfx) for pfx in exclude_lower):
                    cur_acc = acc
                else:
                    cur_acc = None
                seq_parts = []
            elif cur_acc is not None:
                seq_parts.append(stripped)
    _flush()

    logger.info(
        "FASTA digestion fallback: matched %d peptides to proteins",
        len(peptide_proteins),
    )
    return peptide_proteins


def _trypsin_digest(
    seq: str, min_len: int, max_len: int, missed_cleavages: int,
) -> list[str]:
    """In-silico trypsin digestion: cleave after K/R except before P."""
    sites = [0]
    for i in range(len(seq) - 1):
        if seq[i] in ("K", "R") and seq[i + 1] != "P":
            sites.append(i + 1)
    sites.append(len(seq))

    peptides: list[str] = []
    max_jump = max(int(missed_cleavages), 0) + 1
    for start in range(len(sites) - 1):
        for end in range(start + 1, min(start + 1 + max_jump, len(sites))):
            pep = seq[sites[start]:sites[end]]
            if min_len <= len(pep) <= max_len:
                peptides.append(pep)
    return peptides
