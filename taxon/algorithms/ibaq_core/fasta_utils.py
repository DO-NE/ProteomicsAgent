"""FASTA utilities for iBAQ: protein-to-species mapping and theoretical peptide counts."""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def build_protein_species_map(
    fasta_path: str,
    exclude_prefixes: list[str] | None = None,
) -> dict[str, str]:
    """Parse a FASTA file and return accession -> species name mapping.

    Species is extracted from the ``OS=`` field in UniProt-style headers.
    Falls back to bracket-delimited organism names (e.g. ``[Homo sapiens]``).

    Parameters
    ----------
    fasta_path : str
        Path to protein FASTA database.
    exclude_prefixes : list of str, optional
        Accession prefixes to skip (default ``["DECOY", "contag"]``).

    Returns
    -------
    dict[str, str]
        ``protein_accession -> species_name``.  Proteins whose species
        cannot be determined are omitted.
    """
    path = Path(fasta_path)
    if not path.exists():
        logger.warning("FASTA file not found: %s", fasta_path)
        return {}

    if exclude_prefixes is None:
        exclude_prefixes = ["DECOY", "contag"]
    exclude_lower = [p.lower() for p in exclude_prefixes]

    mapping: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if not line.startswith(">"):
                    continue
                header = line[1:].strip()
                accession = extract_accession(header)
                if not accession:
                    continue
                if any(accession.lower().startswith(pfx) for pfx in exclude_lower):
                    continue
                species = _extract_species(header)
                if species:
                    mapping[accession] = species
    except Exception:
        logger.exception("Failed to parse FASTA: %s", fasta_path)

    logger.info(
        "Protein-species map: %d proteins from %s",
        len(mapping),
        fasta_path,
    )
    return mapping


def build_theoretical_peptide_counts(
    fasta_path: str,
    min_len: int = 7,
    max_len: int = 30,
    missed_cleavages: int = 2,
    exclude_prefixes: list[str] | None = None,
) -> dict[str, int]:
    """Count theoretical tryptic peptides per protein from a FASTA file.

    Parameters
    ----------
    fasta_path : str
        Path to protein FASTA database.
    min_len, max_len : int
        Peptide length bounds (inclusive).
    missed_cleavages : int
        Maximum allowed missed cleavages.
    exclude_prefixes : list of str, optional
        Accession prefixes to skip.

    Returns
    -------
    dict[str, int]
        ``protein_accession -> theoretical_peptide_count``.
    """
    path = Path(fasta_path)
    if not path.exists():
        logger.warning("FASTA file not found: %s", fasta_path)
        return {}

    if exclude_prefixes is None:
        exclude_prefixes = ["DECOY", "contag"]
    exclude_lower = [p.lower() for p in exclude_prefixes]

    counts: dict[str, int] = {}
    cur_acc: str | None = None
    seq_parts: list[str] = []

    def _flush() -> None:
        if cur_acc is None:
            return
        seq = "".join(seq_parts)
        if seq:
            counts[cur_acc] = count_tryptic_peptides(
                seq, min_len=min_len, max_len=max_len,
                missed_cleavages=missed_cleavages,
            )

    try:
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    continue
                if stripped.startswith(">"):
                    _flush()
                    header = stripped[1:]
                    acc = extract_accession(header)
                    if acc and not any(acc.lower().startswith(pfx) for pfx in exclude_lower):
                        cur_acc = acc
                    else:
                        cur_acc = None
                    seq_parts = []
                elif cur_acc is not None:
                    seq_parts.append(stripped)
        _flush()
    except Exception:
        logger.exception("Failed to parse FASTA for theoretical counts: %s", fasta_path)

    logger.info(
        "Theoretical peptide counts: %d proteins from %s",
        len(counts),
        fasta_path,
    )
    return counts


def count_tryptic_peptides(
    seq: str,
    min_len: int = 7,
    max_len: int = 30,
    missed_cleavages: int = 2,
) -> int:
    """Count theoretical tryptic peptides for a protein sequence.

    Trypsin cleaves after K or R, except when followed by P.
    """
    if not seq:
        return 0

    sites = [0]
    for i in range(len(seq) - 1):
        if seq[i] in ("K", "R") and seq[i + 1] != "P":
            sites.append(i + 1)
    sites.append(len(seq))

    count = 0
    max_jump = max(int(missed_cleavages), 0) + 1
    for start in range(len(sites) - 1):
        for end in range(start + 1, min(start + 1 + max_jump, len(sites))):
            pep_len = sites[end] - sites[start]
            if min_len <= pep_len <= max_len:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def extract_accession(header: str) -> str | None:
    """Extract protein accession from a FASTA header line (without '>').

    For UniProt-style headers like ``sp|P12345|NAME`` or ``tr|Q9XXX|NAME``,
    returns the accession (e.g. ``P12345``).  Otherwise returns the first
    whitespace-delimited token.
    """
    text = header.strip()
    if not text:
        return None
    # UniProt style: sp|P12345|NAME or tr|Q9XXX|NAME
    m = re.search(r"[a-z]{2}\|([^|]+)\|", text, flags=re.IGNORECASE)
    if m:
        return m.group(1)
    # Fall back to first whitespace-delimited token.
    return text.split(None, 1)[0] or None


def _extract_species(header: str) -> str | None:
    """Extract species name from a FASTA header.

    Tries in order:
    1. UniProt ``OS=...`` field
    2. Bracket-delimited ``[Organism name]``
    """
    # OS= field (stops at the next XX= tag or end of line)
    m = re.search(r"\bOS=(.+?)(?:\s[A-Z]{2}=|$)", header)
    if m:
        name = _normalize_species(m.group(1))
        if name:
            return name

    # Quoted species= field (GeneMark / ATCC style)
    m = re.search(r'\bspecies="([^"]+)"', header)
    if m:
        name = _normalize_species(m.group(1))
        if name:
            return name

    # Bracket-delimited
    m = re.search(r"\[([^\]]+)\]", header)
    if m:
        name = _normalize_species(m.group(1))
        if name:
            return name

    return None


def _normalize_species(raw: str) -> str | None:
    """Normalize a species string to 'Genus species' form."""
    text = raw.strip()
    # Strip strain info in parentheses
    text = re.sub(r"\([^)]*\)", "", text).strip()
    tokens = text.split()
    if len(tokens) >= 2:
        return tokens[0].capitalize() + " " + tokens[1].lower()
    if len(tokens) == 1:
        return tokens[0].capitalize()
    return None
