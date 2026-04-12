"""Parse pepXML files to extract peptide-level PSMs and spectral counts.

Uses :mod:`xml.etree.ElementTree` iterative parsing for memory efficiency
with large files.  Only rank-1 search hits are counted.  Decoy and
contaminant proteins are excluded based on configurable accession prefixes.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_EXCLUDE_PREFIXES = ["DECOY", "contag"]


def parse_pepxml(
    pepxml_path: str,
    exclude_prefixes: list[str] | None = None,
) -> tuple[dict[str, int], dict[str, set[str]]]:
    """Extract peptide spectral counts and protein mappings from a pepXML file.

    Parameters
    ----------
    pepxml_path : str
        Path to the validated pepXML file.
    exclude_prefixes : list of str, optional
        Protein-accession prefixes to exclude (default ``["DECOY", "contag"]``).

    Returns
    -------
    spectral_counts : dict[str, int]
        ``peptide_sequence -> PSM count``.
    peptide_protein_map : dict[str, set[str]]
        ``peptide_sequence -> {protein_accession, ...}`` (filtered).
    """
    path = Path(pepxml_path)
    if not path.exists():
        raise FileNotFoundError(f"pepXML file not found: {pepxml_path}")

    if exclude_prefixes is None:
        exclude_prefixes = list(_DEFAULT_EXCLUDE_PREFIXES)
    exclude_lower = [p.lower() for p in exclude_prefixes]

    spectral_counts: dict[str, int] = {}
    peptide_protein_map: dict[str, set[str]] = {}
    n_total = 0
    n_excluded = 0

    for event, elem in ET.iterparse(str(path), events=["end"]):
        tag = _local_tag(elem.tag)

        if tag == "search_hit":
            hit_rank = elem.get("hit_rank", "1")
            if hit_rank != "1":
                continue

            peptide = elem.get("peptide", "")
            protein = elem.get("protein", "")
            if not peptide:
                continue

            n_total += 1

            # Collect primary + alternative proteins.
            all_proteins: set[str] = set()
            if protein:
                all_proteins.add(protein)
            for child in elem:
                if _local_tag(child.tag) == "alternative_protein":
                    alt = child.get("protein", "")
                    if alt:
                        all_proteins.add(alt)

            # Apply exclusion filter.
            filtered = {
                p
                for p in all_proteins
                if not any(p.lower().startswith(pfx) for pfx in exclude_lower)
            }

            if not filtered:
                n_excluded += 1
                continue

            spectral_counts[peptide] = spectral_counts.get(peptide, 0) + 1
            peptide_protein_map.setdefault(peptide, set()).update(filtered)

        elif tag == "spectrum_query":
            # Free memory for the processed query subtree.
            elem.clear()

    logger.info(
        "Parsed pepXML: %d rank-1 PSMs, %d excluded (decoy/contaminant), "
        "%d unique peptides, %d unique proteins",
        n_total,
        n_excluded,
        len(spectral_counts),
        len({p for prots in peptide_protein_map.values() for p in prots}),
    )

    return spectral_counts, peptide_protein_map


def _local_tag(tag: str) -> str:
    """Strip XML namespace prefix from a tag name."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag
