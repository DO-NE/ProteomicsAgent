"""Build the peptide-to-taxon mapping matrix from a FASTA database.

Performs an in-silico digestion of every protein in the database, intersects
the resulting peptide set with an observed peptide list, and assembles the
binary mapping matrix used by :class:`AbundanceEM`.

Uses :mod:`pyteomics.parser` for digestion when available; falls back to a
self-contained trypsin implementation otherwise.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from scipy.sparse import csc_matrix

logger = logging.getLogger(__name__)

try:
    from pyteomics import parser as _pyteomics_parser  # type: ignore
    _HAVE_PYTEOMICS = True
except Exception:  # pragma: no cover - exercised only when pyteomics absent
    _pyteomics_parser = None
    _HAVE_PYTEOMICS = False

# Header parsing patterns -----------------------------------------------------
_OX_RE = re.compile(r"\bOX=(\d+)")
_OS_RE = re.compile(r"\bOS=(.+?)(?=\s+(?:OX|GN|PE|SV)=|$)")
_BRACKET_RE = re.compile(r"\[([^\[\]]+)\]")
_GENERIC_TAXID_RE = re.compile(r"\btaxon[_:]?(\d+)\b", re.IGNORECASE)


def build_mapping_matrix(
    peptides: list,
    fasta_path: str,
    enzyme: str = "trypsin",
    missed_cleavages: int = 2,
    min_length: int = 7,
    max_length: int = 50,
) -> tuple:
    """Build a peptide-taxon mapping matrix from a FASTA database.

    Parameters
    ----------
    peptides : list of str
        Observed peptide sequences (e.g., from PSM results). Order is
        preserved on the row axis of the returned matrix.
    fasta_path : str
        Path to the protein FASTA database.
    enzyme : str, optional
        Digestion enzyme. Currently only ``"trypsin"`` is implemented in the
        fallback path; if :mod:`pyteomics` is available, any enzyme name it
        recognizes is supported (default ``"trypsin"``).
    missed_cleavages : int, optional
        Number of allowed missed cleavages (default ``2``).
    min_length : int, optional
        Minimum peptide length to keep after digestion (default ``7``).
    max_length : int, optional
        Maximum peptide length to keep after digestion (default ``50``).

    Returns
    -------
    A : np.ndarray, shape ``(P, T)``
        Dense ``int8`` mapping matrix. Internally a sparse representation is
        used during construction; conversion to dense happens at return time.
    peptide_list : list of str
        Row labels matching the original ``peptides`` order, with duplicates
        removed (first occurrence wins). Peptides not present in any digested
        protein still appear in the output as all-zero rows so the row order
        matches the caller's input expectation.
    taxon_list : list of str
        Column labels in the form ``"<taxon_id>|<taxon_name>"``. Taxa with
        zero matched peptides are dropped from the column dimension.

    Notes
    -----
    The taxon column labels encode both the identifier and the readable
    organism string so the plugin wrapper can split them back out without
    re-parsing FASTA headers. Taxon identity is extracted from the FASTA
    headers using a series of patterns:

    - UniProt: ``>sp|P12345|... OS=Organism OX=12345 ...``
    - NCBI bracketed: ``>... [Organism Name]``
    - Inline taxid: ``>... taxon_12345 ...``
    - Generic ``OS=`` field on its own
    - Pipe-prefixed organism: ``>OrganismName|accession|description``

    For maximum portability, when no organism string can be extracted, the
    sequence is bucketed under taxon name ``"unknown"`` with id ``"0"``.
    """
    fasta = Path(fasta_path)
    if not fasta.exists():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")

    # Deduplicate observed peptides while preserving first-seen order.
    seen_pep: dict = {}
    peptide_list: list = []
    for pep in peptides:
        if pep and pep not in seen_pep:
            seen_pep[pep] = len(peptide_list)
            peptide_list.append(pep)
    P = len(peptide_list)
    pep_index = seen_pep  # alias for readability

    # Iterate proteins, group sequences by taxon, then digest. Grouping by
    # taxon first lets us digest each taxon's pooled proteome in one pass.
    # taxon_key -> (taxon_id, taxon_name, list_of_sequences)
    taxon_buckets: dict = {}
    for header, seq in _iter_fasta(fasta):
        taxon_id, taxon_name = _parse_header(header)
        key = (taxon_id, taxon_name)
        if key not in taxon_buckets:
            taxon_buckets[key] = []
        taxon_buckets[key].append(seq)

    if not taxon_buckets:
        logger.warning("FASTA file %s yielded no parseable records", fasta_path)
        empty = np.zeros((P, 0), dtype=np.int8)
        return empty, peptide_list, []

    logger.info(
        "Digesting %d taxa from %s (enzyme=%s, missed_cleavages=%d)",
        len(taxon_buckets),
        fasta_path,
        enzyme,
        missed_cleavages,
    )

    # Sparse construction: collect (row, col) hits as we digest.
    rows: list = []
    cols: list = []
    taxon_list: list = []
    taxon_keys_ordered = sorted(taxon_buckets.keys())

    target_pepset = set(peptide_list)  # for O(1) membership tests
    if not target_pepset:
        empty = np.zeros((0, 0), dtype=np.int8)
        return empty, peptide_list, []

    surviving_columns: list = []
    next_col = 0
    for taxon_id, taxon_name in taxon_keys_ordered:
        seqs = taxon_buckets[(taxon_id, taxon_name)]
        hits: set = set()
        for seq in seqs:
            for pep in _digest(
                seq,
                enzyme=enzyme,
                missed_cleavages=missed_cleavages,
                min_length=min_length,
                max_length=max_length,
            ):
                if pep in target_pepset:
                    hits.add(pep)
        if not hits:
            continue
        for pep in hits:
            rows.append(pep_index[pep])
            cols.append(next_col)
        surviving_columns.append((taxon_id, taxon_name))
        next_col += 1

    T = next_col
    if T == 0:
        logger.warning(
            "No peptides matched any taxon in %s; returning empty matrix",
            fasta_path,
        )
        empty = np.zeros((P, 0), dtype=np.int8)
        return empty, peptide_list, []

    data = np.ones(len(rows), dtype=np.int8)
    sparse = csc_matrix((data, (rows, cols)), shape=(P, T), dtype=np.int8)
    # Densify in one shot. The trade-off threshold from the spec
    # ("use scipy.sparse internally when P*T > 100,000") refers to keeping
    # the construction sparse, which we do; the returned matrix matches the
    # type the model expects (numpy array).
    A = sparse.toarray()

    # Encode column labels as "id|name" for the plugin wrapper.
    taxon_labels = [f"{tid}|{tname}" for tid, tname in surviving_columns]

    n_matched = int((A.sum(axis=1) > 0).sum())
    logger.info(
        "Built %d x %d mapping matrix (%d / %d observed peptides matched at least one taxon)",
        P,
        T,
        n_matched,
        P,
    )

    return A, peptide_list, taxon_labels


# --------------------------------------------------------------------- helpers


def _iter_fasta(path: Path) -> Iterator[tuple]:
    """Yield ``(header, sequence)`` pairs from a FASTA file.

    The reader is encoding-tolerant and ignores blank lines and comment
    lines beginning with ``;``.
    """
    header: Optional[str] = None
    seq_lines: list = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line or line.startswith(";"):
                continue
            if line.startswith(">"):
                if header is not None:
                    yield header, "".join(seq_lines)
                header = line[1:]
                seq_lines = []
            else:
                seq_lines.append(line.strip())
        if header is not None:
            yield header, "".join(seq_lines)


def _parse_header(header: str) -> tuple:
    """Extract a ``(taxon_id, taxon_name)`` pair from a FASTA header.

    Tries the following parsers in order:
    1. UniProt-style ``OS=`` / ``OX=`` fields.
    2. NCBI-style ``[Organism Name]`` brackets.
    3. Inline ``taxon_<id>`` markers.
    4. Pipe-prefixed organism (``>OrganismName|accession|...``).
    """
    if not header:
        return ("0", "unknown")

    # 1. UniProt OS / OX
    ox = _OX_RE.search(header)
    os_match = _OS_RE.search(header)
    if os_match:
        name = os_match.group(1).strip()
        taxon_id = ox.group(1) if ox else _slug(name)
        return (taxon_id, name)

    # 2. NCBI bracketed organism
    bracket = _BRACKET_RE.search(header)
    if bracket:
        name = bracket.group(1).strip()
        taxid = ox.group(1) if ox else _slug(name)
        return (taxid, name)

    # 3. Inline taxon_<id>
    inline = _GENERIC_TAXID_RE.search(header)
    if inline:
        return (inline.group(1), f"taxon_{inline.group(1)}")

    # 4. Pipe-prefixed organism: >OrganismName|accession|description
    if "|" in header:
        first = header.split("|", 1)[0].strip()
        # Heuristic: treat as an organism only if it has at least one alpha
        # char and is not a common UniProt namespace marker.
        if first and first.lower() not in {"sp", "tr", "gi", "ref", "gb"} and re.search(r"[A-Za-z]", first):
            return (_slug(first), first)

    return ("0", "unknown")


def _slug(name: str) -> str:
    """Make a deterministic id-like slug from a name when no real id exists."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return cleaned or "unknown"


# --- digestion ---------------------------------------------------------------


def _digest(
    sequence: str,
    enzyme: str,
    missed_cleavages: int,
    min_length: int,
    max_length: int,
) -> Iterator[str]:
    """Yield peptides produced by an in-silico digestion of ``sequence``.

    Uses :mod:`pyteomics.parser.cleave` if available, otherwise the
    self-contained trypsin path below. Sequences are upper-cased and
    stripped of any non-alphabetic characters before digestion.
    """
    if not sequence:
        return
    seq = re.sub(r"[^A-Za-z]", "", sequence).upper()
    if not seq:
        return

    if _HAVE_PYTEOMICS and enzyme.lower() == "trypsin":
        # pyteomics ships a "trypsin" rule string out of the box.
        peptides = _pyteomics_parser.cleave(  # type: ignore[attr-defined]
            seq,
            "trypsin",
            missed_cleavages=missed_cleavages,
            min_length=min_length,
        )
        for pep in peptides:
            if min_length <= len(pep) <= max_length:
                yield pep
        return

    if enzyme.lower() != "trypsin":
        raise NotImplementedError(
            f"Enzyme {enzyme!r} requires pyteomics; only 'trypsin' is "
            "supported in the fallback path."
        )
    yield from _trypsin_cleave(seq, missed_cleavages, min_length, max_length)


def _trypsin_cleave(
    sequence: str,
    missed_cleavages: int,
    min_length: int,
    max_length: int,
) -> Iterator[str]:
    """Pure-python trypsin digestion ('cuts after K/R, not before P').

    Generates all peptides obtainable with up to ``missed_cleavages`` skipped
    cleavage sites and emits those whose length lies in
    ``[min_length, max_length]``.
    """
    n = len(sequence)
    if n == 0:
        return

    # Cut points: indices i (1..n-1) where sequence[i-1] in {K,R} and
    # sequence[i] != P. The peptide ends just before each cut point and the
    # final peptide ends at n.
    cut_sites: list = [0]
    for i in range(1, n):
        if sequence[i - 1] in ("K", "R") and sequence[i] != "P":
            cut_sites.append(i)
    cut_sites.append(n)

    n_sites = len(cut_sites)
    # Each contiguous run of (missed_cleavages + 1) "fragments" is one peptide.
    for start_idx in range(n_sites - 1):
        for skip in range(missed_cleavages + 1):
            end_idx = start_idx + skip + 1
            if end_idx >= n_sites:
                break
            pep = sequence[cut_sites[start_idx] : cut_sites[end_idx]]
            if min_length <= len(pep) <= max_length:
                yield pep
