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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
from scipy.sparse import csc_matrix

from .accession_resolver import (
    extract_uniprot_accession,
    resolve_accessions,
)

logger = logging.getLogger(__name__)


@dataclass
class MappingMatrixResult:
    """Bundle of artifacts produced by :func:`build_mapping_matrix`.

    Attributes
    ----------
    matrix : np.ndarray, shape ``(P, T)``
        Dense ``int8`` peptide-to-taxon mapping matrix.
    peptide_list : list of str
        Row labels (observed peptides, deduplicated, in first-seen order).
    peptide_index : dict[str, int]
        ``peptide_sequence -> row index``.  Inverse of :attr:`peptide_list`.
    taxon_labels : list of str
        Column labels in the form ``"<taxon_id>|<taxon_name>"``.
    taxon_protein_peptides : dict[str, dict[str, list[str]]]
        ``taxon_label -> {protein_accession -> [peptide_sequences]}``.  Only
        proteins that produced at least one *observed* peptide are stored;
        the inner peptide lists are sorted and deduplicated.  This is the
        provenance information consumed by post-EM marker-based correction.
    unclassified_peptides : list of (str, str)
        ``(peptide_sequence, protein_accession)`` pairs for peptides whose
        only matches were against unclassified proteins (so the row sum in
        :attr:`matrix` is zero).
    """

    matrix: np.ndarray
    peptide_list: list
    peptide_index: dict
    taxon_labels: list
    taxon_protein_peptides: dict = field(default_factory=dict)
    unclassified_peptides: list = field(default_factory=list)

    # Backwards-compatible iteration: the historical return order was
    # ``(matrix, peptide_list, taxon_labels, unclassified_peptides)``.  Old
    # callers using positional unpacking continue to work without changes.
    def __iter__(self):
        yield self.matrix
        yield self.peptide_list
        yield self.taxon_labels
        yield self.unclassified_peptides

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
_SPECIES_RE = re.compile(r'\bspecies="([^"]+)"')

# Taxon-name sanity filter ----------------------------------------------------
# Substrings (case-insensitive) that indicate a functional annotation rather
# than an organism name. If any appears in a candidate taxon name, the name
# is rejected and the entry falls back to "unclassified".
_FUNCTIONAL_KEYWORDS = (
    "protein", "enzyme", "synthase", "transferase", "dehydrogenase",
    "kinase", "reductase", "oxidase", "subunit", "domain", "transporter",
    "receptor", "binding", "carrier", "hydrolase", "ligase", "lyase",
    "isomerase", "mutase", "permease", "peptidase", "protease", "lipase",
    "phosphatase", "helicase", "polymerase", "nuclease", "catalytic",
    "ribosomal", "hypothetical", "putative", "uncharacterized",
    "predicted", "probable", "conserved", "degradation", "biosynthesis",
    "metabolism",
)
# Common biochemical abbreviations that should never be treated as taxa.
_BIOCHEM_BLACKLIST = frozenset({
    "atp", "adp", "nad", "nadh", "nadp", "nadph", "fad", "coa",
    "amp", "gtp", "gdp", "ctp", "utp",
    "mn", "zn", "fe", "cu", "mg",
})

# Structural patterns indicating SEED/RAST gene identifiers rather than taxa.
_SEED_SUBSTRINGS = ("seed:", "fig|", "_peg.", "_peg ", "peg.")
_LOCUS_TAG_RE = re.compile(r"[A-Za-z]+[0-9]*_[a-z]+\.[0-9]+")

# Sub-species / infraspecific markers stripped during species-level
# normalisation.  Each marker includes a LEADING SPACE so that strings like
# "Burkholderiaceae" are not falsely clipped by a bare "bv".
_SUBSPECIES_MARKERS = (
    " bv. ", " subsp. ", " var. ", " strain ", " str. ", " pv. ",
    " serovar ", " serotype ", " biotype ", " ecotype ",
)


def build_mapping_matrix(
    peptides: list,
    fasta_path: str,
    enzyme: str = "trypsin",
    missed_cleavages: int = 2,
    min_length: int = 7,
    max_length: int = 50,
    exclude_prefixes: list | None = None,
    pepxml_protein_map: dict | None = None,
    resolve_uniprot: bool = False,
    prefix_map_file: str | None = None,
    taxon_level: str = "species",
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
    prefix_map_file : str, optional
        Path to a two-column TSV mapping accession prefixes to organism
        names (``<prefix>\\t<organism>``, no header).  Entries take
        priority over the automatic prefix-cohort inference and are used
        to rescue unclassified entries that share a prefix.
    taxon_level : str, optional
        ``"species"`` (default) or ``"strain"``.  Controls the depth of
        name normalisation applied to organism strings extracted from
        FASTA headers (see :func:`_normalize_taxon_name`).

    Returns
    -------
    MappingMatrixResult
        Dataclass bundling the dense ``(P, T)`` matrix, the peptide row
        labels and their inverse index, the taxon column labels in the
        form ``"<taxon_id>|<taxon_name>"``, the per-taxon per-protein
        peptide provenance map, and the unclassified-peptide diagnostic
        list.  For backwards compatibility, the dataclass also iterates
        as the historical 4-tuple
        ``(matrix, peptide_list, taxon_labels, unclassified_peptides)``,
        so positional unpacking by older callers continues to work.

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

    Entries for which no parser succeeds are rescued in two additional
    passes: a **prefix cohort** pass that reuses organism annotations from
    other entries sharing the same accession prefix (text before the first
    underscore), and an optional **user-supplied prefix map** that takes
    priority over the inferred cohort.  Proteins that remain unclassified
    after every rescue step are excluded from the matrix entirely.
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
    protein_taxon: dict = {}  # accession -> taxon_key (for substring matching)
    protein_seqs: dict = {}   # accession -> sequence (for substring matching)

    # Load the user-supplied prefix map up-front so it can override inferred
    # mappings during the cohort phase.
    user_prefix_map: dict = (
        _load_prefix_map(prefix_map_file) if prefix_map_file else {}
    )

    # First pass: parse every header so we can optionally look up organisms
    # for bare UniProt-accession headers before building the taxon buckets.
    parsed_records: list = []  # (accession, uniprot_acc, initial_key, seq)
    known_acc_organism: dict = {}  # UniProt accession -> organism (from fasta)
    organism_to_taxid: dict = {}   # organism name -> taxon_id (from fasta)
    unresolved_uniprot: set = set()
    total_proteins = 0
    classified_by_header = 0
    unclassified_initial = 0
    rejected_by_filter = 0
    for header, seq in _iter_fasta(fasta):
        if _should_exclude(header, exclude_prefixes):
            continue
        total_proteins += 1
        accession = _extract_accession(header)
        uniprot_acc = extract_uniprot_accession(accession)
        key, rejected_name = _parse_header_detailed(header, taxon_level=taxon_level)
        if rejected_name is not None:
            rejected_by_filter += 1
            logger.debug(
                "sanity filter rejected candidate organism %r from header %r",
                rejected_name, accession,
            )
        if key == ("0", "unclassified"):
            unclassified_initial += 1
            if resolve_uniprot and uniprot_acc:
                unresolved_uniprot.add(uniprot_acc)
        else:
            classified_by_header += 1
            if uniprot_acc and uniprot_acc not in known_acc_organism:
                known_acc_organism[uniprot_acc] = key[1]
            # Prefer a numeric OX-derived id over a slug for the same name.
            existing = organism_to_taxid.get(key[1])
            if existing is None or (not existing.isdigit() and key[0].isdigit()):
                organism_to_taxid[key[1]] = key[0]
        parsed_records.append((accession, uniprot_acc, key, seq))

    if rejected_by_filter:
        logger.info(
            "Taxon-name sanity filter rejected %d candidate organism "
            "string(s) that looked like functional annotations",
            rejected_by_filter,
        )

    # Phase 1 + 2 resolution for bare UniProt accessions.
    acc_to_organism: dict = {}
    if resolve_uniprot and unresolved_uniprot:
        acc_to_organism = resolve_accessions(
            fasta_path=fasta_path,
            unresolved_accessions=unresolved_uniprot,
            known_accession_organism=known_acc_organism,
            use_api=True,
        )

    # Apply UniProt resolution results to parsed_records in place.
    resolved_parsed: list = []
    resolved_via_lookup = 0
    for accession, uniprot_acc, key, seq in parsed_records:
        if key == ("0", "unclassified") and uniprot_acc:
            organism = acc_to_organism.get(uniprot_acc)
            if organism and _is_valid_taxon_name(organism):
                taxid = organism_to_taxid.get(organism) or _slug(organism)
                key = (taxid, organism)
                resolved_via_lookup += 1
        resolved_parsed.append((accession, uniprot_acc, key, seq))

    # --- Prefix-cohort inference ------------------------------------------
    # For entries that remain unclassified, consult the accession prefix
    # (text before the first underscore) and reuse organism assignments
    # learned from other proteins that share the same prefix.
    prefix_totals: dict = {}   # prefix -> total entries (classified+unclassified)
    prefix_votes: dict = {}    # prefix -> {key: count}
    for accession, _uniprot_acc, key, _seq in resolved_parsed:
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
            # Singleton prefixes carry no predictive value; skip to avoid
            # polluting the rescue path with unique per-protein prefixes
            # (typical of UniProt-style sp|ACC|NAME_ORG accessions).
            continue
        best_key, best_n = max(votes.items(), key=lambda kv: kv[1])
        if total > 0 and best_n / total >= 0.5 and _is_valid_taxon_name(best_key[1]):
            inferred_prefix_map[prefix] = best_key
            logger.info(
                "Prefix cohort map: %s -> %s (%d/%d headers)",
                prefix, best_key[1], best_n, total,
            )

    # Merge: user entries (already vetted by _load_prefix_map) override
    # inferred ones, and can also supply prefixes that had zero classified
    # headers on their own.
    combined_prefix_map: dict = dict(inferred_prefix_map)
    combined_prefix_map.update(user_prefix_map)

    rescued_by_user = 0
    rescued_by_cohort = 0
    still_unclassified = 0
    final_records: list = []
    for accession, uniprot_acc, key, seq in resolved_parsed:
        if key == ("0", "unclassified"):
            prefix = _extract_prefix(accession)
            rescued_key = combined_prefix_map.get(prefix) if prefix else None
            if rescued_key is not None:
                key = rescued_key
                if prefix in user_prefix_map:
                    rescued_by_user += 1
                else:
                    rescued_by_cohort += 1
            else:
                still_unclassified += 1
        final_records.append((accession, uniprot_acc, key, seq))

    logger.info(
        "Classification summary: total=%d, by_header=%d, via_uniprot_api=%d, "
        "via_cohort_prefix=%d, via_user_prefix=%d, unclassified=%d",
        total_proteins,
        classified_by_header,
        resolved_via_lookup,
        rescued_by_cohort,
        rescued_by_user,
        still_unclassified,
    )

    # --- Species-level deduplication ------------------------------------
    # After normalisation, multiple strain-level OX ids may map to the
    # same species name (e.g. OX=123 and OX=456 both yield "Rhizobium
    # leguminosarum").  Merge them into a single canonical key so that
    # the EM sees one column per species.
    name_to_keys: dict = {}  # taxon_name -> set of (taxon_id, taxon_name)
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

    if key_remap:
        n_original = sum(len(ks) for ks in name_to_keys.values() if len(ks) > 1)
        n_merged = sum(1 for ks in name_to_keys.values() if len(ks) > 1)
        logger.info(
            "Collapsed %d strain-level entries into %d species-level taxa",
            n_original,
            n_merged,
        )
        final_records = [
            (acc, uacc, key_remap.get(key, key), seq)
            for acc, uacc, key, seq in final_records
        ]

    # Bucket the resolved records.  Unclassified proteins are tracked
    # separately so we can digest them to find peptides that would be
    # excluded from the EM.  Each bucket stores ``(accession, seq)`` pairs
    # so the digestion loop can attribute observed peptides to their
    # source proteins.
    unclassified_seqs: list = []   # list of (accession, seq)
    for accession, _uniprot_acc, key, seq in final_records:
        protein_taxon[accession] = key
        if pepxml_protein_map is not None:
            protein_seqs[accession] = seq
        if key == ("0", "unclassified"):
            unclassified_seqs.append((accession, seq))
            continue
        taxon_buckets.setdefault(key, []).append((accession, seq))

    if not taxon_buckets:
        logger.warning("FASTA file %s yielded no parseable records", fasta_path)
        empty = np.zeros((P, 0), dtype=np.int8)
        return MappingMatrixResult(
            matrix=empty,
            peptide_list=peptide_list,
            peptide_index=dict(pep_index),
            taxon_labels=[],
            taxon_protein_peptides={},
            unclassified_peptides=[],
        )

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
        return MappingMatrixResult(
            matrix=empty,
            peptide_list=peptide_list,
            peptide_index=dict(pep_index),
            taxon_labels=[],
            taxon_protein_peptides={},
            unclassified_peptides=[],
        )

    surviving_columns: list = []
    next_col = 0
    # Per-taxon, per-protein observed-peptide provenance.  Keyed by the
    # ``"id|name"`` taxon label the EM consumes downstream.
    taxon_protein_peptides: dict = {}
    for taxon_id, taxon_name in taxon_keys_ordered:
        proteins = taxon_buckets[(taxon_id, taxon_name)]
        hits: set = set()
        proto_to_peps: dict = {}  # accession -> set of observed peptides
        for accession, seq in proteins:
            for pep in _digest(
                seq,
                enzyme=enzyme,
                missed_cleavages=missed_cleavages,
                min_length=min_length,
                max_length=max_length,
            ):
                if pep in target_pepset:
                    hits.add(pep)
                    proto_to_peps.setdefault(accession, set()).add(pep)
        if not hits:
            continue
        for pep in hits:
            rows.append(pep_index[pep])
            cols.append(next_col)
        surviving_columns.append((taxon_id, taxon_name))
        next_col += 1
        if proto_to_peps:
            label = f"{taxon_id}|{taxon_name}"
            taxon_protein_peptides[label] = {
                acc: sorted(peps) for acc, peps in proto_to_peps.items()
            }

    T = next_col
    if T == 0:
        logger.warning(
            "No peptides matched any taxon in %s; returning empty matrix",
            fasta_path,
        )
        empty = np.zeros((P, 0), dtype=np.int8)
        return MappingMatrixResult(
            matrix=empty,
            peptide_list=peptide_list,
            peptide_index=dict(pep_index),
            taxon_labels=[],
            taxon_protein_peptides={},
            unclassified_peptides=[],
        )

    # Substring matching for unmatched peptides using pepXML protein context.
    n_exact = len(rows)
    n_substring_matched = 0
    if pepxml_protein_map:
        matched_row_set = set(rows)
        unmatched_indices = set(range(P)) - matched_row_set
        if unmatched_indices:
            taxon_col = {key: col for col, key in enumerate(surviving_columns)}
            protein_digest_cache: dict = {}
            for idx in unmatched_indices:
                pep = peptide_list[idx]
                proteins = pepxml_protein_map.get(pep, set())
                if not proteins:
                    continue
                matched_taxa: set = set()
                for protein_acc in proteins:
                    tk = protein_taxon.get(protein_acc)
                    if not tk or tk not in taxon_col or tk in matched_taxa:
                        continue
                    if protein_acc not in protein_digest_cache:
                        seq = protein_seqs.get(protein_acc, "")
                        protein_digest_cache[protein_acc] = set(
                            _digest(seq, enzyme=enzyme,
                                    missed_cleavages=missed_cleavages,
                                    min_length=min_length,
                                    max_length=max_length)
                        )
                    if _has_substring_match(pep, protein_digest_cache[protein_acc]):
                        rows.append(idx)
                        cols.append(taxon_col[tk])
                        n_substring_matched += 1
                        matched_taxa.add(tk)
                        # Record provenance so marker correction sees
                        # substring-matched peptides too.
                        label = f"{tk[0]}|{tk[1]}"
                        protein_map = taxon_protein_peptides.setdefault(label, {})
                        peps = protein_map.setdefault(protein_acc, [])
                        if pep not in peps:
                            peps.append(pep)
        if n_substring_matched > 0:
            logger.info(
                "Substring matching: %d additional peptide-taxon assignments "
                "(%d exact, %d substring)",
                n_substring_matched, n_exact, n_substring_matched,
            )

    data = np.ones(len(rows), dtype=np.int8)
    sparse = csc_matrix((data, (rows, cols)), shape=(P, T), dtype=np.int8)
    # Densify in one shot. The trade-off threshold from the spec
    # ("use scipy.sparse internally when P*T > 100,000") refers to keeping
    # the construction sparse, which we do; the returned matrix matches the
    # type the model expects (numpy array).
    A = sparse.toarray()

    # Encode column labels as "id|name" for the plugin wrapper.
    taxon_labels = [f"{tid}|{tname}" for tid, tname in surviving_columns]

    # --- Track peptides that only matched unclassified proteins -----------
    # Digest the unclassified proteins to see which observed peptides they
    # contain. Then cross-reference with A: if a peptide has no real-taxon
    # column set (row sum == 0), it mapped ONLY to unclassified proteins.
    unclassified_pep_hits: dict = {}  # peptide -> list[accession]
    for acc, seq in unclassified_seqs:
        for pep in _digest(
            seq,
            enzyme=enzyme,
            missed_cleavages=missed_cleavages,
            min_length=min_length,
            max_length=max_length,
        ):
            if pep in target_pepset:
                unclassified_pep_hits.setdefault(pep, []).append(acc)

    row_sums = A.sum(axis=1)
    unclassified_peptides: list = []
    for pep, accs in unclassified_pep_hits.items():
        idx = pep_index.get(pep)
        if idx is not None and row_sums[idx] == 0:
            for acc in accs:
                unclassified_peptides.append((pep, acc))

    n_matched = int((row_sums > 0).sum())
    logger.info(
        "Built %d x %d mapping matrix (%d / %d observed peptides matched at least one taxon)",
        P,
        T,
        n_matched,
        P,
    )
    if unclassified_peptides:
        n_unique = len({pep for pep, _acc in unclassified_peptides})
        logger.info(
            "%d peptide(s) mapped only to unclassified proteins "
            "(excluded from EM, %d peptide-protein pairs total)",
            n_unique,
            len(unclassified_peptides),
        )

    # Re-canonicalise per-protein peptide lists (substring matching above
    # may have appended unsorted entries).
    for label, prot_map in taxon_protein_peptides.items():
        for acc, peps in prot_map.items():
            prot_map[acc] = sorted(set(peps))

    return MappingMatrixResult(
        matrix=A,
        peptide_list=peptide_list,
        peptide_index=dict(pep_index),
        taxon_labels=taxon_labels,
        taxon_protein_peptides=taxon_protein_peptides,
        unclassified_peptides=unclassified_peptides,
    )


def apply_detectability_weights(
    A: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Build a detectability-weighted emission matrix from a mapping matrix.

    Computes ``W`` where ``W_{pt} = d_p * A_{pt} / sum_{p'}(d_{p'} * A_{p't})``,
    i.e. the mapping matrix element-wise scaled by per-peptide detectability
    scores and then column-normalised per taxon.

    Parameters
    ----------
    A : np.ndarray or scipy.sparse matrix, shape ``(P, T)``
        Peptide-to-taxon mapping matrix (binary or float).  Sparse matrices
        are converted to dense internally.
    weights : np.ndarray, shape ``(P,)``
        Per-peptide detectability scores ``d_p``.

    Returns
    -------
    W : np.ndarray, shape ``(P, T)``
        Column-normalised weighted emission matrix.  The original ``A`` is
        not modified.

    Raises
    ------
    ValueError
        If the length of *weights* does not match the row dimension of *A*.
    """
    from scipy.sparse import issparse

    A_arr = A.toarray().astype(np.float64) if issparse(A) else np.asarray(A, dtype=np.float64)
    d = np.asarray(weights, dtype=np.float64)

    if d.shape[0] != A_arr.shape[0]:
        raise ValueError(
            f"weights length ({d.shape[0]}) must match number of peptides "
            f"({A_arr.shape[0]})"
        )

    # Element-wise weight: dA_{pt} = d_p * A_{pt}
    dA = A_arr * d[:, np.newaxis]

    # Column-normalise per taxon.
    col_sums = dA.sum(axis=0)
    col_sums_safe = np.where(col_sums == 0, 1.0, col_sums)
    W = dA / col_sums_safe[np.newaxis, :]

    # Force empty-taxon columns back to zero.
    W[:, col_sums == 0] = 0.0

    return W


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


def _parse_header(header: str, taxon_level: str = "species") -> tuple:
    """Extract a ``(taxon_id, taxon_name)`` pair from a FASTA header.

    Parameters
    ----------
    header : str
        Raw FASTA header line (without the leading ``>``).
    taxon_level : str
        ``"species"`` (default) or ``"strain"``.  Passed through to
        :func:`_normalize_taxon_name`.

    Candidate organism names are validated by :func:`_is_valid_taxon_name`;
    candidates that look like functional annotations (e.g. ``"ATP synthase"``)
    are rejected and the entry is reported as ``("0", "unclassified")``.

    Tries the following parsers in order:
    1. UniProt-style ``OS=`` / ``OX=`` fields (with taxon name normalization).
    2. GeneMark-style ``species="..."`` field.
    3. NCBI-style ``[Organism Name]`` brackets.
    4. Inline ``taxon_<id>`` markers.
    5. Pipe-prefixed organism (``>OrganismName|accession|...``).
    """
    key, _ = _parse_header_detailed(header, taxon_level=taxon_level)
    return key


# Namespace markers that should never be treated as organism names when they
# appear before the first pipe character.
_PIPE_NAMESPACE_BLACKLIST = frozenset({
    "sp", "tr", "gi", "ref", "gb", "seed", "fig", "peg",
})


def _parse_header_detailed(
    header: str, taxon_level: str = "species",
) -> tuple:
    """Like :func:`_parse_header` but also reports filtered candidates.

    Parameters
    ----------
    header : str
        Raw FASTA header line (without the leading ``>``).
    taxon_level : str
        ``"species"`` (default) or ``"strain"``.  Forwarded to
        :func:`_normalize_taxon_name` for every candidate name.

    Returns
    -------
    (key, rejected_candidate) : tuple
        ``key`` is the ``(taxon_id, taxon_name)`` pair.  ``rejected_candidate``
        is the raw organism string that the sanity filter discarded, or
        ``None`` if no candidate was rejected (either parsing succeeded or no
        parser matched).
    """
    if not header:
        return ("0", "unclassified"), None

    ox = _OX_RE.search(header)

    # 1. UniProt OS / OX
    os_match = _OS_RE.search(header)
    if os_match:
        raw_name = os_match.group(1).strip()
        name = _normalize_taxon_name(raw_name, level=taxon_level)
        if _is_valid_taxon_name(name):
            taxon_id = ox.group(1) if ox else _slug(name)
            return (taxon_id, name), None
        return ("0", "unclassified"), name

    # 2. GeneMark species="..."
    species_match = _SPECIES_RE.search(header)
    if species_match:
        raw_name = species_match.group(1).strip()
        name = _normalize_taxon_name(raw_name, level=taxon_level)
        if _is_valid_taxon_name(name):
            return (_slug(name), name), None
        return ("0", "unclassified"), name

    # 3. NCBI bracketed organism
    bracket = _BRACKET_RE.search(header)
    if bracket:
        raw_name = bracket.group(1).strip()
        name = _normalize_taxon_name(raw_name, level=taxon_level)
        if _is_valid_taxon_name(name):
            taxid = ox.group(1) if ox else _slug(name)
            return (taxid, name), None
        return ("0", "unclassified"), name

    # 4. Inline taxon_<id>
    inline = _GENERIC_TAXID_RE.search(header)
    if inline:
        # Numeric taxon ids are treated as valid on their face.
        return (inline.group(1), f"taxon_{inline.group(1)}"), None

    # 5. Pipe-prefixed organism: >OrganismName|accession|description
    #    Tightened: the pre-pipe token must contain at least one whitespace
    #    character ("Genus species" pattern).  Single tokens before a pipe
    #    are almost always accession or namespace markers, not organisms.
    if "|" in header:
        first = header.split("|", 1)[0].strip()
        if (
            first
            and first.lower() not in _PIPE_NAMESPACE_BLACKLIST
            and re.search(r"[A-Za-z]", first)
            and " " in first
        ):
            name = _normalize_taxon_name(first, level=taxon_level)
            if _is_valid_taxon_name(name):
                return (_slug(name), name), None
            return ("0", "unclassified"), first

    return ("0", "unclassified"), None


def _is_valid_taxon_name(name: str) -> bool:
    """Reject obvious non-taxonomic strings (functional annotations etc.).

    **Structural checks** (run first, cheapest):

    - Empty / whitespace-only.
    - Contains tab ``\\t``, newline ``\\n``, or pipe ``|`` characters
      (e.g. ``"AK199_peg.906\\tSEED:fig"`` → **rejected**).
    - Contains two or more consecutive whitespace characters, which
      usually indicate column-separated text, not a species name
      (e.g. ``"ORF1  some desc"`` → **rejected**).
    - Contains SEED/RAST gene-ID substrings (case-insensitive):
      ``"SEED:"``, ``"fig|"``, ``"_peg."``, ``"_peg "``, ``"peg."``
      (e.g. ``"AK199_peg.906 SEED:fig"`` → **rejected**).
    - Matches the gene-locus-tag pattern
      ``[A-Za-z]+[0-9]*_[a-z]+\\.[0-9]+`` anywhere in the string
      (e.g. ``"CV_peg.67"`` → **rejected**).
    - Contains **no lowercase letters** at all — real binomial names
      always have a lowercase species epithet; all-caps strings are
      accession codes or abbreviations
      (e.g. ``"FIG027190"`` → **rejected**,
      ``"Escherichia coli"`` → accepted).

    **Semantic checks** (existing):

    - Biochemical abbreviation blacklist (e.g. ``"ATP"``).
    - Functional-annotation keyword scan (e.g. ``"synthase"``).
    - Single token of four or fewer characters.
    """
    if not name or not name.strip():
        return False

    # --- Structural checks (cheap, catch SEED/RAST junk) ---

    # Control / separator characters
    if re.search(r"[\t\n|]", name):
        return False
    # Multiple consecutive whitespace (column-separated text)
    if re.search(r"\s{2,}", name):
        return False
    # SEED/RAST gene-identifier substrings (case-insensitive)
    lowered = name.strip().lower()
    for sub in _SEED_SUBSTRINGS:
        if sub in lowered:
            return False
    # Gene-locus-tag pattern (e.g. AK199_peg.906, CV_peg.67)
    if _LOCUS_TAG_RE.search(name):
        return False
    # No lowercase letters at all → accession code, not a taxon name
    if not re.search(r"[a-z]", name):
        return False

    # --- Semantic checks (existing) ---

    if lowered in _BIOCHEM_BLACKLIST:
        return False
    for kw in _FUNCTIONAL_KEYWORDS:
        if kw in lowered:
            return False
    # Reject very short single-word names.
    if len(name.split()) == 1 and len(name.strip()) <= 4:
        return False
    return True


def _extract_prefix(accession: str) -> str:
    """Return the FASTA-accession prefix used by the cohort inference step.

    The prefix is the substring before the first underscore in *accession*.
    Accessions with no underscore are returned verbatim.  An empty accession
    yields an empty string.

    Examples
    --------
    >>> _extract_prefix("CV_peg.67")
    'CV'
    >>> _extract_prefix("PaD_peg.1")
    'PaD'
    >>> _extract_prefix("bareAcc")
    'bareAcc'
    >>> _extract_prefix("")
    ''
    """
    if not accession:
        return ""
    return accession.split("_", 1)[0]


def _load_prefix_map(path: str) -> dict:
    """Load a user-provided ``prefix -> (taxid, organism)`` map from TSV.

    The file must be a two-column TSV (no header) of the form::

        PaD<TAB>Paracoccus denitrificans
        137<TAB>Staphylococcus aureus ATCC 13709

    Lines starting with ``#`` are treated as comments.  Entries whose
    organism name fails :func:`_is_valid_taxon_name` are discarded (and
    logged at ``INFO``).  The taxid column may be numeric or any other
    token; if it is empty or non-numeric a slug derived from the organism
    name is used instead.
    """
    out: dict = {}
    p = Path(path)
    if not p.is_file():
        logger.warning("prefix_map_file not found: %s", path)
        return out
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            prefix = parts[0].strip()
            organism = parts[1].strip()
            if not prefix or not organism:
                continue
            if not _is_valid_taxon_name(organism):
                logger.info(
                    "prefix_map_file: rejected %r -> %r (failed sanity filter)",
                    prefix, organism,
                )
                continue
            taxid = _slug(organism)
            out[prefix] = (taxid, organism)
    logger.info("Loaded %d user-provided prefix mappings from %s",
                len(out), path)
    return out


def _slug(name: str) -> str:
    """Make a deterministic id-like slug from a name when no real id exists."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return cleaned or "unknown"


def _normalize_taxon_name(name: str, level: str = "species") -> str:
    """Normalise a raw organism string to a canonical taxon name.

    Always strips trailing parenthetical content (strain annotations that
    appear inside parentheses, e.g. ``"(strain ATCC 15692)"``).

    When *level* is ``"species"`` (the default) three additional
    reductions are applied:

    1. **Sub-species markers** — the name is truncated at the first
       occurrence of any of: ``" bv. "``, ``" subsp. "``, ``" var. "``,
       ``" strain "``, ``" str. "``, ``" pv. "``, ``" serovar "``,
       ``" serotype "``, ``" biotype "``, ``" ecotype "``.
    2. **Colon suffixes** — trailing ``" : <AlphanumericToken>"`` is
       stripped (common in JGI/IMG identifiers).
    3. **Binomial truncation** — if the result has three or more tokens
       where the first is Capitalised and the second is lowercase
       (classic binomial pattern), only the first two are kept.

    When *level* is ``"strain"``, only the parenthetical strip is applied
    and the remaining text is returned as-is.

    Examples
    --------
    >>> _normalize_taxon_name("Pseudomonas aeruginosa (strain ATCC 15692)")
    'Pseudomonas aeruginosa'
    >>> _normalize_taxon_name("Rhizobium leguminosarum bv. viciae 3841", level="species")
    'Rhizobium leguminosarum'
    >>> _normalize_taxon_name("Pseudomonas fluorescens ATCC 13525 : Ga0070645_11", level="species")
    'Pseudomonas fluorescens'
    >>> _normalize_taxon_name("Rhizobium leguminosarum bv. viciae 3841", level="strain")
    'Rhizobium leguminosarum bv. viciae 3841'
    """
    # Step 0: strip trailing parenthetical content (always, both modes).
    result = re.sub(r"\s*\(.*?\)\s*$", "", name).strip() or name

    if level == "strain":
        return result

    # --- species-level reductions ---

    # Step 1: strip from first sub-species marker onward.
    result_lower = result.lower()
    earliest = len(result)
    for marker in _SUBSPECIES_MARKERS:
        idx = result_lower.find(marker)
        if 0 <= idx < earliest:
            earliest = idx
    if earliest < len(result):
        result = result[:earliest].strip()

    # Step 2: strip trailing " : AlphanumericToken" suffix.
    colon_match = re.search(r" : ([A-Za-z0-9_]+)\s*$", result)
    if colon_match:
        result = result[: colon_match.start()].strip()

    # Step 3: binomial truncation — keep first two tokens when the name
    # looks like "Genus species <extra stuff>".
    tokens = result.split()
    if (
        len(tokens) >= 3
        and tokens[0][:1].isupper()
        and tokens[1][:1].islower()
    ):
        result = " ".join(tokens[:2])

    return result or name


def _should_exclude(header: str, exclude_prefixes: list | None) -> bool:
    """Return *True* if the FASTA header matches any exclusion prefix."""
    if not exclude_prefixes:
        return False
    header_lower = header.lower()
    return any(header_lower.startswith(p.lower()) for p in exclude_prefixes)


def _extract_accession(header: str) -> str:
    """Return the first whitespace-delimited token of a FASTA header."""
    return header.split()[0] if header else ""


def _has_substring_match(
    observed: str,
    theoretical_set: set,
    min_ratio: float = 0.7,
) -> bool:
    """Check containment between *observed* and any member of *theoretical_set*.

    A match requires that one sequence is a substring of the other **and**
    ``min(len_obs, len_theo) / max(len_obs, len_theo) >= min_ratio``.
    """
    obs_len = len(observed)
    for theo in theoretical_set:
        theo_len = len(theo)
        shorter = min(obs_len, theo_len)
        longer = max(obs_len, theo_len)
        if shorter / longer < min_ratio:
            continue
        if observed in theo or theo in observed:
            return True
    return False


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
