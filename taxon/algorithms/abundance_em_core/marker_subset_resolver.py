"""Resolve bac120/ar53 marker families into ribosomal / Nayfach-30 subsets.

The post-EM marker correction (``marker_correction.compute_cell_equivalent_abundance``)
defaults to using **every** marker family that HMMER fires against the
reference FASTA — i.e. the full GTDB bac120 (120 bacterial) and ar53 (53
archaeal) profile sets, ~173 families combined.  Two stricter subsets are
also useful for cross-checking:

* **Ribosomal subset.**  Restricting signal to ribosomal-protein markers
  approximates the operational definition used by GTDB-Tk's "fast" tree
  pipeline (Parks 2018).  Auto-detected from each HMM's ``DESC`` field
  so it is portable across HMM releases.

* **Nayfach-30 subset.**  MicrobeCensus (Nayfach & Pollard 2015, Genome
  Biology 16:51, Supplementary Table 1) uses 30 essential single-copy
  marker proteins (27 ribosomal + IF-2 + the two PheRS subunits) for
  average-genome-size estimation.  Restricting signal to the bac120/ar53
  families that correspond to these 30 markers yields a "translation-
  core" cell-equivalent abundance that is comparable across releases of
  the bac120 set and against single-copy-marker tools.

This module provides three small, pure-Python utilities for that
resolution.  Nothing here depends on HMMER actually being installed; the
HMM-header parsing reads plain text and stops at the start of each
profile's emission matrix.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# Regex for the suffix code of a ribosomal protein description, e.g.
# "ribosomal protein L11" -> "L11", "ribosomal protein S12/S23" -> "S12"
# (the leading group is taken; "/S23" is consumed only by the broader
# substring fallback in resolve_nayfach30_to_bac120).
_RIBO_NAYFACH_SUFFIX = re.compile(r"protein\s+([SL]\d+[a-z]?)", re.IGNORECASE)

# Regex for "is this DESC field a ribosomal protein?" — the bare suffix
# token, e.g. "L7", "L7Ae", "S12".
_RIBO_TOKEN = re.compile(r"\b[rsl]\d+[a-z]?\b")


@dataclass
class NayfachResolution:
    """Output of :func:`resolve_nayfach30_to_bac120`.

    Attributes
    ----------
    subset : set[str]
        Union of all bac120/ar53 family accessions that matched at least
        one of the 30 Nayfach markers.
    audit_map : dict[str, list[str]]
        Per-Nayfach-marker breakdown: ``nayfach_id -> [bac120/ar53
        family accession, ...]``.  Empty list for unmatched markers.
    unmatched : list[str]
        Nayfach marker ids that produced zero matches.
    """

    subset: set = field(default_factory=set)
    audit_map: dict = field(default_factory=dict)
    unmatched: list = field(default_factory=list)


# ---------------------------------------------------------------- HMM headers


def load_bac120_ar53_descriptions(hmm_profile_dir: Path) -> dict:
    """Return ``{family_accession: DESC_string}`` for every HMM in *dir*.

    The HMM3 text format records one profile per block, opened by a
    ``HMMER3/f`` line and closed by ``//``.  The model matrix proper
    begins with a line starting with ``HMM`` (and an alphabet on the
    same line); we stop reading the current record's header at that
    boundary so the rest of the file can be skipped cheaply.

    Multiple concatenated profiles per ``*.hmm`` file are supported —
    the bac120 / ar53 bundles produced by ``scripts/download_marker_hmms.py``
    use exactly that layout.

    Parameters
    ----------
    hmm_profile_dir : Path
        Directory containing one or more ``*.hmm`` / ``*.HMM`` files.
        Subdirectories are not searched.

    Returns
    -------
    dict[str, str]
        Keyed on the ``NAME`` field of each profile (e.g. ``TIGR00001``).
        The value is the verbatim ``DESC`` string with surrounding
        whitespace stripped.  Profiles missing a ``DESC`` field get the
        empty string.

    Notes
    -----
    The function does not raise if a file is unreadable or malformed —
    it logs at WARNING and skips.  Pure diagnostics should not crash a
    pipeline run.
    """
    profile_dir = Path(hmm_profile_dir)
    if not profile_dir.is_dir():
        raise FileNotFoundError(
            f"HMM profile directory not found: {hmm_profile_dir}"
        )

    descriptions: dict = {}
    # Accept both lowercase and uppercase extensions; sorted for stable
    # iteration in tests / logs.
    hmm_files = sorted(
        list(profile_dir.glob("*.hmm")) + list(profile_dir.glob("*.HMM"))
    )
    for path in hmm_files:
        try:
            _parse_one_hmm_file(path, descriptions)
        except OSError as exc:
            logger.warning(
                "Could not read HMM profile file %s: %s", path, exc,
            )
    return descriptions


def _parse_one_hmm_file(path: Path, descriptions: dict) -> None:
    """Append every ``(NAME, DESC)`` pair found in *path* to *descriptions*.

    A single ``*.hmm`` file may concatenate many profile records (the
    bac120 bundles do).  Each record starts with ``HMMER3/f`` and ends
    with ``//``; the model matrix line ``HMM <alphabet>`` marks the end
    of the header for that record, so we drop into a fast-skip mode
    until ``//`` is seen.
    """
    current_name: str | None = None
    current_desc: str | None = None
    in_matrix = False

    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if line.startswith("HMMER3"):
                # Start of a new record.  If we already had a name buffered
                # from the previous (malformed) record, flush it under the
                # best-effort assumption.
                if current_name is not None:
                    descriptions[current_name] = current_desc or ""
                current_name = None
                current_desc = None
                in_matrix = False
                continue
            if line.strip() == "//":
                # End of record.
                if current_name is not None:
                    descriptions[current_name] = current_desc or ""
                current_name = None
                current_desc = None
                in_matrix = False
                continue
            if in_matrix:
                continue
            # First field is the tag, rest is the value.
            parts = line.split(None, 1)
            if not parts:
                continue
            tag = parts[0]
            value = parts[1].strip() if len(parts) > 1 else ""
            if tag == "NAME":
                current_name = value
            elif tag == "DESC":
                current_desc = value
            elif tag == "HMM":
                # Start of the emission-matrix block — stop reading
                # header fields for this record.
                in_matrix = True
    # Catch a final record that ended at EOF without a terminating "//".
    if current_name is not None:
        descriptions[current_name] = current_desc or ""


# ------------------------------------------------------------- ribo subset


def identify_ribo_subset(family_descriptions: dict) -> tuple:
    """Return ``(ribo_family_set, audit_list)``.

    A DESC field qualifies as ribosomal iff (case-insensitive on the
    stripped/lowercased string):

    * the substring ``"ribosomal protein"`` is present, **or**
    * the regex ``\\b[rsl]\\d+[a-z]?\\b`` (e.g. ``S1``, ``L7``,
      ``L7Ae``) matches.

    The audit list is the same set, ordered as ``[(family_acc, DESC), ...]``
    sorted by family accession, suitable for direct serialization.
    """
    ribo: set = set()
    audit: list = []
    for acc, desc in family_descriptions.items():
        normalized = (desc or "").strip().lower()
        if not normalized:
            continue
        if "ribosomal protein" in normalized or _RIBO_TOKEN.search(normalized):
            ribo.add(acc)
            audit.append((acc, desc))
    audit.sort(key=lambda r: r[0])
    return ribo, audit


# ---------------------------------------------------------- nayfach-30 subset


def resolve_nayfach30_to_bac120(
    nayfach_tsv_path: Path,
    family_descriptions: dict,
) -> NayfachResolution:
    """Resolve the 30 Nayfach markers against bac120/ar53 DESC strings.

    Parameters
    ----------
    nayfach_tsv_path : Path
        TSV with header ``marker_id\\tko_id\\tdescription``.  Trailing
        whitespace on each line is stripped; blank and ``#``-prefixed
        lines are skipped.
    family_descriptions : dict
        Output of :func:`load_bac120_ar53_descriptions`.

    Returns
    -------
    NayfachResolution

    Notes
    -----
    Matching rules (all case-insensitive after ``.lower().strip()``):

    * Ribosomal markers (``"ribosomal protein"`` in the Nayfach DESC):
      extract the suffix code with
      ``r"protein\\s+([SL]\\d+[a-z]?)"`` and match against bac120 DESC
      containing ``r"ribosomal\\s+protein\\s+\\1\\b"``.
    * ``"translation initiation factor IF-2"``: bac120 DESC contains
      ``"initiation factor if-2"``, ``"if-2"``, or ``"if2"``.
    * ``"phenylalanyl-tRNA synthetase alpha chain"``: bac120 DESC
      contains ``"phenylalanyl-trna"`` AND (``"alpha"`` or ``"α"``).
    * ``"phenylalanyl-tRNA synthetase beta chain"``: ``"phenylalanyl-trna"``
      AND (``"beta"`` or ``"β"``).

    One Nayfach marker may resolve to multiple bac120/ar53 families
    (e.g. when the bac120 set has both a TIGRFAM and a PFAM profile for
    the same protein).  The final ``subset`` is the union across all 30
    Nayfach markers.
    """
    path = Path(nayfach_tsv_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Nayfach-30 reference TSV not found: {nayfach_tsv_path}"
        )

    # Pre-compute lowercase DESC strings so we don't redo .lower() in
    # the inner matching loop.
    lower_desc = {acc: (d or "").strip().lower() for acc, d in family_descriptions.items()}

    resolution = NayfachResolution()

    with path.open("r", encoding="utf-8") as fh:
        for ln_no, raw in enumerate(fh, 1):
            line = raw.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            marker_id, _ko_id, desc = parts[0], parts[1], parts[2]
            # Skip the header row.
            if marker_id.strip().lower() == "marker_id":
                continue
            matches = _match_one_nayfach(desc, lower_desc)
            resolution.audit_map[marker_id] = matches
            if matches:
                resolution.subset.update(matches)
            else:
                resolution.unmatched.append(marker_id)
    return resolution


def _match_one_nayfach(desc: str, lower_desc: dict) -> list:
    """Resolve a single Nayfach marker description.  Returns sorted list of acc."""
    normalized = (desc or "").strip().lower()
    if not normalized:
        return []

    matched: set = set()

    if "ribosomal protein" in normalized:
        m = _RIBO_NAYFACH_SUFFIX.search(desc)
        if m is None:
            return []
        suffix = m.group(1).upper()  # e.g. "L11"
        # Match "ribosomal protein L11" or "ribosomal\nprotein\tL11" etc.
        # The bac120 DESC may include surrounding context (e.g. "Ribosomal
        # protein L11, RNA binding domain") so we use \b on both sides of
        # the suffix.
        pattern = re.compile(
            r"ribosomal\s+protein\s+" + re.escape(suffix) + r"\b",
            re.IGNORECASE,
        )
        for acc, ldesc in lower_desc.items():
            if pattern.search(ldesc):
                matched.add(acc)
        return sorted(matched)

    if "translation initiation factor if-2" in normalized or "initiation factor if-2" in normalized:
        for acc, ldesc in lower_desc.items():
            if (
                "initiation factor if-2" in ldesc
                or "if-2" in ldesc
                or "if2" in ldesc
            ):
                matched.add(acc)
        return sorted(matched)

    if "phenylalanyl-trna" in normalized or "phenylalanyl-trna synthetase" in normalized:
        is_alpha = "alpha" in normalized or "α" in normalized
        is_beta = "beta" in normalized or "β" in normalized
        for acc, ldesc in lower_desc.items():
            if "phenylalanyl-trna" not in ldesc:
                continue
            if is_alpha and ("alpha" in ldesc or "α" in ldesc):
                matched.add(acc)
            elif is_beta and ("beta" in ldesc or "β" in ldesc):
                matched.add(acc)
        return sorted(matched)

    # Unknown marker type — leave unmatched so the caller can flag it.
    return []


# ------------------------------------------------------------- diagnostic IO


def write_subset_diagnostics(
    output_dir: Path,
    ribo_audit: list,
    nayfach_resolution: NayfachResolution,
    family_descriptions: dict,
    nayfach_tsv_path: Path | None = None,
) -> tuple:
    """Write the two diagnostic TSVs under ``output_dir/diagnostics/``.

    Returns the pair of paths so callers / tests can verify the write.

    * ``nayfach30_bac120_mapping.tsv``:
      ``nayfach_marker_id, ko_id, description, matched_bac120_families``
      (the last column is a comma-separated list, empty for unmatched
      markers).
    * ``ribo_subset_families.tsv``:
      ``family_accession, DESC``.
    """
    diag_dir = Path(output_dir) / "diagnostics"
    diag_dir.mkdir(parents=True, exist_ok=True)

    nayfach_path = diag_dir / "nayfach30_bac120_mapping.tsv"
    ribo_path = diag_dir / "ribo_subset_families.tsv"

    # Re-read the Nayfach TSV to recover ko_id + description for each row.
    nayfach_rows: list = []
    if nayfach_tsv_path is not None and Path(nayfach_tsv_path).is_file():
        with Path(nayfach_tsv_path).open("r", encoding="utf-8") as fh:
            for raw in fh:
                line = raw.rstrip("\r\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split("\t")
                if len(parts) < 3 or parts[0].strip().lower() == "marker_id":
                    continue
                nayfach_rows.append(parts[:3])

    with nayfach_path.open("w", encoding="utf-8") as fh:
        fh.write("nayfach_marker_id\tko_id\tdescription\tmatched_bac120_families\n")
        if nayfach_rows:
            for marker_id, ko_id, desc in nayfach_rows:
                matched = nayfach_resolution.audit_map.get(marker_id, [])
                fh.write(
                    f"{marker_id}\t{ko_id}\t{desc}\t{','.join(matched)}\n"
                )
        else:
            # Fall back to whatever the resolution carries — at least we
            # emit a row per Nayfach id with the family list, even if the
            # TSV path is unavailable.
            for marker_id, matched in sorted(nayfach_resolution.audit_map.items()):
                fh.write(f"{marker_id}\t\t\t{','.join(matched)}\n")

    with ribo_path.open("w", encoding="utf-8") as fh:
        fh.write("family_accession\tDESC\n")
        for acc, desc in ribo_audit:
            fh.write(f"{acc}\t{desc}\n")

    return nayfach_path, ribo_path


# --------------------------------------------------------------------- utility


DEFAULT_NAYFACH30_TSV = (
    Path(__file__).parent / "data" / "nayfach30_markers.tsv"
)
