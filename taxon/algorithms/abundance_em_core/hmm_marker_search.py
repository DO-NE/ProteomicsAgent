"""HMMER-based marker-protein identification for cell-equivalent correction.

Wraps the external ``hmmsearch`` binary so a reference FASTA can be
scanned against the GTDB bac120 (bacterial) and ar53 (archaeal) HMM
profiles.  Each hit yields a ``(protein_accession, marker_family)``
pair that downstream :mod:`marker_correction` uses to identify which
peptides are universal-single-copy markers.

External requirements
---------------------
- ``hmmsearch`` on ``$PATH``.  Install via
  ``conda install -c bioconda hmmer`` or ``apt-get install hmmer``.
- A directory containing the GTDB bac120 / ar53 marker HMM profiles.
  The default layout produced by :mod:`scripts.download_marker_hmms`
  places concatenated profiles at
  ``<dir>/bac120_markers.hmm`` and ``<dir>/ar53_markers.hmm``;
  individual ``*.hmm`` files in the directory are also accepted.

This module never spawns ``hmmsearch`` if both prerequisites are
available — every public function fails fast with a clear error
message instead of producing partial output.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


HMMER_INSTALL_HINT = (
    "hmmsearch binary not found on PATH. Install HMMER 3.x "
    "(`conda install -c bioconda hmmer` or `apt-get install hmmer`) "
    "and rerun, or disable the marker correction with --no-marker-correction."
)


HMM_PROFILE_HINT = (
    "No HMM profiles found in {dir!r}.  Run "
    "`python scripts/download_marker_hmms.py --output-dir {dir}` "
    "to fetch the bac120/ar53 marker HMMs from EBI/InterPro, or place "
    "concatenated profile files (bac120_markers.hmm / ar53_markers.hmm) "
    "in that directory manually."
)


@dataclass
class MarkerSearchResult:
    """Result bundle returned by :func:`run_hmmsearch`."""

    # protein_accession -> (taxon_label, [marker_family, ...], e_value, score)
    # The family list is ordered best-score first; multi-domain proteins
    # accumulate all matching HMM family names rather than just the best.
    marker_proteins: dict = field(default_factory=dict)
    # taxon_label -> set of marker_family names found in that taxon
    taxon_marker_families: dict = field(default_factory=dict)
    # marker_family -> set of protein_accessions
    family_proteins: dict = field(default_factory=dict)
    # Counts
    n_markers_found: int = 0
    n_taxa_with_markers: int = 0
    # Bookkeeping — paths searched, HMM files used, etc., for caching/logging.
    hmm_files_used: list = field(default_factory=list)
    cache_path: Optional[str] = None


# ---------------------------------------------------------------------- public


def run_hmmsearch(
    fasta_path: str,
    hmm_profile_dir: str,
    evalue_threshold: float = 1e-10,
    score_threshold: Optional[float] = None,
    cache_dir: Optional[str] = None,
    taxon_protein_peptides: Optional[dict] = None,
    cpu: Optional[int] = None,
) -> MarkerSearchResult:
    """Identify marker proteins in *fasta_path* using HMMER.

    Parameters
    ----------
    fasta_path : str
        Reference protein FASTA used during the database search.
    hmm_profile_dir : str
        Directory containing GTDB bac120/ar53 HMM profile files.  Either
        concatenated single-file profiles
        (``bac120_markers.hmm`` / ``ar53_markers.hmm``) or a flat
        directory of individual ``*.hmm`` profiles is accepted; both are
        searched if present.
    evalue_threshold : float, default ``1e-10``
        ``hmmsearch -E`` cutoff.  Profile-specific gathering thresholds
        from the HMM headers are NOT used — the same E-value is applied
        uniformly so that bac120 and ar53 are comparable.
    score_threshold : float, optional
        Optional minimum bit-score; hits with ``score < threshold`` are
        dropped after parsing.
    cache_dir : str, optional
        Directory where the parsed result is cached as JSON, keyed on
        the FASTA + HMM checksums.  When unset, no cache is written.
    taxon_protein_peptides : dict, optional
        ``taxon_label -> {protein_accession -> peptides}`` from the
        :class:`MappingMatrixResult`.  Used to attach the canonical
        taxon label to each marker hit; without it, every hit gets
        the placeholder label ``""``.
    cpu : int, optional
        Forwarded as ``hmmsearch --cpu``.  When unset, HMMER picks
        a default based on host topology.

    Returns
    -------
    MarkerSearchResult

    Raises
    ------
    FileNotFoundError
        ``fasta_path`` or ``hmm_profile_dir`` does not exist, or no
        ``*.hmm`` files were found in the profile directory.
    RuntimeError
        ``hmmsearch`` is not installed on ``$PATH``, or it returned a
        non-zero exit code.
    """
    fasta = Path(fasta_path)
    if not fasta.is_file():
        raise FileNotFoundError(f"FASTA file not found: {fasta_path}")
    profile_dir = Path(hmm_profile_dir)
    if not profile_dir.is_dir():
        raise FileNotFoundError(
            f"HMM profile directory not found: {hmm_profile_dir}. "
            + HMM_PROFILE_HINT.format(dir=hmm_profile_dir)
        )

    hmm_files = _collect_hmm_files(profile_dir)
    if not hmm_files:
        raise FileNotFoundError(HMM_PROFILE_HINT.format(dir=hmm_profile_dir))

    binary = shutil.which("hmmsearch")
    if binary is None:
        raise RuntimeError(HMMER_INSTALL_HINT)

    # Try loading cache before running hmmsearch.
    cache_key = _cache_key(fasta, hmm_files, evalue_threshold, score_threshold)
    cache_path: Optional[Path] = None
    if cache_dir:
        cache_dir_p = Path(cache_dir)
        cache_dir_p.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir_p / f"hmm_marker_cache_{cache_key}.json"
        cached = _load_cache(cache_path)
        if cached is not None:
            logger.info("hmmsearch cache hit: %s", cache_path)
            cached.cache_path = str(cache_path)
            return cached

    # Run hmmsearch once per HMM file and merge tblout outputs.
    parsed_hits: list = []
    for hmm in hmm_files:
        hits = _run_one_hmmsearch(
            binary=binary,
            hmm_path=hmm,
            fasta_path=fasta,
            evalue_threshold=evalue_threshold,
            cpu=cpu,
        )
        parsed_hits.extend(hits)

    # Optional bit-score filter (post-parse so HMMER doesn't drop hits
    # before we can report the diagnostic count).
    if score_threshold is not None:
        before = len(parsed_hits)
        parsed_hits = [h for h in parsed_hits if h["score"] >= score_threshold]
        logger.info(
            "Score threshold %.2f filtered %d/%d hits",
            score_threshold, before - len(parsed_hits), before,
        )

    result = _aggregate_hits(parsed_hits, taxon_protein_peptides or {})
    result.hmm_files_used = [str(p) for p in hmm_files]

    if cache_path is not None:
        _save_cache(cache_path, result)
        result.cache_path = str(cache_path)

    logger.info(
        "hmmsearch found %d marker hits across %d taxa "
        "(%d distinct families)",
        result.n_markers_found,
        result.n_taxa_with_markers,
        len(result.family_proteins),
    )
    return result


def parse_hmmsearch_tblout(
    tblout_path: str,
    evalue_threshold: float = 1e-10,
) -> list:
    """Parse a ``hmmsearch --tblout`` file.

    The format is whitespace-delimited with comment lines beginning
    with ``#``.  Column layout follows the HMMER 3.x manual::

        target_name  target_acc  query_name  query_acc  E-value  score  bias
        best_dom_E   best_dom_score  ... (further columns ignored)

    Hits with ``E-value > evalue_threshold`` are skipped.
    """
    out: list = []
    p = Path(tblout_path)
    if not p.is_file():
        return out
    with p.open("r", encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            target_name = parts[0]
            query_name = parts[2]
            try:
                evalue = float(parts[4])
                score = float(parts[5])
            except ValueError:
                continue
            if evalue > evalue_threshold:
                continue
            out.append(
                {
                    "target_name": target_name,
                    "query_name": query_name,
                    "evalue": evalue,
                    "score": score,
                }
            )
    return out


# --------------------------------------------------------------------- helpers


def _collect_hmm_files(profile_dir: Path) -> list:
    """Return a stable, sorted list of ``*.hmm`` files in *profile_dir*.

    Concatenated bundles (``bac120_markers.hmm`` / ``ar53_markers.hmm``)
    take precedence: when present, only those are used to avoid scanning
    the same family twice.  Otherwise every ``*.hmm`` file in the
    directory (non-recursive) is used.
    """
    bundled = [
        profile_dir / "bac120_markers.hmm",
        profile_dir / "ar53_markers.hmm",
    ]
    bundles_present = [p for p in bundled if p.is_file()]
    if bundles_present:
        return bundles_present

    files = sorted(p for p in profile_dir.glob("*.hmm") if p.is_file())
    return files


def _run_one_hmmsearch(
    binary: str,
    hmm_path: Path,
    fasta_path: Path,
    evalue_threshold: float,
    cpu: Optional[int],
) -> list:
    """Run hmmsearch against one HMM file and return parsed hits."""
    with tempfile.NamedTemporaryFile("r", suffix=".tblout", delete=False) as tmp:
        tblout_path = tmp.name
    try:
        cmd: list = [
            binary,
            "--tblout", tblout_path,
            "--noali",
            "-E", _fmt_evalue(evalue_threshold),
        ]
        if cpu is not None:
            cmd += ["--cpu", str(int(cpu))]
        cmd += [str(hmm_path), str(fasta_path)]

        logger.info("Running: %s", " ".join(cmd))
        completed = subprocess.run(
            cmd, capture_output=True, text=True, check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"hmmsearch failed (exit {completed.returncode}) for "
                f"{hmm_path.name}: {completed.stderr.strip()[:500]}"
            )
        hits = parse_hmmsearch_tblout(tblout_path, evalue_threshold)
        logger.info(
            "hmmsearch %s -> %d hits at E<=%g",
            hmm_path.name, len(hits), evalue_threshold,
        )
        return hits
    finally:
        try:
            os.unlink(tblout_path)
        except OSError:
            pass


def _aggregate_hits(
    hits: list,
    taxon_protein_peptides: dict,
) -> MarkerSearchResult:
    """Aggregate raw tblout hits into a :class:`MarkerSearchResult`.

    For every ``(target_name, query_name)`` we keep the *best* (lowest
    E-value, highest bit-score) hit.  Cross-references the target
    accession against the per-taxon protein index from the mapping-matrix
    build to assign a taxon label.  Targets not present in any taxon
    bucket get the empty-string label.
    """
    # Reverse index: protein_accession -> taxon_label.  The matrix build
    # may bucket the same accession under multiple labels (rare); pick
    # the lexicographically smallest for stability.
    acc_to_label: dict = {}
    for label, prot_map in taxon_protein_peptides.items():
        for acc in prot_map:
            existing = acc_to_label.get(acc)
            if existing is None or label < existing:
                acc_to_label[acc] = label

    # Best-hit dedup.  HMMER may report the same target/query pair with
    # multiple domains; we just want one row per pair.
    best: dict = {}
    for h in hits:
        key = (h["target_name"], h["query_name"])
        prev = best.get(key)
        if prev is None or (h["evalue"] < prev["evalue"]):
            best[key] = h

    marker_proteins: dict = {}
    family_proteins: dict = {}
    taxon_marker_families: dict = {}

    for (target, query), h in best.items():
        # Record all matching family names for multi-domain proteins so
        # downstream family counters see every membership, not just the
        # best-scoring one.  The family list is ordered best-score first.
        family_proteins.setdefault(query, set()).add(target)

        label = acc_to_label.get(target, "")
        if label:
            taxon_marker_families.setdefault(label, set()).add(query)

        existing = marker_proteins.get(target)
        if existing is None:
            marker_proteins[target] = (label, [query], h["evalue"], h["score"])
        else:
            ex_label, ex_families, ex_evalue, ex_score = existing
            new_families = list(ex_families)
            if query not in new_families:
                if h["evalue"] < ex_evalue:
                    new_families.insert(0, query)
                else:
                    new_families.append(query)
            if h["evalue"] < ex_evalue:
                marker_proteins[target] = (ex_label, new_families, h["evalue"], h["score"])
            else:
                marker_proteins[target] = (ex_label, new_families, ex_evalue, ex_score)

    return MarkerSearchResult(
        marker_proteins=marker_proteins,
        taxon_marker_families=taxon_marker_families,
        family_proteins=family_proteins,
        n_markers_found=len(marker_proteins),
        n_taxa_with_markers=len(taxon_marker_families),
    )


# --------------------------------------------------------------- caching


def _cache_key(
    fasta: Path,
    hmm_files: Iterable[Path],
    evalue: float,
    score_threshold: Optional[float],
) -> str:
    """Build a short content-derived cache key.

    Hashing the full FASTA / full HMMs would dominate runtime on big
    files, so we use the same lightweight fingerprint pattern as
    :mod:`accession_resolver`: file size + first/last 10KB.
    """
    h = hashlib.sha256()
    h.update(_fingerprint(fasta).encode())
    for hmm in sorted(hmm_files):
        h.update(b"|")
        h.update(_fingerprint(hmm).encode())
    h.update(f"|E={evalue}|S={score_threshold}".encode())
    return h.hexdigest()[:16]


def _fingerprint(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return f"missing:{path.name}"
    head = b""
    tail = b""
    try:
        with path.open("rb") as fh:
            head = fh.read(10 * 1024)
            if size > 20 * 1024:
                fh.seek(max(0, size - 10 * 1024))
                tail = fh.read(10 * 1024)
    except OSError:
        return f"err:{path.name}:{size}"
    digest = hashlib.sha256(head + tail).hexdigest()[:16]
    return f"{path.name}:{size}:{digest}"


def _load_cache(cache_path: Path) -> Optional[MarkerSearchResult]:
    if not cache_path.is_file():
        return None
    try:
        with cache_path.open("r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read hmm cache %s: %s", cache_path, exc)
        return None
    return MarkerSearchResult(
        marker_proteins={
            k: tuple(v) for k, v in obj.get("marker_proteins", {}).items()
        },
        taxon_marker_families={
            k: set(v) for k, v in obj.get("taxon_marker_families", {}).items()
        },
        family_proteins={
            k: set(v) for k, v in obj.get("family_proteins", {}).items()
        },
        n_markers_found=int(obj.get("n_markers_found", 0)),
        n_taxa_with_markers=int(obj.get("n_taxa_with_markers", 0)),
        hmm_files_used=list(obj.get("hmm_files_used", [])),
    )


def _save_cache(cache_path: Path, result: MarkerSearchResult) -> None:
    obj = {
        "marker_proteins": {
            k: list(v) for k, v in result.marker_proteins.items()
        },
        "taxon_marker_families": {
            k: sorted(v) for k, v in result.taxon_marker_families.items()
        },
        "family_proteins": {
            k: sorted(v) for k, v in result.family_proteins.items()
        },
        "n_markers_found": result.n_markers_found,
        "n_taxa_with_markers": result.n_taxa_with_markers,
        "hmm_files_used": list(result.hmm_files_used),
    }
    try:
        with cache_path.open("w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2)
    except OSError as exc:
        logger.warning("Failed to write hmm cache %s: %s", cache_path, exc)


def _fmt_evalue(e: float) -> str:
    """Format an E-value for the hmmsearch CLI without scientific-notation gotchas."""
    return f"{e:.3e}" if e < 1.0 else f"{e}"
