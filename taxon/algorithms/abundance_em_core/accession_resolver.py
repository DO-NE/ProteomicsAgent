"""Resolve bare UniProt accession FASTA headers to organism names.

Two-phase pipeline used as a preprocessing step for
:func:`build_mapping_matrix`:

1. *FASTA-internal inference*: when the same UniProt accession appears in
   the database with both a bare header (``>Q2FYC6``) and a fully annotated
   header (``>sp|Q2FYC6|HEM3_STAA8 ... OS=Staphylococcus aureus``), reuse
   the organism parsed from the annotated entry.
2. *UniProt REST lookup*: for accessions still missing after phase 1, batch
   query the UniProt ID-mapping API, cache results to a TSV next to the
   FASTA, and degrade gracefully when the API is unreachable.
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Iterable

logger = logging.getLogger(__name__)

# Canonical UniProt accession pattern, as documented at
# https://www.uniprot.org/help/accession_numbers
UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]"
    r"|[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})$"
)

_UNIPROT_API_BASE = "https://rest.uniprot.org"
_BATCH_SIZE = 100
_REQUEST_TIMEOUT = 30
_MAX_RETRIES = 3
_POLL_INTERVAL = 2.0
_MAX_POLL_SECONDS = 180


def extract_uniprot_accession(first_token: str) -> str | None:
    """Return a UniProt accession embedded in *first_token*, or ``None``.

    Handles both bare accessions (``Q2FYC6``) and the standard
    ``sp|ACC|NAME`` / ``tr|ACC|NAME`` encodings.
    """
    if not first_token:
        return None
    for part in first_token.split("|"):
        if UNIPROT_ACCESSION_RE.match(part):
            return part
    return None


def resolve_accessions(
    fasta_path: str | Path,
    unresolved_accessions: Iterable[str],
    known_accession_organism: dict[str, str],
    use_api: bool = True,
) -> dict[str, str]:
    """Return a ``{accession: organism_name}`` map for *unresolved_accessions*.

    Parameters
    ----------
    fasta_path :
        Used to locate the adjacent cache file
        ``<fasta_path>.accession_cache.tsv``.
    unresolved_accessions :
        UniProt accessions whose FASTA headers did not yield an organism.
    known_accession_organism :
        Map of ``accession -> organism_name`` gathered from FASTA entries
        that *did* parse successfully. Used for phase 1.
    use_api :
        When ``False``, phase 2 is skipped entirely.
    """
    unresolved = {a for a in unresolved_accessions if a}
    if not unresolved:
        return {}

    resolved: dict[str, str] = {}

    # Phase 1: FASTA-internal inference.
    for acc in list(unresolved):
        organism = known_accession_organism.get(acc)
        if organism:
            resolved[acc] = organism
            unresolved.discard(acc)

    if not unresolved:
        return resolved

    cache_path = Path(str(fasta_path) + ".accession_cache.tsv")
    cached = _load_cache(cache_path)
    if cached:
        for acc in list(unresolved):
            if acc in cached:
                resolved[acc] = cached[acc]
                unresolved.discard(acc)

    if not unresolved or not use_api:
        return resolved

    # Phase 2: UniProt batch API.
    try:
        api_results = _uniprot_batch_lookup(sorted(unresolved))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "UniProt accession lookup failed (%s); %d accession(s) will "
            "remain unclassified",
            exc,
            len(unresolved),
        )
        return resolved

    if api_results:
        resolved.update(api_results)
        _append_cache(cache_path, api_results)

    missing = unresolved - set(api_results)
    if missing:
        logger.info(
            "UniProt API returned no organism for %d of %d queried accessions",
            len(missing),
            len(unresolved),
        )
    return resolved


# ----------------------------------------------------------------- cache I/O


def _load_cache(path: Path) -> dict[str, str]:
    if not path.is_file() or path.stat().st_size == 0:
        return {}
    out: dict[str, str] = {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip("\n").split("\t")
                if len(parts) >= 2 and parts[0]:
                    out[parts[0]] = parts[1]
    except OSError as exc:
        logger.warning("Failed to read accession cache %s: %s", path, exc)
        return {}
    logger.info("Loaded %d cached accession -> organism entries from %s",
                len(out), path)
    return out


def _append_cache(path: Path, entries: dict[str, str]) -> None:
    if not entries:
        return
    try:
        exists = path.is_file() and path.stat().st_size > 0
        with path.open("a", encoding="utf-8") as fh:
            if not exists:
                fh.write("accession\torganism_name\n")
            for acc, org in entries.items():
                fh.write(f"{acc}\t{org}\n")
    except OSError as exc:
        logger.warning("Failed to write accession cache %s: %s", path, exc)


# ------------------------------------------------------------ UniProt API


def _uniprot_batch_lookup(accessions: list[str]) -> dict[str, str]:
    """Return ``{accession: organism_name}`` for IDs resolved by UniProt."""
    import requests  # local import — optional at call time

    combined: dict[str, str] = {}
    for start in range(0, len(accessions), _BATCH_SIZE):
        chunk = accessions[start : start + _BATCH_SIZE]
        chunk_result = _run_single_batch(requests, chunk)
        combined.update(chunk_result)
    return combined


def _run_single_batch(requests_mod, chunk: list[str]) -> dict[str, str]:
    job_id = _submit_idmapping(requests_mod, chunk)
    if not job_id:
        return {}
    if not _await_job(requests_mod, job_id):
        logger.warning("UniProt job %s did not complete within %ds",
                       job_id, _MAX_POLL_SECONDS)
        return {}
    return _fetch_results(requests_mod, job_id)


def _submit_idmapping(requests_mod, chunk: list[str]) -> str | None:
    url = f"{_UNIPROT_API_BASE}/idmapping/run"
    data = {
        "from": "UniProtKB_AC-ID",
        "to": "UniProtKB",
        "ids": ",".join(chunk),
    }
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            resp = requests_mod.post(url, data=data, timeout=_REQUEST_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            job_id = payload.get("jobId")
            if job_id:
                return job_id
            logger.warning("UniProt submit response lacked jobId: %s", payload)
            return None
        except Exception as exc:  # noqa: BLE001
            if attempt == _MAX_RETRIES:
                raise
            logger.info("UniProt submit retry %d/%d after error: %s",
                        attempt, _MAX_RETRIES, exc)
            time.sleep(2.0 * attempt)
    return None


def _await_job(requests_mod, job_id: str) -> bool:
    url = f"{_UNIPROT_API_BASE}/idmapping/status/{job_id}"
    deadline = time.monotonic() + _MAX_POLL_SECONDS
    while time.monotonic() < deadline:
        try:
            resp = requests_mod.get(url, timeout=_REQUEST_TIMEOUT,
                                    allow_redirects=False)
        except Exception as exc:  # noqa: BLE001
            logger.info("UniProt status poll error: %s", exc)
            time.sleep(_POLL_INTERVAL)
            continue
        # 303 See Other => job done and results available at redirect target.
        if resp.status_code in (200, 303):
            try:
                payload = resp.json()
            except ValueError:
                return True
            status = payload.get("jobStatus")
            if status in (None, "FINISHED"):
                return True
            if status in ("ERROR", "FAILED"):
                logger.warning("UniProt job %s ended with status=%s",
                               job_id, status)
                return False
        time.sleep(_POLL_INTERVAL)
    return False


def _fetch_results(requests_mod, job_id: str) -> dict[str, str]:
    url = f"{_UNIPROT_API_BASE}/idmapping/uniprotkb/results/{job_id}"
    params = {
        "format": "tsv",
        "fields": "accession,organism_name",
        "size": "500",
    }
    out: dict[str, str] = {}
    next_url: str | None = url
    next_params = params
    pages = 0
    while next_url and pages < 50:
        resp = requests_mod.get(next_url, params=next_params,
                                timeout=_REQUEST_TIMEOUT)
        resp.raise_for_status()
        _parse_idmapping_tsv(resp.text, out)
        next_url = _next_link(resp.headers.get("Link", ""))
        next_params = None  # cursor URL already contains params
        pages += 1
    return out


def _parse_idmapping_tsv(text: str, out: dict[str, str]) -> None:
    for i, line in enumerate(text.splitlines()):
        if i == 0 and line.lower().startswith("from\t"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        queried, _entry, organism = parts[0], parts[1], parts[2]
        organism = organism.strip()
        if queried and organism and queried not in out:
            out[queried] = _strip_strain_details(organism)


def _strip_strain_details(name: str) -> str:
    return re.sub(r"\s*\(.*?\)\s*$", "", name).strip() or name


def _next_link(link_header: str) -> str | None:
    # UniProt pages with standard RFC 5988 Link headers: `<url>; rel="next"`.
    for part in link_header.split(","):
        segments = part.split(";")
        if len(segments) < 2:
            continue
        url_seg = segments[0].strip().strip("<>").strip()
        for attr in segments[1:]:
            if attr.strip().lower() == 'rel="next"':
                return url_seg
    return None
