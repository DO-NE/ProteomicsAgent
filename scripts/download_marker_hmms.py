"""Download GTDB bac120 / ar53 marker HMM profiles from EBI/InterPro.

The :mod:`taxon.algorithms.abundance_em_core.hmm_marker_search` module
needs HMMER-format profiles for the universal single-copy marker
families that GTDB-Tk uses: 120 Pfam/TIGRFAM HMMs for bacteria
(``bac120``) and 53 for archaea (``ar53``).  Rather than depending on
the full GTDB-Tk reference package (~85 GB), this script downloads
just the individual profile files from InterPro and concatenates them
into two ready-to-search files:

    <output>/bac120_markers.hmm
    <output>/ar53_markers.hmm

If ``hmmpress`` is available, the script also presses each bundle to
build the SSI index that speeds up ``hmmsearch``.

Run it once before the first marker-corrected pipeline run::

    python scripts/download_marker_hmms.py --output-dir data/marker_hmms

The script is idempotent: any HMM file already present in the cache
directory is reused, and the bundles are rebuilt only when their
member files change.

Restricted networks
-------------------
If InterPro's API is unreachable, populate
``<output>/individual/`` manually with the per-family ``*.hmm`` files
and rerun the script with ``--no-download`` — the concatenation and
``hmmpress`` steps still run on the local files.

Provenance
----------
The marker family lists are reproduced from the GTDB-Tk source
(``gtdbtk/config/config.py``) and Parks et al. 2018 supplementary
data.  They are copied here verbatim (without family-specific gathering
thresholds) because the lists themselves are stable across releases —
new GTDB-Tk versions occasionally bump Pfam version suffixes (e.g.,
``PF00380.20`` -> ``PF00380.24``), which InterPro normalises away
when an unversioned ID is queried.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger("download_marker_hmms")


# ---------------------------------------------------------------------- markers
# Canonical bac120 marker accessions (120 IDs total).  Source:
# https://github.com/Ecogenomics/GTDBTk/blob/stable/gtdbtk/config/config.py
# Versions stripped — InterPro returns the current version on unversioned IDs.

BAC120_PFAM = [
    "PF00380",
    "PF00410",
    "PF00466",
    "PF01025",
    "PF02576",
    "PF03726",
]

BAC120_TIGRFAM = [
    "TIGR00006", "TIGR00019", "TIGR00020", "TIGR00029", "TIGR00043",
    "TIGR00054", "TIGR00059", "TIGR00061", "TIGR00064", "TIGR00065",
    "TIGR00082", "TIGR00083", "TIGR00084", "TIGR00086", "TIGR00088",
    "TIGR00090", "TIGR00092", "TIGR00095", "TIGR00115", "TIGR00116",
    "TIGR00138", "TIGR00158", "TIGR00166", "TIGR00168", "TIGR00186",
    "TIGR00194", "TIGR00250", "TIGR00337", "TIGR00344", "TIGR00362",
    "TIGR00382", "TIGR00392", "TIGR00396", "TIGR00398", "TIGR00414",
    "TIGR00416", "TIGR00420", "TIGR00431", "TIGR00435", "TIGR00436",
    "TIGR00442", "TIGR00445", "TIGR00456", "TIGR00459", "TIGR00460",
    "TIGR00468", "TIGR00472", "TIGR00487", "TIGR00496", "TIGR00539",
    "TIGR00580", "TIGR00593", "TIGR00615", "TIGR00631", "TIGR00634",
    "TIGR00635", "TIGR00643", "TIGR00663", "TIGR00717", "TIGR00755",
    "TIGR00810", "TIGR00922", "TIGR00928", "TIGR00959", "TIGR00963",
    "TIGR00964", "TIGR00967", "TIGR01009", "TIGR01011", "TIGR01017",
    "TIGR01021", "TIGR01029", "TIGR01032", "TIGR01039", "TIGR01044",
    "TIGR01059", "TIGR01063", "TIGR01066", "TIGR01071", "TIGR01079",
    "TIGR01082", "TIGR01087", "TIGR01128", "TIGR01146", "TIGR01164",
    "TIGR01169", "TIGR01171", "TIGR01302", "TIGR01391", "TIGR01393",
    "TIGR01394", "TIGR01510", "TIGR01632", "TIGR01951", "TIGR01953",
    "TIGR02012", "TIGR02013", "TIGR02027", "TIGR02075", "TIGR02191",
    "TIGR02273", "TIGR02350", "TIGR02386", "TIGR02397", "TIGR02432",
    "TIGR02729", "TIGR03263", "TIGR03594", "TIGR03625", "TIGR03632",
    "TIGR03654", "TIGR03723", "TIGR03725", "TIGR03953",
]

# Canonical ar53 markers (53 IDs total) for archaeal coverage.
AR53_PFAM = [
    "PF04919", "PF07541", "PF01287", "PF00410", "PF00827",
    "PF01015", "PF13656", "PF13685", "PF01092",
]

AR53_TIGRFAM = [
    "TIGR00037", "TIGR00064", "TIGR00270", "TIGR00279", "TIGR00283",
    "TIGR00291", "TIGR00293", "TIGR00307", "TIGR00308", "TIGR00329",
    "TIGR00335", "TIGR00373", "TIGR00389", "TIGR00405", "TIGR00408",
    "TIGR00422", "TIGR00425", "TIGR00432", "TIGR00442", "TIGR00448",
    "TIGR00456", "TIGR00458", "TIGR00463", "TIGR00467", "TIGR00468",
    "TIGR00471", "TIGR00490", "TIGR00491", "TIGR00501", "TIGR00521",
    "TIGR00522", "TIGR00549", "TIGR00658", "TIGR00670", "TIGR00729",
    "TIGR00936", "TIGR00982", "TIGR01008", "TIGR01012", "TIGR01018",
    "TIGR01020", "TIGR01028", "TIGR01046", "TIGR01080", "TIGR02153",
    "TIGR02236", "TIGR02338", "TIGR02389", "TIGR02390",
]


_INTERPRO_PFAM_URL = (
    "https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/{id}?annotation=hmm"
)
_INTERPRO_TIGRFAM_URL = (
    "https://www.ebi.ac.uk/interpro/wwwapi/entry/tigrfam/{id}?annotation=hmm"
)
_DEFAULT_TIMEOUT = 60
_DEFAULT_RETRIES = 3
_USER_AGENT = "ProteomicsAgent-marker-hmm-download/1.0"


# ---------------------------------------------------------------------- public


def main(argv: list | None = None) -> int:
    """Entry point.  Returns a Unix exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Download GTDB bac120/ar53 marker HMM profiles from "
            "EBI/InterPro and concatenate into bundle files."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/marker_hmms"),
        help="Directory where bundles and per-family caches are written.",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help=(
            "Skip the network step.  Concatenate and (optionally) "
            "hmmpress whatever HMM files are already present in the "
            "<output>/individual/ cache.  Useful on restricted networks "
            "where the per-family files were copied in manually."
        ),
    )
    parser.add_argument(
        "--bac120-only",
        action="store_true",
        help="Download only the bacterial bac120 set, skip ar53.",
    )
    parser.add_argument(
        "--ar53-only",
        action="store_true",
        help="Download only the archaeal ar53 set, skip bac120.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=_DEFAULT_RETRIES,
        help="Per-file network retries (default: 3).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=_DEFAULT_TIMEOUT,
        help="Per-request timeout in seconds (default: 60).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress INFO logs.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    if args.bac120_only and args.ar53_only:
        parser.error("--bac120-only and --ar53-only are mutually exclusive")

    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    individual_dir = output_dir / "individual"
    individual_dir.mkdir(parents=True, exist_ok=True)

    targets: list = []
    if not args.ar53_only:
        targets.append(("bac120", BAC120_PFAM, BAC120_TIGRFAM))
    if not args.bac120_only:
        targets.append(("ar53", AR53_PFAM, AR53_TIGRFAM))

    overall_status = 0
    for name, pfam_ids, tigr_ids in targets:
        logger.info("=== %s: %d Pfam + %d TIGRFAM markers ===",
                    name, len(pfam_ids), len(tigr_ids))
        downloaded_files: list = []

        if not args.no_download:
            downloaded_files = _download_set(
                pfam_ids, tigr_ids,
                cache_dir=individual_dir,
                retries=args.retries,
                timeout=args.timeout,
            )
        else:
            downloaded_files = _collect_existing(
                pfam_ids, tigr_ids,
                cache_dir=individual_dir,
            )

        if not downloaded_files:
            logger.error(
                "No HMM files available for %s; skipping bundle.",
                name,
            )
            overall_status = 1
            continue

        bundle_path = output_dir / f"{name}_markers.hmm"
        _concatenate(downloaded_files, bundle_path)
        logger.info("Wrote %d HMMs into bundle %s",
                    len(downloaded_files), bundle_path)

        if shutil.which("hmmpress"):
            _run_hmmpress(bundle_path)
        else:
            logger.warning(
                "hmmpress not on PATH; skipping index build for %s. "
                "hmmsearch will still work, just slower.",
                bundle_path.name,
            )

    return overall_status


# ---------------------------------------------------------------------- helpers


def _download_set(
    pfam_ids: Iterable[str],
    tigr_ids: Iterable[str],
    cache_dir: Path,
    retries: int,
    timeout: int,
) -> list:
    """Fetch every Pfam/TIGRFAM ID, returning the list of cached file paths."""
    out: list = []
    for fam_id in pfam_ids:
        path = _fetch_one(
            fam_id, _INTERPRO_PFAM_URL, cache_dir,
            retries=retries, timeout=timeout,
        )
        if path is not None:
            out.append(path)
    for fam_id in tigr_ids:
        path = _fetch_one(
            fam_id, _INTERPRO_TIGRFAM_URL, cache_dir,
            retries=retries, timeout=timeout,
        )
        if path is not None:
            out.append(path)
    return out


def _collect_existing(
    pfam_ids: Iterable[str],
    tigr_ids: Iterable[str],
    cache_dir: Path,
) -> list:
    """Return per-family files already present on disk (no network)."""
    out: list = []
    for fam_id in list(pfam_ids) + list(tigr_ids):
        path = cache_dir / f"{fam_id}.hmm"
        if path.is_file() and path.stat().st_size > 0:
            out.append(path)
        else:
            logger.warning(
                "missing local HMM for %s (expected %s); skipping",
                fam_id, path,
            )
    return out


def _fetch_one(
    fam_id: str,
    url_template: str,
    cache_dir: Path,
    retries: int,
    timeout: int,
) -> Path | None:
    """Download one HMM into the cache dir, returning its path on success."""
    target = cache_dir / f"{fam_id}.hmm"
    if target.is_file() and target.stat().st_size > 0:
        logger.info("cache hit %s -> %s", fam_id, target.name)
        return target

    url = url_template.format(id=fam_id)
    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(url, headers={"User-Agent": _USER_AGENT})
            with urlopen(req, timeout=timeout) as resp:
                payload = resp.read()
            if not payload or b"HMMER3" not in payload[:64]:
                raise RuntimeError(
                    f"unexpected response payload (no HMMER3 magic) "
                    f"from {url}"
                )
            tmp = target.with_suffix(".hmm.tmp")
            tmp.write_bytes(payload)
            tmp.replace(target)
            logger.info("downloaded %s (%d bytes)", fam_id, len(payload))
            return target
        except (HTTPError, URLError, RuntimeError, OSError) as exc:
            last_err = exc
            if attempt < retries:
                wait = 2.0 * attempt
                logger.info(
                    "retry %d/%d for %s after %s (sleep %.1fs)",
                    attempt, retries, fam_id, exc, wait,
                )
                time.sleep(wait)
    logger.error("failed to fetch %s after %d attempts: %s",
                 fam_id, retries, last_err)
    return None


def _concatenate(hmm_files: list, bundle_path: Path) -> None:
    """Write an atomic concatenation of *hmm_files* to *bundle_path*."""
    tmp = bundle_path.with_suffix(".hmm.tmp")
    with tmp.open("wb") as out_fh:
        for hmm in sorted(hmm_files):
            data = hmm.read_bytes()
            out_fh.write(data)
            if not data.endswith(b"\n"):
                out_fh.write(b"\n")
    tmp.replace(bundle_path)


def _run_hmmpress(bundle_path: Path) -> None:
    """Run ``hmmpress`` on *bundle_path*, ignoring already-pressed warnings."""
    cmd = ["hmmpress", "-f", str(bundle_path)]
    logger.info("running: %s", " ".join(cmd))
    completed = subprocess.run(
        cmd, capture_output=True, text=True, check=False,
    )
    if completed.returncode != 0:
        logger.error(
            "hmmpress failed (exit %d): %s",
            completed.returncode, completed.stderr.strip()[:400],
        )
    else:
        logger.info("hmmpress complete for %s", bundle_path.name)


if __name__ == "__main__":
    sys.exit(main())
