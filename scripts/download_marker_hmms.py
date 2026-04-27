#!/usr/bin/env python3
"""
Download and extract GTDB bac120/ar53 marker HMM profiles.

Strategy:
  1. Download full Pfam-A HMM library from EBI FTP (~340 MB gzipped)
  2. Download full TIGRFAMs HMM library from NCBI FTP (~100 MB gzipped)
  3. Extract the specific profiles needed for bac120/ar53 markers
     - If hmmfetch (HMMER) is available: use it for fast indexed extraction
     - Otherwise: pure-Python fallback that parses HMM blocks directly
  4. Concatenate into bac120_markers.hmm and ar53_markers.hmm
  5. Run hmmpress (if available) to build binary search indices

Usage:
    # Standard run (downloads source files, then cleans them up):
    python scripts/download_marker_hmms.py --output-dir data/marker_hmms/

    # Keep the large source files (Pfam-A.hmm, TIGRFAMs.LIB) after extraction:
    python scripts/download_marker_hmms.py --output-dir data/marker_hmms/ --keep-source

    # If you already have the source files locally:
    python scripts/download_marker_hmms.py \
        --pfam-hmm /path/to/Pfam-A.hmm \
        --tigrfam-hmm /path/to/TIGRFAMs_15.0_HMM.LIB \
        --output-dir data/marker_hmms/

    # Show marker counts and exit:
    python scripts/download_marker_hmms.py --dry-run
"""

import argparse
import gzip
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# GTDB bac120 marker IDs (120 families: 6 Pfam + 114 TIGRFAM)
# Source: GTDB-Tk stable branch, gtdbtk/config/config.py BAC120_MARKERS
# ---------------------------------------------------------------------------
BAC120_MARKERS = [
    "PF00380", "PF00410", "PF00466", "PF01025", "PF02576", "PF03726",
    "TIGR00006", "TIGR00019", "TIGR00020", "TIGR00029", "TIGR00043", "TIGR00054",
    "TIGR00059", "TIGR00061", "TIGR00064", "TIGR00065", "TIGR00082",
    "TIGR00083", "TIGR00084", "TIGR00086", "TIGR00088", "TIGR00090",
    "TIGR00092", "TIGR00095", "TIGR00115", "TIGR00116", "TIGR00152",
    "TIGR00153", "TIGR00158", "TIGR00166", "TIGR00168", "TIGR00186",
    "TIGR00194", "TIGR00250", "TIGR00337", "TIGR00344", "TIGR00362",
    "TIGR00382", "TIGR00392", "TIGR00396", "TIGR00409", "TIGR00414",
    "TIGR00416", "TIGR00420", "TIGR00431", "TIGR00435", "TIGR00436",
    "TIGR00442", "TIGR00443", "TIGR00445", "TIGR00456", "TIGR00459",
    "TIGR00460", "TIGR00468", "TIGR00472", "TIGR00487", "TIGR00496",
    "TIGR00575", "TIGR00631", "TIGR00634", "TIGR00635", "TIGR00643",
    "TIGR00663", "TIGR00717", "TIGR00755", "TIGR00810", "TIGR00922",
    "TIGR00928", "TIGR00959", "TIGR00963", "TIGR00981", "TIGR01009",
    "TIGR01011", "TIGR01017", "TIGR01021", "TIGR01029", "TIGR01032",
    "TIGR01039", "TIGR01044", "TIGR01059", "TIGR01063", "TIGR01066",
    "TIGR01071", "TIGR01079", "TIGR01082", "TIGR01087", "TIGR01128",
    "TIGR01130", "TIGR01145", "TIGR01146", "TIGR01164", "TIGR01169",
    "TIGR01171", "TIGR01302", "TIGR01341", "TIGR01391", "TIGR01393",
    "TIGR01510", "TIGR01632", "TIGR01951", "TIGR01952", "TIGR02012",
    "TIGR02013", "TIGR02027", "TIGR02075", "TIGR02191", "TIGR02273",
    "TIGR02350", "TIGR02386", "TIGR02397", "TIGR02432", "TIGR02729",
    "TIGR03263", "TIGR03594", "TIGR03625", "TIGR03632", "TIGR03654",
    "TIGR03723", "TIGR03725", "TIGR03953",
]

# ---------------------------------------------------------------------------
# GTDB ar53 marker IDs (53 families)
# Source: GTDB-Tk stable branch, gtdbtk/config/config.py AR53_MARKERS
# ---------------------------------------------------------------------------
AR53_MARKERS = [
    "PF01000", "PF01015", "PF01092", "PF01200", "PF01280",
    "TIGR00037", "TIGR00064", "TIGR00111", "TIGR00134", "TIGR00279",
    "TIGR00291", "TIGR00323", "TIGR00324", "TIGR00335", "TIGR00373",
    "TIGR00405", "TIGR00448", "TIGR00463", "TIGR00468", "TIGR00490",
    "TIGR00499", "TIGR00967", "TIGR01012", "TIGR01018", "TIGR01020",
    "TIGR01028", "TIGR01038", "TIGR01060", "TIGR01077", "TIGR01080",
    "TIGR01213", "TIGR01952", "TIGR02236", "TIGR02258", "TIGR02264",
    "TIGR02338", "TIGR02389", "TIGR02390", "TIGR03626", "TIGR03627",
    "TIGR03628", "TIGR03629", "TIGR03671", "TIGR03672", "TIGR03673",
    "TIGR03674", "TIGR03676", "TIGR03677", "TIGR03679", "TIGR03680",
    "TIGR03681", "TIGR03682", "TIGR03684",
]

# FTP source URLs
PFAM_URL = "http://ftp.ebi.ac.uk/pub/databases/Pfam/current_release/Pfam-A.hmm.gz"
TIGRFAM_URL = "https://ftp.ncbi.nlm.nih.gov/hmm/TIGRFAMs/release_15.0/TIGRFAMs_15.0_HMM.LIB.gz"


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest: Path, chunk_size: int = 1 << 20) -> None:
    """Download url to dest, printing a progress bar."""
    log.info(f"Downloading {url}")
    log.info(f"  → {dest}")

    tmp = dest.with_suffix(".tmp")
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            total = int(response.headers.get("Content-Length", 0))
            downloaded = 0
            start = time.time()
            with open(tmp, "wb") as fh:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    fh.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        pct = downloaded / total * 100
                        mb = downloaded / 1e6
                        elapsed = time.time() - start
                        rate = mb / elapsed if elapsed > 0 else 0
                        print(
                            f"\r  {pct:5.1f}%  {mb:.0f}/{total/1e6:.0f} MB"
                            f"  {rate:.1f} MB/s",
                            end="", flush=True,
                        )
        print()  # newline after progress
        tmp.rename(dest)
        log.info(f"  Download complete: {dest.stat().st_size / 1e6:.1f} MB")
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


def _decompress_gz(gz_path: Path, out_path: Path) -> None:
    """Decompress a .gz file."""
    log.info(f"Decompressing {gz_path.name} ...")
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    log.info(f"  Decompressed: {out_path.stat().st_size / 1e6:.1f} MB")


def _ensure_decompressed(gz_path: Path, out_path: Path) -> Path:
    """Download if needed, decompress if needed, return path to plain file."""
    if not out_path.exists():
        if not gz_path.exists():
            _download_with_progress(
                PFAM_URL if "Pfam" in gz_path.name else TIGRFAM_URL,
                gz_path,
            )
        _decompress_gz(gz_path, out_path)
    else:
        log.info(f"  Using existing: {out_path}")
    return out_path


# ---------------------------------------------------------------------------
# HMM extraction: hmmfetch (fast) or Python fallback (no HMMER needed)
# ---------------------------------------------------------------------------

def _hmmfetch_available() -> bool:
    return shutil.which("hmmfetch") is not None


def _extract_with_hmmfetch(hmm_lib: Path, ids: list[str], out_path: Path) -> int:
    """Use hmmfetch to extract profiles by ID. Returns count extracted."""
    # Index first if .h3i doesn't exist (makes hmmfetch fast).
    # Note: hmmfetch --index creates <file>.h3i alongside the library.
    idx = Path(str(hmm_lib) + ".h3i")
    if not idx.exists():
        log.info(f"  Indexing {hmm_lib.name} with hmmfetch --index ...")
        idx_result = subprocess.run(
            ["hmmfetch", "--index", str(hmm_lib)],
            capture_output=True,
        )
        if idx_result.returncode != 0:
            log.warning(
                f"  hmmfetch --index failed (rc={idx_result.returncode}): "
                f"{idx_result.stderr.decode(errors='replace')[:300].strip()}"
            )
            log.warning("  Falling back to Python extraction ...")
            return _extract_with_python(hmm_lib, set(ids), out_path)
    else:
        log.info(f"  Index already exists for {hmm_lib.name}")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".keys", delete=False) as kf:
        kf.write("\n".join(ids) + "\n")
        keys_file = kf.name

    try:
        result = subprocess.run(
            ["hmmfetch", "-f", str(hmm_lib), keys_file],
            capture_output=True, text=True, check=False,
        )
        if result.returncode != 0 and result.stderr:
            log.warning(f"  hmmfetch stderr: {result.stderr[:200]}")
            if not result.stdout.strip():
                log.warning("  No output from hmmfetch — falling back to Python extraction ...")
                return _extract_with_python(hmm_lib, set(ids), out_path)

        with open(out_path, "a") as fh:
            fh.write(result.stdout)

        # Count how many HMMs were actually written
        count = result.stdout.count("HMMER3")
        return count
    finally:
        os.unlink(keys_file)


def _extract_with_python(hmm_lib: Path, ids: set[str], out_path: Path) -> int:
    """
    Pure-Python HMM extraction. Reads the library line-by-line, collecting
    blocks between 'HMMER3/f' and '//' that match any ID in `ids`.

    HMM block format:
        HMMER3/f [3.3.2 | Nov 2020]
        NAME  TIGR00436
        ACC   TIGR00436.1
        ...
        //
    Matches on NAME or ACC field (stripping version suffix from ACC).
    """
    log.info(f"  Using Python fallback extraction from {hmm_lib.name} ...")
    extracted = 0
    in_block = False
    block_lines: list[str] = []
    block_matches = False

    with open(hmm_lib, "r", errors="replace") as src, \
         open(out_path, "a") as dst:

        for line in src:
            if line.startswith("HMMER3/f"):
                in_block = True
                block_lines = [line]
                block_matches = False
                continue

            if in_block:
                block_lines.append(line)

                # Check NAME field
                if line.startswith("NAME "):
                    name = line.split()[1].strip()
                    if name in ids:
                        block_matches = True

                # Check ACC field (strip version like .1 or .24)
                elif line.startswith("ACC "):
                    acc = line.split()[1].strip().split(".")[0]
                    if acc in ids:
                        block_matches = True

                # End of block
                if line.strip() == "//":
                    if block_matches:
                        dst.writelines(block_lines)
                        extracted += 1
                    in_block = False
                    block_lines = []
                    block_matches = False

    return extracted


def extract_profiles(
    hmm_lib: Path,
    ids: list[str],
    out_path: Path,
) -> int:
    """Extract profiles matching `ids` from `hmm_lib`, appending to `out_path`."""
    id_set = set(ids)
    if _hmmfetch_available():
        log.info(f"  hmmfetch available — fast extraction")
        return _extract_with_hmmfetch(hmm_lib, ids, out_path)
    else:
        log.info(f"  hmmfetch not found — using Python extraction (slower)")
        return _extract_with_python(hmm_lib, id_set, out_path)


# ---------------------------------------------------------------------------
# hmmpress
# ---------------------------------------------------------------------------

def _hmmpress(hmm_path: Path) -> None:
    if shutil.which("hmmpress") is None:
        log.warning(
            f"  hmmpress not found — skipping. Run manually after installing HMMER:\n"
            f"    hmmpress {hmm_path}"
        )
        return
    log.info(f"  Running hmmpress on {hmm_path.name} ...")
    subprocess.run(["hmmpress", str(hmm_path)], check=True, capture_output=True)
    log.info("  hmmpress complete.")


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------

def build_marker_hmms(
    output_dir: Path,
    pfam_hmm: Path | None = None,
    tigrfam_hmm: Path | None = None,
    keep_source: bool = False,
) -> dict[str, Path]:
    """
    Build bac120_markers.hmm and ar53_markers.hmm in output_dir.

    Returns dict with keys 'bac120' and 'ar53' pointing to the output files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "_source_cache"
    cache_dir.mkdir(exist_ok=True)

    # --- Resolve/download source files ---
    if pfam_hmm is None:
        pfam_gz = cache_dir / "Pfam-A.hmm.gz"
        pfam_hmm = cache_dir / "Pfam-A.hmm"
        if not pfam_hmm.exists():
            if not pfam_gz.exists():
                _download_with_progress(PFAM_URL, pfam_gz)
            _decompress_gz(pfam_gz, pfam_hmm)
        else:
            log.info(f"Using cached Pfam-A.hmm ({pfam_hmm.stat().st_size/1e6:.0f} MB)")
    else:
        pfam_hmm = Path(pfam_hmm)
        log.info(f"Using provided Pfam HMM: {pfam_hmm}")

    if tigrfam_hmm is None:
        tigr_gz = cache_dir / "TIGRFAMs_15.0_HMM.LIB.gz"
        tigrfam_hmm = cache_dir / "TIGRFAMs_15.0_HMM.LIB"
        if not tigrfam_hmm.exists():
            if not tigr_gz.exists():
                _download_with_progress(TIGRFAM_URL, tigr_gz)
            _decompress_gz(tigr_gz, tigrfam_hmm)
        else:
            log.info(f"Using cached TIGRFAMs.LIB ({tigrfam_hmm.stat().st_size/1e6:.0f} MB)")
    else:
        tigrfam_hmm = Path(tigrfam_hmm)
        log.info(f"Using provided TIGRFAM HMM: {tigrfam_hmm}")

    results = {}

    # --- Build bac120_markers.hmm ---
    bac120_out = output_dir / "bac120_markers.hmm"
    if bac120_out.exists():
        log.info(f"bac120_markers.hmm already exists — skipping. Delete to rebuild.")
        results["bac120"] = bac120_out
    else:
        log.info("=== Building bac120_markers.hmm ===")
        bac120_pfam = [m for m in BAC120_MARKERS if m.startswith("PF")]
        bac120_tigr = [m for m in BAC120_MARKERS if m.startswith("TIGR")]
        log.info(f"  {len(bac120_pfam)} Pfam + {len(bac120_tigr)} TIGRFAM markers")

        # Clear output file before appending
        bac120_out.write_text("")

        n_pfam = extract_profiles(pfam_hmm, bac120_pfam, bac120_out)
        log.info(f"  Extracted {n_pfam}/{len(bac120_pfam)} Pfam profiles")

        n_tigr = extract_profiles(tigrfam_hmm, bac120_tigr, bac120_out)
        log.info(f"  Extracted {n_tigr}/{len(bac120_tigr)} TIGRFAM profiles")

        total = n_pfam + n_tigr
        log.info(f"  Total bac120 profiles extracted: {total}/{len(BAC120_MARKERS)}")
        if total < len(BAC120_MARKERS) * 0.9:
            log.warning(
                f"  Only {total}/{len(BAC120_MARKERS)} profiles found. "
                f"This may indicate version mismatches in the source HMM libraries."
            )

        _hmmpress(bac120_out)
        results["bac120"] = bac120_out

    # --- Build ar53_markers.hmm ---
    ar53_out = output_dir / "ar53_markers.hmm"
    if ar53_out.exists():
        log.info(f"ar53_markers.hmm already exists — skipping. Delete to rebuild.")
        results["ar53"] = ar53_out
    else:
        log.info("=== Building ar53_markers.hmm ===")
        ar53_pfam = [m for m in AR53_MARKERS if m.startswith("PF")]
        ar53_tigr = [m for m in AR53_MARKERS if m.startswith("TIGR")]
        log.info(f"  {len(ar53_pfam)} Pfam + {len(ar53_tigr)} TIGRFAM markers")

        ar53_out.write_text("")

        n_pfam = extract_profiles(pfam_hmm, ar53_pfam, ar53_out)
        log.info(f"  Extracted {n_pfam}/{len(ar53_pfam)} Pfam profiles")

        n_tigr = extract_profiles(tigrfam_hmm, ar53_tigr, ar53_out)
        log.info(f"  Extracted {n_tigr}/{len(ar53_tigr)} TIGRFAM profiles")

        total = n_pfam + n_tigr
        log.info(f"  Total ar53 profiles extracted: {total}/{len(AR53_MARKERS)}")

        _hmmpress(ar53_out)
        results["ar53"] = ar53_out

    # --- Cleanup source files ---
    if not keep_source:
        log.info("Cleaning up source cache (use --keep-source to retain) ...")
        shutil.rmtree(cache_dir, ignore_errors=True)
    else:
        log.info(f"Source files retained at: {cache_dir}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output-dir", default="data/marker_hmms",
        help="Directory to write bac120_markers.hmm and ar53_markers.hmm (default: data/marker_hmms/)",
    )
    parser.add_argument(
        "--pfam-hmm", default=None,
        help="Path to existing decompressed Pfam-A.hmm (skip download)",
    )
    parser.add_argument(
        "--tigrfam-hmm", default=None,
        help="Path to existing decompressed TIGRFAMs_15.0_HMM.LIB (skip download)",
    )
    parser.add_argument(
        "--keep-source", action="store_true",
        help="Keep downloaded source HMM libraries after extraction (~440 MB)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print marker counts and exit without downloading anything",
    )
    args = parser.parse_args()

    if args.dry_run:
        bac120_pfam = [m for m in BAC120_MARKERS if m.startswith("PF")]
        bac120_tigr = [m for m in BAC120_MARKERS if m.startswith("TIGR")]
        ar53_pfam = [m for m in AR53_MARKERS if m.startswith("PF")]
        ar53_tigr = [m for m in AR53_MARKERS if m.startswith("TIGR")]
        print(f"bac120: {len(BAC120_MARKERS)} markers ({len(bac120_pfam)} Pfam, {len(bac120_tigr)} TIGRFAM)")
        print(f"ar53:   {len(AR53_MARKERS)} markers ({len(ar53_pfam)} Pfam, {len(ar53_tigr)} TIGRFAM)")
        print(f"\nPfam source:    {PFAM_URL}")
        print(f"TIGRFAM source: {TIGRFAM_URL}")
        print(f"\nOutput dir: {args.output_dir}")
        print(f"\nhmmfetch available: {_hmmfetch_available()}")
        print(f"hmmpress available: {shutil.which('hmmpress') is not None}")
        return

    results = build_marker_hmms(
        output_dir=Path(args.output_dir),
        pfam_hmm=args.pfam_hmm,
        tigrfam_hmm=args.tigrfam_hmm,
        keep_source=args.keep_source,
    )

    print("\n=== Done ===")
    for name, path in results.items():
        size_kb = path.stat().st_size / 1024 if path.exists() else 0
        print(f"  {name}: {path}  ({size_kb:.0f} KB)")
    print(f"\nNext step — install HMMER if not already installed:")
    print(f"  conda install -c bioconda hmmer")
    print(f"\nThen run the pipeline with:")
    print(f"  python main.py run --input <mzML> --db <fasta> --no-llm \\")
    print(f"    --marker-correction --hmm-profile-dir {args.output_dir}")


if __name__ == "__main__":
    main()
