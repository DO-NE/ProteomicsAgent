#!/usr/bin/env python3
"""Direct pipeline: mzML -> Comet -> pepXML -> local taxon inference -> TSV.

Bypasses the LLM orchestrator and TPP entirely.

Usage:
    python run_direct.py \
        --input  data/LFQ_Orbitrap_DDA_Condition_A_Sample_Alpha_01.mzML \
        --db     data/PXD028735.fasta \
        --comet  /jhcnas4/gzr/jacobkwak/ProteomicsAgent/models/Comet/comet.2021010.linux.exe

Optional flags:
    --output-dir  ./output_direct     (default)
    --algorithm   local_db            (local_db or unipept_api)
    --xcorr       2.0                 (Comet xcorr cutoff; lower if too few peptides)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path

from taxon.registry import TaxonRegistry
from visualization.report import export_tsv


# ---------------------------------------------------------------------------
# Stage 1: format conversion (copy mzML into run dir)
# ---------------------------------------------------------------------------

def run_format_conversion(input_mzml: Path, run_dir: Path) -> Path:
    outdir = run_dir / "mzml"
    outdir.mkdir(parents=True, exist_ok=True)
    target = outdir / input_mzml.name
    shutil.copy2(input_mzml, target)
    print(f"[format_conversion] {input_mzml.name} -> {target}")
    return target


# ---------------------------------------------------------------------------
# Stage 2: peptide ID via Comet
# ---------------------------------------------------------------------------

def run_comet(mzml: Path, fasta: Path, comet_path: str, run_dir: Path) -> Path:
    params_dir = run_dir / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    param_file = params_dir / "comet.params"

    param_file.write_text(
        "\n".join([
            f"database_name = {fasta}",
            "output_pepxmlfile = 1",
            "output_txtfile = 0",
            "output_percolatorfile = 0",
            "peptide_mass_tolerance = 10.0",
            "peptide_mass_units = 2",       # 2 = ppm
            "isotope_error = 0",
            "num_enzyme_termini = 2",
            "missed_cleavages = 1",
            "minimum_length = 7",
            "maximum_length = 50",
            "max_fragment_charge = 3",
            "max_precursor_charge = 6",
            "search_enzyme_number = 1",
            "sample_enzyme_number = 1",
            "num_threads = 0",              # 0 = use all cores
            "decoy_search = 1",
            "decoy_prefix = DECOY_",
            "fragment_bin_tol = 0.02",
            "fragment_bin_offset = 0.0",
            "",
            "[COMET_ENZYME_INFO]",
            "0.  No_enzyme              0      -         -",
            "1.  Trypsin                1      KR        P",
            "2.  Trypsin/P             1      KR        -",
            "3.  Lys_C                 1      K         P",
            "4.  Lys_N                 0      K         -",
            "5.  Arg_C                 1      R         P",
            "6.  Asp_N                 0      D         -",
            "7.  CNBr                  1      M         -",
            "8.  Glu_C                 1      DE        P",
            "9.  PepsinA               1      FL        P",
            "10. Chymotrypsin          1      FWYL      P",
        ]),
        encoding="utf-8",
    )

    cmd = [comet_path, f"-P{param_file}", str(mzml)]
    print(f"[peptide_id] Running Comet: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0:
        print(f"[ERROR] Comet failed (rc={result.returncode}):\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    # Comet writes pepXML alongside the input mzML (not in params_dir)
    pepxml = mzml.with_suffix(".pep.xml")
    if not pepxml.exists():
        print(f"[ERROR] Expected pepXML not found at {pepxml}", file=sys.stderr)
        print(f"        Comet stdout:\n{result.stdout}", file=sys.stderr)
        sys.exit(1)

    print(f"[peptide_id] pepXML produced: {pepxml}")
    return pepxml


# ---------------------------------------------------------------------------
# Stage 3: extract peptides from pepXML (no TPP/PeptideProphet)
# ---------------------------------------------------------------------------

def extract_peptides(pepxml: Path, xcorr_cutoff: float) -> list[str]:
    tree = ET.parse(pepxml)
    root = tree.getroot()
    ns = root.tag.split("}")[0].strip("{") if root.tag.startswith("{") else ""

    def q(name: str) -> str:
        return f"{{{ns}}}{name}" if ns else name

    seen: set[str] = set()
    for hit in root.iter(q("search_hit")):
        if int(hit.attrib.get("hit_rank", "1")) != 1:
            continue
        peptide = hit.attrib.get("peptide", "")
        if not peptide:
            continue
        xcorr = 0.0
        for score in hit.iter(q("search_score")):
            if score.attrib.get("name") == "xcorr":
                xcorr = float(score.attrib.get("value", "0"))
                break
        if xcorr >= xcorr_cutoff:
            seen.add(peptide)

    peptides = sorted(seen)
    print(f"[extract_peptides] {len(peptides)} unique peptides passing xcorr >= {xcorr_cutoff}")
    return peptides


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct metaproteomics pipeline (no LLM, no TPP)."
    )
    parser.add_argument("--input", required=True, type=Path, help="Input .mzML file")
    parser.add_argument("--db", required=True, type=Path, help="FASTA protein database")
    parser.add_argument(
        "--comet",
        default="/jhcnas4/gzr/jacobkwak/ProteomicsAgent/models/Comet/comet.2021010.linux.exe",
        help="Path to Comet executable",
    )
    parser.add_argument("--output-dir", default="./output_direct", type=Path)
    parser.add_argument(
        "--algorithm",
        default="local_db",
        choices=["local_db", "unipept_api"],
        help="Taxon inference algorithm",
    )
    parser.add_argument(
        "--xcorr",
        type=float,
        default=2.0,
        help="Comet xcorr score cutoff (lower to 1.5 if too few peptides pass)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[ERROR] Input mzML not found: {args.input}", file=sys.stderr)
        sys.exit(1)
    if not args.db.exists():
        print(f"[ERROR] FASTA database not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    run_id = str(uuid.uuid4())[:8]
    run_dir = args.output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Run ID: {run_id}")
    print(f"Run dir: {run_dir}\n")

    # Stage 1
    mzml = run_format_conversion(args.input, run_dir)

    # Stage 2
    pepxml = run_comet(mzml, args.db, args.comet, run_dir)

    # Stage 3
    peptides = extract_peptides(pepxml, xcorr_cutoff=args.xcorr)
    if not peptides:
        print(
            "[WARNING] No peptides passed the score filter. "
            "Try --xcorr 1.5 or lower.",
            file=sys.stderr,
        )

    # Stage 4: taxon inference
    registry = TaxonRegistry()
    config = {"database_path": str(args.db)}
    print(f"\n[taxon] Running '{args.algorithm}' on {len(peptides)} peptides ...")
    results = registry.run(args.algorithm, peptides, config)
    print(f"[taxon] {len(results)} taxa identified.\n")

    # Stage 5: write TSV
    taxon_dir = run_dir / "taxon"
    tsv_path = export_tsv(results, taxon_dir, "results.tsv")
    print(f"[DONE] Results written to: {tsv_path}\n")

    # Print top 10 to stdout
    header = f"{'Taxon':<45} {'ID':<12} {'Rank':<10} {'Abundance%':>10} {'Peptides':>8}"
    print(header)
    print("-" * len(header))
    for r in results[:10]:
        print(
            f"{r.taxon_name:<45} {r.taxon_id:<12} {r.rank:<10} "
            f"{r.abundance * 100:>9.4f}% {r.peptide_count:>8}"
        )


if __name__ == "__main__":
    main()
