#!/usr/bin/env python3
"""iBAQ analysis across PXD006118 mock-community mzML files.

For every mzML in the data dir starting with "P" (equal protein amount) or
"C" (equal cell number):
    mzML -> Comet -> pepXML -> IbaqPlugin (raw_ibaq) -> per-sample CSV.

Then aggregates per-sample estimates, compares to ground truth (derived
from the bundled Composition_Of_* .tab files + FASTA header species
assignments), computes R2/RMSE/L1/Pearson metrics, and renders charts.

Run from the ProteomicsAgent repo root so `taxon.*` imports resolve.

Usage
-----
    python run_ibaq_mock_comm.py              # full run (all P* and C*)
    python run_ibaq_mock_comm.py --limit 1    # sanity-check on 1 file
    python run_ibaq_mock_comm.py --jobs 4     # parallel Comet (threads/n)
    python run_ibaq_mock_comm.py --skip-comet # only re-aggregate + charts
"""

from __future__ import annotations

import argparse
import csv
import logging
import multiprocessing as mp
import os
import re
import shutil
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from taxon.algorithms.ibaq_core.fasta_utils import (  # noqa: E402
    _normalize_species, extract_accession, build_theoretical_peptide_counts,
)
from taxon.algorithms.ibaq_core.psm_table import build_psm_table  # noqa: E402
from taxon.algorithms.ibaq_core.methods import raw_ibaq, ibaq_em, ibaq_theoretical  # noqa: E402
from taxon.algorithms.abundance_em_core.pepxml_parser import parse_pepxml  # noqa: E402

# Methods to run per sample.
#   raw_ibaq         - intensity / N_observed_peptides (baseline; not "true" iBAQ)
#   ibaq_theoretical - intensity / N_theoretical_tryptic_peptides   <-- Schwanhäusser 2011 iBAQ
#   ibaq_em          - EM shared-peptide reassignment, then iBAQ with N_observed
IBAQ_METHOD_NAMES = ("raw_ibaq", "ibaq_theoretical", "ibaq_em")
# Intensity signals to feed into iBAQ:
#   "speccount"     - PSM count per peptide (what Comet's pepXML carries natively)
#   "ms1_precursor" - sum of MS1 precursor peak intensities across the peptide's
#                     PSMs (read from each MS2 scan header of the mzML)
INTENSITY_SOURCES = ("speccount", "ms1_precursor")

DATA_DIR = Path("/mnt/data3/PXD006118")
FASTA = DATA_DIR / "Mock_Comm_RefDB_V3.fasta"
COMP_P = DATA_DIR / "Composition_Of_EQUAL_PROTEIN_AMOUNT_Community.tab"
COMP_C = DATA_DIR / "Composition_Of_EQUAL_CELL_NUMBER_Community.tab"
COMP_U = DATA_DIR / "Composition_Of_UNEVEN_Community.tab"

OUT_ROOT = Path("/home/scjlau/ibaq_results")
WORK_DIR = OUT_ROOT / "_work"
PEPXML_DIR = WORK_DIR / "pepxml"
MZML_LINK_DIR = WORK_DIR / "mzml"

# Output layout:
#   OUT_ROOT/{C,P,U}/{ibaq,spectral_counting}/{csv,charts}/
# "ibaq" folder holds results driven by real MS1 peptide intensities;
# "spectral_counting" folder holds results driven by PSM counts.
COMMUNITY_PREFIXES = {"C": "equal_cell", "P": "equal_protein", "U": "uneven"}


def source_folder(source: str) -> str:
    return "ibaq" if source == "ms1_precursor" else "spectral_counting"


def csv_dir(community: str, source: str) -> Path:
    return OUT_ROOT / community / source_folder(source) / "csv"


def chart_dir(community: str, source: str) -> Path:
    return OUT_ROOT / community / source_folder(source) / "charts"


# Top-level paths for artifacts that span communities (paper-comparison tables).
TOP_CSV_DIR = OUT_ROOT / "_shared" / "csv"
TOP_CHART_DIR = OUT_ROOT / "_shared" / "charts"

COMET_PATH = "/programs/tpp/release_7-3-0/build/gnu-x86_64-release/bin/comet"

XCORR_CUTOFF = 2.0          # static fallback if FDR calibration is disabled
TARGET_FDR = 0.01           # per-sample PSM-level FDR target (1%)
XCORR_FLOOR = 1.0           # never drop below this even if FDR allows it
MIN_ABUNDANCE = 1e-4
TOP_N_SPECIES = 10          # per-sample CSV + charts show this many species

# Canonical species name -> short label used in the Kleiner 2017 composition
# tables (and in our figures).  Where the paper collapses multiple strains
# into one species in the proteomic output, we join labels with "+".
SPECIES_TO_LABEL: dict[str, str] = {
    # Bacteria with single labels
    "Pseudomonas denitrificans":      "PD",
    "Bacillus subtilis":              "BS",
    "Paracoccus denitrificans":       "PaD",
    "Pseudomonas pseudoalcaligenes":  "KF7",
    "Chromobacterium violaceum":      "CV",
    "Stenotrophomonas maltophilia":   "SMS",
    "Pseudomonas fluorescens":        "Pfl",
    "Burkholderia xenovorans":        "BXL",
    "Alteromonas macleodii":          "Am2",
    "Escherichia coli":               "K12",
    "Thermus thermophilus":           "HB2",
    "Chlamydomonas reinhardtii":      "CRH",
    "Desulfovibrio vulgaris":         "DVH",
    "Nitrosomonas europaea":          "Ne1",
    "Nitrosomonas ureae":             "Nu1",
    "Nitrososphaera viennensis":      "Nv",
    "Nitrosospira multiformis":       "Nm1",
    "Roseobacter sp.":                "AK199",
    # Species with FASTA-vs-composition name disagreements (aliases)
    "Agrobacterium tumefaciens":      "ATN",
    "Agrobacterium fabrum":           "ATN",      # FASTA form
    "Cupriavidus metalliredcens":     "Cup",
    "Cupriavidus metallidurans":      "Cup",      # FASTA form
    "Salmonella enterica":            "LT2+",     # LT2 + H88 + 2197 + 91
    "Salmonella typhimurium":         "LT2+",     # FASTA form
    # Strain-collapsed species (multiple .tab labels → one species)
    "Staphylococcus aureus":          "137+259",
    "Rhizobium leguminosarum":        "841+VF",
    # Phages (canonical names from our composition override)
    "Enterobacteria phage M13":       "M13",
    "Enterobacteria phage MS2":       "F2",
    "Enterobacteria phage P22":       "P22",
    "Enterobacteria phage ES18":      "ES18",
    "Salmonella phage Felix O1":      "F0",
}


def species_to_label(species: str) -> str:
    """Return the short composition-table label for a canonical species name,
    falling back to the species name itself when unmapped."""
    return SPECIES_TO_LABEL.get(species, species)

LOG_FMT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
log = logging.getLogger("ibaq_run")


# ---------------------------------------------------------------------------
# Comet
# ---------------------------------------------------------------------------

COMET_PARAMS_TEMPLATE_PATH = REPO_ROOT / "comet.params.new"


def _offline_env() -> dict:
    return {
        **os.environ,
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }


def run_comet(mzml: Path, threads: int) -> Path:
    """Run Comet on ``mzml`` and return the resulting pepXML path.

    mzML is symlinked into ``MZML_LINK_DIR`` so Comet writes its pep.xml
    next to the symlink (inside WORK_DIR) rather than into the read-only
    data directory.
    """
    PEPXML_DIR.mkdir(parents=True, exist_ok=True)
    MZML_LINK_DIR.mkdir(parents=True, exist_ok=True)

    params_path = PEPXML_DIR / "comet.params"
    if not params_path.exists():
        text = COMET_PARAMS_TEMPLATE_PATH.read_text(encoding="utf-8")
        text = re.sub(r"^database_name\s*=.*$", f"database_name = {FASTA}", text, flags=re.M)
        text = re.sub(r"^num_threads\s*=.*$", f"num_threads = {threads}", text, flags=re.M)
        text = re.sub(r"^decoy_search\s*=.*$", "decoy_search = 1", text, flags=re.M)
        text = re.sub(r"^output_pepxmlfile\s*=.*$", "output_pepxmlfile = 1", text, flags=re.M)
        text = re.sub(r"^output_txtfile\s*=.*$", "output_txtfile = 0", text, flags=re.M)
        params_path.write_text(text, encoding="utf-8")

    linked = MZML_LINK_DIR / mzml.name
    if linked.exists() or linked.is_symlink():
        linked.unlink()
    linked.symlink_to(mzml)

    pepxml_out = PEPXML_DIR / (mzml.stem + ".pep.xml")
    if pepxml_out.exists() and pepxml_out.stat().st_size > 0:
        log.info("[%s] pepXML exists, skipping Comet", mzml.name)
        return pepxml_out

    cmd = [COMET_PATH, f"-P{params_path}", str(linked)]
    log.info("[%s] Running Comet (threads=%d)", mzml.name, threads)
    t0 = time.time()
    res = subprocess.run(
        cmd, capture_output=True, text=True, env=_offline_env(), timeout=7200,
    )
    if res.returncode != 0:
        log.error(
            "[%s] Comet failed rc=%d:\n%s", mzml.name, res.returncode, res.stderr[-2000:],
        )
        raise RuntimeError(f"Comet failed for {mzml.name}")

    produced = linked.with_suffix(".pep.xml")
    if not produced.exists():
        raise RuntimeError(f"Comet produced no pepXML for {mzml.name}")
    shutil.move(str(produced), pepxml_out)
    log.info("[%s] Comet done in %.1f s", mzml.name, time.time() - t0)
    return pepxml_out


# ---------------------------------------------------------------------------
# pepXML -> peptide list (xcorr-filtered)
# ---------------------------------------------------------------------------

def extract_peptides(pepxml: Path, xcorr_cutoff: float) -> list[str]:
    """Rank-1 peptides with xcorr >= cutoff, decoys excluded."""
    tree = ET.parse(pepxml)
    root = tree.getroot()
    ns = root.tag.split("}")[0].strip("{") if root.tag.startswith("{") else ""

    def q(name: str) -> str:
        return f"{{{ns}}}{name}" if ns else name

    seen: set[str] = set()
    for hit in root.iter(q("search_hit")):
        if int(hit.attrib.get("hit_rank", "1")) != 1:
            continue
        pep = hit.attrib.get("peptide", "")
        if not pep:
            continue
        prots = {hit.attrib.get("protein", "")}
        for alt in hit.iter(q("alternative_protein")):
            prots.add(alt.attrib.get("protein", ""))
        if all(p.upper().startswith("DECOY") for p in prots if p):
            continue
        xcorr = 0.0
        for score in hit.iter(q("search_score")):
            if score.attrib.get("name") == "xcorr":
                xcorr = float(score.attrib.get("value", "0"))
                break
        if xcorr >= xcorr_cutoff:
            seen.add(pep)
    return sorted(seen)


# ---------------------------------------------------------------------------
# Composition-driven label/species mapping (authoritative for this dataset)
# ---------------------------------------------------------------------------

# Canonical species names for labels where composition "Name and strain"
# normalizes to something awkward or conflicts with FASTA conventions.
# Keys are composition labels (column 0 of the .tab files).
_LABEL_CANONICAL_OVERRIDES: dict[str, str] = {
    # Phages — paper's composition lists these as "Phage M13" etc. which
    # 2-token normalization renders as "Phage m13"; use canonical names.
    "M13": "Enterobacteria phage M13",
    "F2": "Enterobacteria phage MS2",
    "P22": "Enterobacteria phage P22",
    "F0": "Salmonella phage Felix O1",
    "ES18": "Enterobacteria phage ES18",
    # The composition .tab file contains a typo — "Cupriavidus metalliredcens"
    # instead of the correct species "Cupriavidus metallidurans" (ATCC 43123
    # / CH34).  The FASTA uses the correct spelling; force our pipeline to
    # agree with the FASTA so both sides of every comparison line up.
    "Cup": "Cupriavidus metallidurans",
}


def _compose_name_to_species(name: str, label: str | None = None) -> str | None:
    """Extract canonical species name from a composition "Name and strain" string."""
    if label and label in _LABEL_CANONICAL_OVERRIDES:
        return _LABEL_CANONICAL_OVERRIDES[label]
    cleaned = re.sub(r'["\']', "", name)
    cleaned = re.sub(r"\([^)]*\)", "", cleaned).strip()
    cleaned = cleaned.strip(",").strip()
    return _normalize_species(cleaned)


def load_composition_label_species() -> dict[str, str]:
    """Return composition_label -> canonical species name.

    Derived from the strain name in column 1 of the composition .tab files,
    with overrides for phages (where 2-token normalization produces ugly
    names like "Phage m13").  This map is authoritative for both the truth
    and the FASTA protein->species assignment used by iBAQ.
    """
    mapping: dict[str, str] = {}
    for path in (COMP_P, COMP_C):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="ignore") as fh:
            reader = csv.reader(fh, delimiter="\t")
            next(reader, None)  # header
            for row in reader:
                if len(row) < 2 or not row[0].strip():
                    continue
                label = row[0].strip()
                species = _compose_name_to_species(row[1], label=label)
                if species:
                    mapping[label] = species
    return mapping


def build_enriched_protein_species_map(fasta: Path) -> dict[str, str]:
    """protein_acc -> canonical species.

    Strategy:
    1. If accession starts with a known composition label prefix
       ({label}_...), use the composition-derived canonical species. This
       is the primary path and covers all labeled proteins correctly,
       bypassing header-level bracket tags that confuse OS=/[..] parsing
       (e.g. [Acyl-carrier-protein], [NADPH]).
    2. Otherwise fall back to OS= then bracket extraction.
    3. Proteins with no species info at all are dropped (won't appear in
       iBAQ output).
    """
    label_species = load_composition_label_species()
    log.info("Composition label->species map: %d labels", len(label_species))

    # Sort by descending label length so e.g. "2197" wins over "2".
    labels_sorted = sorted(label_species.keys(), key=lambda k: (-len(k), k))

    def _label_match(accession: str) -> str | None:
        for label in labels_sorted:
            if accession.startswith(label + "_"):
                return label_species[label]
        return None

    def _extract_from_header(header: str) -> str | None:
        m = re.search(r"\bOS=(.+?)(?:\s[A-Z]{2}=|$)", header)
        if m:
            return _normalize_species(m.group(1))
        # Prefer the LAST bracket, which conventionally carries the organism
        # (upstream brackets like [Acyl-carrier-protein] are functional tags).
        brackets = re.findall(r"\[([^\]]+)\]", header)
        for b in reversed(brackets):
            norm = _normalize_species(b)
            if not norm or " " not in norm:
                continue
            # require Genus species with alphabetic tokens (rejects tags like
            # [NAD+, L-lysine-forming], [NADPH], [EC:1.-.-.-]).
            genus, epithet = norm.split(" ", 1)
            if len(genus) >= 3 and genus.isalpha() and epithet.replace(".", "").replace("-", "").isalpha():
                return norm
        return None

    mapping: dict[str, str] = {}
    n_label = n_header = n_drop = 0

    with fasta.open("r", encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if not line.startswith(">"):
                continue
            header = line[1:].strip()
            acc = extract_accession(header)
            if not acc:
                continue
            if acc.upper().startswith(("DECOY", "CRAP")) or "contag" in acc.lower():
                continue
            species = _label_match(acc)
            if species:
                mapping[acc] = species
                n_label += 1
                continue
            species = _extract_from_header(header)
            if species:
                mapping[acc] = species
                n_header += 1
            else:
                n_drop += 1

    log.info(
        "Enriched protein->species map: %d proteins (label_match=%d, header_match=%d, unmapped=%d)",
        len(mapping), n_label, n_header, n_drop,
    )
    return mapping


# cached at module level so aggregation doesn't re-read the FASTA repeatedly
_PROTEIN_SPECIES_MAP_CACHE: dict[str, str] | None = None
_THEORETICAL_COUNTS_CACHE: dict[str, int] | None = None


def get_protein_species_map() -> dict[str, str]:
    global _PROTEIN_SPECIES_MAP_CACHE
    if _PROTEIN_SPECIES_MAP_CACHE is None:
        _PROTEIN_SPECIES_MAP_CACHE = build_enriched_protein_species_map(FASTA)
    return _PROTEIN_SPECIES_MAP_CACHE


def get_theoretical_counts() -> dict[str, int]:
    """accession -> number of theoretically observable tryptic peptides
    (7–30 aa, up to 2 missed cleavages; matches Schwanhäusser 2011 iBAQ)."""
    global _THEORETICAL_COUNTS_CACHE
    if _THEORETICAL_COUNTS_CACHE is None:
        log.info("Building theoretical tryptic peptide counts from FASTA ...")
        _THEORETICAL_COUNTS_CACHE = build_theoretical_peptide_counts(
            str(FASTA),
            min_len=7, max_len=30, missed_cleavages=2,
            exclude_prefixes=["DECOY", "contag"],
        )
    return _THEORETICAL_COUNTS_CACHE


def _run_method(method: str, psm_df: pd.DataFrame) -> dict[str, float]:
    if method == "raw_ibaq":
        return raw_ibaq(psm_df)
    if method == "ibaq_em":
        return ibaq_em(psm_df, max_iter=500, tol=1e-8)
    if method == "ibaq_theoretical":
        return ibaq_theoretical(psm_df, get_theoretical_counts())
    raise ValueError(f"Unknown iBAQ method: {method}")


# ---------------------------------------------------------------------------
# Target-decoy FDR calibration for the xcorr threshold
# ---------------------------------------------------------------------------

def compute_xcorr_cutoff_for_fdr(
    pepxml: Path, target_fdr: float, floor: float = XCORR_FLOOR,
) -> tuple[float, int, int]:
    """Sweep rank-1 PSMs, return the lowest xcorr cutoff with FDR <= target.

    FDR is estimated as decoy_hits / target_hits (concatenated TDA).
    Returns (cutoff, n_targets_retained, n_decoys_retained).
    If no xcorr achieves the target, falls back to `floor`.
    """
    tree = ET.parse(pepxml)
    root = tree.getroot()
    ns = root.tag.split("}")[0].strip("{") if root.tag.startswith("{") else ""

    def q(name: str) -> str:
        return f"{{{ns}}}{name}" if ns else name

    psms: list[tuple[float, bool]] = []
    for hit in root.iter(q("search_hit")):
        if int(hit.attrib.get("hit_rank", "1")) != 1:
            continue
        prots = {hit.attrib.get("protein", "")}
        for alt in hit.iter(q("alternative_protein")):
            prots.add(alt.attrib.get("protein", ""))
        is_decoy = bool(prots) and all(p.upper().startswith("DECOY") for p in prots if p)
        xcorr = 0.0
        for s in hit.iter(q("search_score")):
            if s.attrib.get("name") == "xcorr":
                xcorr = float(s.attrib.get("value", "0"))
                break
        psms.append((xcorr, is_decoy))

    if not psms:
        return floor, 0, 0

    psms.sort(key=lambda t: -t[0])
    cum_t = cum_d = 0
    best_cutoff = None
    best_t = best_d = 0
    for xcorr, is_decoy in psms:
        if is_decoy:
            cum_d += 1
        else:
            cum_t += 1
        if cum_t == 0:
            continue
        fdr = cum_d / cum_t
        if fdr <= target_fdr:
            best_cutoff = xcorr
            best_t, best_d = cum_t, cum_d

    if best_cutoff is None:
        return floor, 0, 0
    return max(best_cutoff, floor), best_t, best_d


# ---------------------------------------------------------------------------
# MS1 precursor intensity extraction
# ---------------------------------------------------------------------------

def _ms1_cache_path(mzml: Path) -> Path:
    return WORK_DIR / "ms1_precursor" / f"{mzml.stem}.scan_intensity.tsv"


def _build_scan_intensity_map(mzml: Path) -> dict[int, float]:
    """Scan number -> MS1 precursor peak intensity (as reported in the MS2
    scan header of the mzML).  One intensity per MS2 scan; MS1 scans are
    skipped.  Cached to disk so we only pay the mzML scan cost once.
    """
    cache = _ms1_cache_path(mzml)
    if cache.exists():
        m: dict[int, float] = {}
        with cache.open() as fh:
            next(fh, None)
            for line in fh:
                k, v = line.rstrip("\n").split("\t", 1)
                m[int(k)] = float(v)
        return m

    from pyteomics import mzml as mzml_mod
    log.info("[%s] reading MS1 precursor intensities from mzML ...", mzml.name)
    scan_intensity: dict[int, float] = {}
    t0 = time.time()
    for sp in mzml_mod.read(str(mzml)):
        if sp.get("ms level") != 2:
            continue
        scan_id = sp["id"]
        try:
            scan_num = int(scan_id.rsplit("scan=", 1)[-1])
        except ValueError:
            continue
        intensity = 0.0
        try:
            pr = sp["precursorList"]["precursor"][0]
            si = pr["selectedIonList"]["selectedIon"][0]
            intensity = float(si.get("peak intensity", 0.0) or 0.0)
        except (KeyError, IndexError, TypeError):
            pass
        scan_intensity[scan_num] = intensity

    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("w") as fh:
        fh.write("scan\tprecursor_intensity\n")
        for k, v in scan_intensity.items():
            fh.write(f"{k}\t{v}\n")
    log.info(
        "[%s] %d MS2 scans mapped to MS1 precursor intensity in %.1fs",
        mzml.name, len(scan_intensity), time.time() - t0,
    )
    return scan_intensity


def extract_peptide_ms1_intensities(
    mzml: Path, pepxml: Path, xcorr_cutoff: float,
) -> dict[str, float]:
    """peptide -> sum of MS1 precursor intensities across its rank-1 PSMs
    passing the xcorr cutoff, decoys excluded.  PSMs whose scan has no
    reported precursor intensity contribute 1.0 (a spectral-count fallback
    so the peptide doesn't vanish)."""
    scan_intensity = _build_scan_intensity_map(mzml)

    tree = ET.parse(pepxml)
    root = tree.getroot()
    ns = root.tag.split("}")[0].strip("{") if root.tag.startswith("{") else ""

    def q(name: str) -> str:
        return f"{{{ns}}}{name}" if ns else name

    out: dict[str, float] = {}
    n_hit = n_missing_scan = 0
    for query in root.iter(q("spectrum_query")):
        try:
            scan = int(query.attrib.get("start_scan", "0"))
        except ValueError:
            continue
        for hit in query.iter(q("search_hit")):
            if int(hit.attrib.get("hit_rank", "1")) != 1:
                continue
            pep = hit.attrib.get("peptide", "")
            if not pep:
                continue
            prots = {hit.attrib.get("protein", "")}
            for alt in hit.iter(q("alternative_protein")):
                prots.add(alt.attrib.get("protein", ""))
            if prots and all(p.upper().startswith("DECOY") for p in prots if p):
                continue
            xcorr = 0.0
            for s in hit.iter(q("search_score")):
                if s.attrib.get("name") == "xcorr":
                    xcorr = float(s.attrib.get("value", "0"))
                    break
            if xcorr < xcorr_cutoff:
                continue
            intensity = scan_intensity.get(scan, 0.0)
            if intensity <= 0:
                intensity = 1.0
                n_missing_scan += 1
            out[pep] = out.get(pep, 0.0) + intensity
            n_hit += 1
    log.info(
        "[%s] MS1 intensities aggregated: %d PSMs -> %d peptides (%d PSMs used fallback=1.0)",
        mzml.name, n_hit, len(out), n_missing_scan,
    )
    return out


# ---------------------------------------------------------------------------
# iBAQ run for a single sample
# ---------------------------------------------------------------------------

def _csv_path(sample: str, method: str, source: str) -> Path:
    community = sample[0].upper()
    out = csv_dir(community, source)
    out.mkdir(parents=True, exist_ok=True)
    return out / f"{sample}__{method}.csv"


def _write_method_csv(out_csv: Path, psm_df: pd.DataFrame, abundances: dict[str, float]) -> int:
    """Write every species above MIN_ABUNDANCE, sorted descending.  Adds a
    `label` column mapping each canonical species name to the Kleiner
    composition-table code so downstream charts can render shorter names.
    CSV output keeps the full distribution; graphs cap at TOP_N_SPECIES."""
    results: list[tuple[str, float, int]] = []
    for species, abundance in abundances.items():
        if abundance <= MIN_ABUNDANCE:
            continue
        n_pep = psm_df[psm_df["species"] == species]["peptide"].nunique()
        results.append((species, abundance, n_pep))
    results.sort(key=lambda r: r[1], reverse=True)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "species", "abundance", "peptide_count"])
        for species, abundance, n_pep in results:
            w.writerow([species_to_label(species), species,
                        f"{abundance:.6f}", n_pep])
    return len(results)


def run_ibaq_sample(mzml: Path, threads: int, target_fdr: float) -> list[Path]:
    sample = mzml.stem
    out_paths = [
        _csv_path(sample, m, s)
        for m in IBAQ_METHOD_NAMES
        for s in INTENSITY_SOURCES
    ]
    if all(p.exists() for p in out_paths):
        log.info("[%s] CSVs exist for all (method, intensity source) combos, skipping", sample)
        return out_paths

    pepxml = run_comet(mzml, threads=threads)
    xcorr_cutoff, n_tgt, n_dec = compute_xcorr_cutoff_for_fdr(pepxml, target_fdr)
    achieved_fdr = n_dec / max(n_tgt, 1)
    log.info(
        "[%s] FDR calibration: xcorr>=%.3f retains %d target / %d decoy PSMs "
        "(achieved FDR %.3f%%, target %.1f%%)",
        sample, xcorr_cutoff, n_tgt, n_dec, achieved_fdr * 100, target_fdr * 100,
    )

    peptides = extract_peptides(pepxml, xcorr_cutoff)
    log.info("[%s] %d unique peptides pass FDR filter", sample, len(peptides))
    if not peptides:
        log.warning("[%s] no peptides — empty CSVs", sample)
        for p in out_paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("species,abundance,peptide_count\n")
        return out_paths

    spectral_counts, peptide_protein_map = parse_pepxml(
        str(pepxml), exclude_prefixes=["DECOY", "contag"],
    )
    filtered_peptides = [p for p in peptides if p in peptide_protein_map]
    protein_species_map = get_protein_species_map()

    psm_tables: dict[str, pd.DataFrame] = {}
    if "speccount" in INTENSITY_SOURCES:
        psm_tables["speccount"] = build_psm_table(
            peptides=filtered_peptides,
            spectral_counts=spectral_counts,
            protein_species_map=protein_species_map,
            peptide_protein_map=peptide_protein_map,
        )
    if "ms1_precursor" in INTENSITY_SOURCES:
        ms1_intensities = extract_peptide_ms1_intensities(mzml, pepxml, xcorr_cutoff)
        psm_tables["ms1_precursor"] = build_psm_table(
            peptides=filtered_peptides,
            spectral_counts=ms1_intensities,
            protein_species_map=protein_species_map,
            peptide_protein_map=peptide_protein_map,
        )

    for source, psm_df in psm_tables.items():
        for method in IBAQ_METHOD_NAMES:
            out_csv = _csv_path(sample, method, source)
            if out_csv.exists():
                continue
            log.info("[%s] running method=%s intensity=%s", sample, method, source)
            abundances = _run_method(method, psm_df)
            n = _write_method_csv(out_csv, psm_df, abundances)
            log.info("[%s] %s/%s: %d taxa above %.0e",
                     sample, method, source, n, MIN_ABUNDANCE)
    return out_paths


# ---------------------------------------------------------------------------
# Parallel driver
# ---------------------------------------------------------------------------

def _worker(args: tuple[Path, int, float]) -> tuple[str, str]:
    mzml, threads, target_fdr = args
    try:
        run_ibaq_sample(mzml, threads=threads, target_fdr=target_fdr)
        return (mzml.name, "ok")
    except Exception as exc:  # noqa: BLE001
        log.exception("[%s] failed", mzml.name)
        return (mzml.name, f"error: {exc}")


def process_all(
    mzmls: list[Path], jobs: int, total_threads: int, target_fdr: float,
) -> list[tuple[str, str]]:
    per_job_threads = max(1, total_threads // max(jobs, 1))
    log.info(
        "Processing %d mzML files; jobs=%d, threads/job=%d, target_fdr=%.2f%%",
        len(mzmls), jobs, per_job_threads, target_fdr * 100,
    )
    args_list = [(m, per_job_threads, target_fdr) for m in mzmls]

    if jobs <= 1:
        return [_worker(a) for a in args_list]

    with mp.get_context("fork").Pool(processes=jobs) as pool:
        results = pool.map(_worker, args_list)
    return results


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

def parse_composition(path: Path, percent_col_idx: int) -> list[tuple[str, str, float]]:
    """Return list of (label, name_and_strain, percentage) from a composition .tab file."""
    result: list[tuple[str, str, float]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh, delimiter="\t")
        next(reader, None)  # header
        for row in reader:
            if len(row) < 2 or not row[0].strip():
                continue
            label = row[0].strip()
            name = row[1].strip() if len(row) > 1 else ""
            try:
                pct = float(row[percent_col_idx])
            except (ValueError, IndexError):
                continue
            result.append((label, name, pct))
    return result


def truth_by_species(comp_rows: list[tuple[str, str, float]]) -> dict[str, float]:
    """Collapse composition rows to normalized species proportions summing to 1.0.

    Species is derived from the 'Name and strain' column (authoritative),
    so every row contributes to the truth — including strains whose FASTA
    headers lack an OS= tag, and strains not present in the FASTA at all.
    """
    agg: dict[str, float] = defaultdict(float)
    for label, name, pct in comp_rows:
        species = _compose_name_to_species(name, label=label)
        if not species:
            log.warning("Truth: could not parse species from name %r (label %s)", name, label)
            continue
        agg[species] += pct
    total = sum(agg.values())
    if total <= 0:
        return {}
    return {sp: pct / total for sp, pct in agg.items()}


# ---------------------------------------------------------------------------
# Aggregation + metrics + charts
# ---------------------------------------------------------------------------

def compute_metrics(est: dict[str, float], truth: dict[str, float]) -> dict[str, float]:
    species = sorted(set(est) | set(truth))
    if not species:
        return {}
    y_true = np.array([truth.get(s, 0.0) for s in species])
    y_pred = np.array([est.get(s, 0.0) for s in species])

    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    l1 = float(np.sum(np.abs(y_true - y_pred)))

    if y_true.std() > 0 and y_pred.std() > 0:
        pearson = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson = float("nan")

    common = [s for s in species if s in est and s in truth]
    return {
        "n_species_union": len(species),
        "n_species_common": len(common),
        "n_species_est": len(est),
        "n_species_truth": len(truth),
        "R2": r2,
        "RMSE": rmse,
        "L1_error": l1,
        "Pearson_r": pearson,
    }


def _top_species_by_sum(est: dict[str, float], truth: dict[str, float], n: int) -> list[str]:
    """Pick the top-n species ranked by (est + truth) so that both large
    estimates and large truth values are represented."""
    return sorted(
        set(est) | set(truth),
        key=lambda s: -(est.get(s, 0) + truth.get(s, 0)),
    )[:n]


def plot_sample_bar(
    sample: str, est: dict[str, float], truth: dict[str, float], dataset: str,
    method: str, out: Path,
) -> None:
    species = _top_species_by_sum(est, truth, TOP_N_SPECIES)
    labels = [species_to_label(s) for s in species]
    y_est  = [est.get(s, 0) * 100 for s in species]
    y_true = [truth.get(s, 0) * 100 for s in species]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(species))
    w = 0.4
    ax.bar(x - w / 2, y_est, w, label="iBAQ estimate")
    ax.bar(x + w / 2, y_true, w, label=f"Ground truth ({dataset})", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Abundance (%)")
    ax.set_title(
        f"iBAQ vs ground truth — sample {sample}  (top {TOP_N_SPECIES})\n"
        f"PXD006118, {dataset}, method={method}",
        fontsize=10,
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def plot_sample_scatter(
    sample: str, est: dict[str, float], truth: dict[str, float],
    dataset: str, metrics: dict[str, float], method: str, out: Path,
) -> None:
    species = _top_species_by_sum(est, truth, TOP_N_SPECIES)
    y_true = np.array([truth.get(s, 0) for s in species])
    y_pred = np.array([est.get(s, 0) for s in species])

    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    ax.scatter(y_true * 100, y_pred * 100, alpha=0.8, s=55)
    for xi, yi, sp in zip(y_true, y_pred, species):
        ax.annotate(species_to_label(sp), (xi * 100, yi * 100),
                    xytext=(4, 3), textcoords="offset points", fontsize=8)
    mx = max(y_true.max(), y_pred.max()) * 100 * 1.1 if len(species) else 1
    ax.plot([0, mx], [0, mx], "k--", lw=1, label="y = x")
    ax.set_xlabel(f"Ground truth ({dataset}) %")
    ax.set_ylabel("iBAQ estimate %")
    r2 = metrics.get("R2", float("nan"))
    pr = metrics.get("Pearson_r", float("nan"))
    rmse = metrics.get("RMSE", float("nan"))
    ax.set_title(
        f"{sample}  (top {TOP_N_SPECIES}) — R²={r2:.3f}, Pearson r={pr:.3f}, RMSE={rmse:.4f}\n"
        f"PXD006118, {dataset}, {method}",
        fontsize=10,
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def plot_heatmap(wide: pd.DataFrame, method: str, out: Path) -> None:
    """Rows = species, cols = samples, log-proportion color. Top-N species only."""
    if wide.empty:
        return
    # keep the top-N species by mean abundance across samples
    top_idx = wide.mean(axis=1).sort_values(ascending=False).head(TOP_N_SPECIES).index
    trimmed = wide.loc[top_idx]
    data = np.log10(trimmed.fillna(0.0).to_numpy() + 1e-5)
    ylabels = [species_to_label(s) for s in trimmed.index]

    fig, ax = plt.subplots(figsize=(max(10, 0.3 * trimmed.shape[1]),
                                    max(4, 0.35 * trimmed.shape[0])))
    sns.heatmap(
        data, xticklabels=trimmed.columns, yticklabels=ylabels,
        cmap="viridis", cbar_kws={"label": "log10(abundance proportion)"}, ax=ax,
    )
    ax.set_title(
        f"iBAQ abundance — top {TOP_N_SPECIES}, PXD006118, method={method}",
        fontsize=11,
    )
    ax.set_xlabel("sample (mzML)")
    ax.set_ylabel("species")
    plt.xticks(rotation=75, ha="right", fontsize=7)
    plt.yticks(fontsize=9, rotation=0)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def plot_r2_bars(metrics_df: pd.DataFrame, out: Path) -> None:
    """Grouped bars: per-sample R² with side-by-side bars for each method."""
    if metrics_df.empty:
        return
    df = metrics_df.sort_values(["dataset", "sample", "method"]).copy()
    methods = sorted(df["method"].unique())
    samples = sorted(df["sample"].unique(), key=lambda s: (dataset_of(s), s))
    dataset_map = {s: dataset_of(s) for s in samples}

    import matplotlib.patches as mpatches
    dcolors = {"equal_protein": "#1f77b4", "equal_cell": "#ff7f0e", "uneven": "#2ca02c"}
    hatch_cycle = ["", "///", "xxx", "...", "\\\\", "++", "oo", "**"]
    method_hatches = {m: hatch_cycle[i % len(hatch_cycle)] for i, m in enumerate(methods)}

    fig, ax = plt.subplots(figsize=(max(14, 0.45 * len(samples)), 5.5))
    x = np.arange(len(samples))
    total_w = 0.8
    w = total_w / max(len(methods), 1)
    for mi, method in enumerate(methods):
        sub = df[df["method"] == method].set_index("sample").reindex(samples)
        y = sub["R2"].clip(lower=-1).to_numpy()
        offset = (mi - (len(methods) - 1) / 2) * w
        colors = [dcolors.get(dataset_map[s], "gray") for s in samples]
        ax.bar(x + offset, y, w, color=colors, edgecolor="black", linewidth=0.5,
               hatch=method_hatches[method], label=method)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("R² (estimate vs truth)")
    ax.set_title(
        "Per-sample R² — iBAQ estimate vs composition truth\n"
        f"PXD006118 (P=equal protein, C=equal cell), methods: {', '.join(methods)}",
        fontsize=11,
    )
    handles = [mpatches.Patch(color=dcolors["equal_protein"], label="P (equal protein)"),
               mpatches.Patch(color=dcolors["equal_cell"], label="C (equal cell)")]
    for m in methods:
        handles.append(mpatches.Patch(facecolor="white", edgecolor="black",
                                      hatch=method_hatches[m], label=f"method={m}"))
    ax.legend(handles=handles, fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def plot_mean_vs_truth(
    wide: pd.DataFrame, truth: dict[str, float], dataset: str,
    method: str, out: Path,
) -> None:
    if wide.empty or not truth:
        return
    means = wide.mean(axis=1)
    stds = wide.std(axis=1)
    # Rank by combined mean-estimate + truth, keep top N.
    species = sorted(
        set(means.index) | set(truth),
        key=lambda s: -(means.get(s, 0) + truth.get(s, 0)),
    )[:TOP_N_SPECIES]
    labels = [species_to_label(s) for s in species]
    y_mean  = np.array([means.get(s, 0) for s in species]) * 100
    y_std   = np.array([stds.get(s, 0)  for s in species]) * 100
    y_truth = np.array([truth.get(s, 0) for s in species]) * 100

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(species))
    w = 0.4
    ax.bar(x - w / 2, y_mean, w, yerr=y_std, capsize=3,
           label=f"iBAQ mean ± SD (n={wide.shape[1]})")
    ax.bar(x + w / 2, y_truth, w, alpha=0.7, label=f"Ground truth ({dataset})")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10)
    ax.set_ylabel("Abundance (%)")
    ax.set_title(
        f"Mean iBAQ estimate vs ground truth — {dataset} samples (top {TOP_N_SPECIES})\n"
        f"PXD006118, method={method}",
        fontsize=11,
    )
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


def plot_overall_scatter(pooled: pd.DataFrame, out: Path) -> None:
    """Pooled scatter across all samples, panels per method, colored by dataset."""
    if pooled.empty:
        return
    methods = sorted(pooled["method"].unique())
    fig, axes = plt.subplots(1, len(methods), figsize=(7 * len(methods), 6.5), squeeze=False)
    for ax, method in zip(axes[0], methods):
        sub = pooled[pooled["method"] == method]
        for dataset, dsub in sub.groupby("dataset"):
            ax.scatter(dsub["truth"] * 100, dsub["estimate"] * 100,
                       alpha=0.5, s=25, label=dataset)
        mx = max(sub["truth"].max(), sub["estimate"].max()) * 100 * 1.05 if len(sub) else 1
        ax.plot([0, mx], [0, mx], "k--", lw=1, label="y = x")
        ax.set_xlabel("Ground truth %")
        ax.set_ylabel("iBAQ estimate %")
        ax.set_title(f"method={method}\nPXD006118, pooled across samples", fontsize=10)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def find_mzmls(
    limit: int | None,
    mm_filter: str | None = None,
    rep_filter: list[str] | None = None,
    communities: str = "CPU",
) -> list[Path]:
    """Return sorted mzML paths matching the filters.

    mm_filter:   e.g. "100mM"   — keep only files containing this token
    rep_filter:  e.g. ["P1","C1"] — keep only these replicate prefixes
    communities: subset of "CPU"  (C=equal cell, P=equal protein, U=uneven)
    """
    communities = communities.upper()
    pattern = f"^[{re.escape(communities)}]"
    mzmls = [p for p in DATA_DIR.iterdir()
             if p.suffix.lower() == ".mzml" and re.match(pattern, p.name)]
    if mm_filter:
        token = f"_{mm_filter}."
        mzmls = [p for p in mzmls if token in p.name]
    if rep_filter:
        rep_set = {r.upper() for r in rep_filter}
        mzmls = [p for p in mzmls if p.name.split("_", 1)[0].upper() in rep_set]
    mzmls.sort()
    if limit:
        mzmls = mzmls[:limit]
    return mzmls


def dataset_of(sample: str) -> str:
    prefix = sample[0].upper()
    return COMMUNITY_PREFIXES.get(prefix, "unknown")


def _load_method_csvs(
    community: str, source: str, method: str,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    per_sample: dict[str, dict[str, float]] = {}
    per_sample_npep: dict[str, dict[str, int]] = {}
    cdir = csv_dir(community, source)
    if not cdir.exists():
        return per_sample, per_sample_npep
    suffix = f"__{method}.csv"
    for csv_path in sorted(cdir.glob(f"*{suffix}")):
        # Skip the wide cross-sample summary tables that share the suffix.
        if csv_path.name.startswith("summary_all_samples"):
            continue
        sample = csv_path.name[:-len(suffix)]
        df = pd.read_csv(csv_path)
        if df.empty or "species" not in df.columns:
            continue
        per_sample[sample] = dict(zip(df["species"], df["abundance"]))
        per_sample_npep[sample] = dict(zip(df["species"], df["peptide_count"]))
    return per_sample, per_sample_npep


def _load_truth() -> dict[str, dict[str, float]]:
    """Ground truth for iBAQ validation = Protein abundance % (col index 3).

    iBAQ measures protein mass, so the right reference for every community
    type is the protein input column — not the cell-number column.  The
    paper's Fig. 3 uses protein input for all three communities too.  Cell
    abundance % (column 5) is retained separately for context (e.g.
    comparisons against 16S/FISH methods that count cells, not mass).
    """
    return {
        "equal_protein": truth_by_species(parse_composition(COMP_P, percent_col_idx=3)),
        "equal_cell":    truth_by_species(parse_composition(COMP_C, percent_col_idx=3)),
        "uneven":        truth_by_species(parse_composition(COMP_U, percent_col_idx=3)),
    }


def _load_cell_truth() -> dict[str, dict[str, float]]:
    """Cell abundance % truth (not what iBAQ measures; kept for comparison)."""
    return {
        "equal_protein": truth_by_species(parse_composition(COMP_P, percent_col_idx=5)),
        "equal_cell":    truth_by_species(parse_composition(COMP_C, percent_col_idx=5)),
        "uneven":        truth_by_species(parse_composition(COMP_U, percent_col_idx=5)),
    }


def aggregate() -> None:
    TOP_CSV_DIR.mkdir(parents=True, exist_ok=True)
    TOP_CHART_DIR.mkdir(parents=True, exist_ok=True)

    truths = _load_truth()
    cell_truths = _load_cell_truth()
    all_species = sorted({sp for t in truths.values() for sp in t}
                         | {sp for t in cell_truths.values() for sp in t})
    truth_df = pd.DataFrame({"species": all_species})
    truth_df["label"] = truth_df["species"].map(species_to_label)
    for ds, t in truths.items():
        truth_df[f"truth_protein_{ds}"] = truth_df["species"].map(t).fillna(0.0)
    for ds, t in cell_truths.items():
        truth_df[f"truth_cell_{ds}"] = truth_df["species"].map(t).fillna(0.0)
    truth_df.to_csv(TOP_CSV_DIR / "ground_truth_by_species.csv", index=False)

    all_metrics: list[dict] = []
    all_pooled: list[dict] = []
    samples_found = False

    for community, dataset_name in COMMUNITY_PREFIXES.items():
        truth = truths[dataset_name]
        for source in INTENSITY_SOURCES:
            cdir = csv_dir(community, source)
            chdir = chart_dir(community, source)
            chdir.mkdir(parents=True, exist_ok=True)

            # If no CSVs present for this (community, source) combo, skip.
            if not any(cdir.glob("*.csv")):
                continue

            for method in IBAQ_METHOD_NAMES:
                per_sample, per_sample_npep = _load_method_csvs(community, source, method)
                if not per_sample:
                    continue
                samples_found = True
                combo = f"{method}/{source}"

                for sample, est in per_sample.items():
                    m = compute_metrics(est, truth)
                    m.update(
                        sample=sample, dataset=dataset_name, community=community,
                        method=method, intensity_source=source,
                        n_peptides_total=sum(per_sample_npep.get(sample, {}).values()),
                    )
                    all_metrics.append(m)
                    plot_sample_bar(sample, est, truth, dataset_name, combo,
                                    chdir / f"{sample}__{method}_bar.png")
                    plot_sample_scatter(sample, est, truth, dataset_name, m, combo,
                                        chdir / f"{sample}__{method}_scatter.png")
                    for sp in set(est) | set(truth):
                        all_pooled.append({
                            "sample": sample, "dataset": dataset_name,
                            "community": community, "method": method,
                            "intensity_source": source, "species": sp,
                            "estimate": est.get(sp, 0.0), "truth": truth.get(sp, 0.0),
                        })

                all_species = sorted({sp for est in per_sample.values() for sp in est})
                wide = pd.DataFrame(index=all_species,
                                    columns=sorted(per_sample.keys()), dtype=float)
                for sample, est in per_sample.items():
                    for sp, val in est.items():
                        wide.at[sp, sample] = val
                wide.fillna(0.0, inplace=True)
                wide.to_csv(cdir / f"summary_all_samples__{method}.csv")

                plot_heatmap(wide, combo, chdir / f"heatmap_all_samples__{method}.png")
                plot_mean_vs_truth(wide, truth, dataset_name, combo,
                                   chdir / f"mean_vs_truth__{method}.png")

            # Per-(community, source) metrics + R² chart.
            per_combo = [m for m in all_metrics
                         if m["community"] == community and m["intensity_source"] == source]
            if per_combo:
                df = pd.DataFrame(per_combo)
                df.to_csv(cdir / "metrics_summary.csv", index=False)
                plot_r2_bars(df.assign(method=df["method"]),
                             chdir / "R2_by_sample.png")

    if not samples_found:
        log.warning("No per-sample CSVs found; nothing to aggregate.")
        return

    metrics_df = pd.DataFrame(all_metrics)[[
        "sample", "community", "dataset", "method", "intensity_source",
        "n_peptides_total",
        "n_species_est", "n_species_truth", "n_species_common",
        "R2", "RMSE", "L1_error", "Pearson_r",
    ]]
    metrics_df.to_csv(TOP_CSV_DIR / "metrics_summary.csv", index=False)
    log.info("Wrote %s (%d rows)", TOP_CSV_DIR / "metrics_summary.csv", len(metrics_df))

    pooled_df = pd.DataFrame(all_pooled)
    pooled_df.to_csv(TOP_CSV_DIR / "pooled_estimate_vs_truth.csv", index=False)

    # Cross-community R² chart with method+source as key.
    metrics_df_grouped = metrics_df.copy()
    metrics_df_grouped["method"] = metrics_df_grouped["method"] + "/" + metrics_df_grouped["intensity_source"]
    pooled_df_grouped = pooled_df.copy()
    pooled_df_grouped["method"] = pooled_df_grouped["method"] + "/" + pooled_df_grouped["intensity_source"]
    plot_r2_bars(metrics_df_grouped, TOP_CHART_DIR / "R2_by_sample_all_communities.png")
    plot_overall_scatter(pooled_df_grouped, TOP_CHART_DIR / "pooled_scatter_all_communities.png")

    total_charts = sum(1 for _ in OUT_ROOT.rglob("*.png"))
    log.info("Wrote %d charts total under %s", total_charts, OUT_ROOT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N matching mzML (for testing)")
    parser.add_argument("--jobs", type=int, default=1,
                        help="Parallel Comet jobs (default 1)")
    parser.add_argument("--threads", type=int, default=os.cpu_count() or 8,
                        help="Total Comet threads to distribute across jobs")
    parser.add_argument("--skip-comet", action="store_true",
                        help="Skip Comet/iBAQ; only re-aggregate existing CSVs + charts")
    parser.add_argument("--mm", type=str, default=None,
                        help="Filter mzML by concentration token (e.g. '100mM')")
    parser.add_argument("--reps", type=str, default=None,
                        help="Comma-separated replicate prefixes (e.g. 'P1,C1')")
    parser.add_argument("--target-fdr", type=float, default=TARGET_FDR,
                        help=f"Per-sample PSM-level FDR target (default {TARGET_FDR})")
    parser.add_argument("--communities", type=str, default="CPU",
                        help="Community prefixes to include, e.g. 'CPU' (default), 'CP', 'U'")
    args = parser.parse_args()
    rep_list = [r.strip() for r in args.reps.split(",")] if args.reps else None

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    for c in args.communities.upper():
        if c not in COMMUNITY_PREFIXES:
            continue
        for src in INTENSITY_SOURCES:
            csv_dir(c, src).mkdir(parents=True, exist_ok=True)
            chart_dir(c, src).mkdir(parents=True, exist_ok=True)

    if not args.skip_comet:
        mzmls = find_mzmls(args.limit, mm_filter=args.mm, rep_filter=rep_list,
                           communities=args.communities)
        log.info(
            "Found %d mzML files (communities=%s, mm=%s, reps=%s) under %s",
            len(mzmls), args.communities, args.mm, rep_list, DATA_DIR,
        )
        results = process_all(
            mzmls, jobs=args.jobs, total_threads=args.threads,
            target_fdr=args.target_fdr,
        )
        n_ok = sum(1 for _, s in results if s == "ok")
        log.info("Per-sample iBAQ done: %d ok / %d total", n_ok, len(results))
        for name, status in results:
            if status != "ok":
                log.error("  %s -> %s", name, status)

    log.info("Aggregating ...")
    aggregate()
    log.info("DONE. Outputs at %s", OUT_ROOT)


if __name__ == "__main__":
    main()
