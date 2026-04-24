#!/usr/bin/env python3
"""Benchmark our Comet+iBAQ pipeline against the Kleiner 2017 paper's
MaxQuant-derived quantitation (Supplementary Data 1, Figure 3 source data).

For each P*/C* sample we have three things:
    - truth   : composition input (ug protein or cell count), from the .tab files
    - paper   : MaxQuant PSMs and peptide intensities from the paper's .csv
    - ours    : our Comet + iBAQ pipeline output (CSV per sample/method/intensity)

This script computes metrics for (paper vs truth), (ours vs truth), and
(ours vs paper), renders scatter panels, and writes a combined CSV.

Run with no args; reads paths from module constants."""

from __future__ import annotations

import csv
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))
from run_ibaq_mock_comm import (  # noqa: E402
    _compose_name_to_species, COMP_P, COMP_C, COMP_U,
    csv_dir, parse_composition, truth_by_species,
    IBAQ_METHOD_NAMES, INTENSITY_SOURCES, COMMUNITY_PREFIXES,
)

PAPER_CSV = Path("/home/scjlau/ibaq_results/41467_2017_1544_MOESM2_ESM.csv")
OUT_DIR = Path("/home/scjlau/ibaq_results")
CMP_CSV_DIR = OUT_DIR / "_shared" / "csv"
CMP_CHART_DIR = OUT_DIR / "_shared" / "charts"
CMP_CSV_DIR.mkdir(parents=True, exist_ok=True)
CMP_CHART_DIR.mkdir(parents=True, exist_ok=True)

LOG_FMT = "%(asctime)s %(levelname)s %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%H:%M:%S")
log = logging.getLogger("compare_to_paper")

# Samples the paper quantifies that we also process.
P_SAMPLES = ["P1", "P2", "P3", "P4"]
C_SAMPLES = ["C1", "C2", "C3", "C4"]
U_SAMPLES = ["U1", "U2", "U3", "U4"]
ALL_SAMPLES = C_SAMPLES + P_SAMPLES + U_SAMPLES

# Column layout of the paper CSV (see the "Label,Species,..." row):
#   0  Label
#   1  Species
#   2-4  Input Protein Amount      (U, C, P)
#   5-7  Input Cell Number         (U, C, P)
#   8-19  PSMs                      (C1..C4, P1..P4, U1..U4)
#   20-31 Intensities Unique        (C1..C4, P1..P4, U1..U4)
#   32-43 Intensities Razor+Unique  (C1..C4, P1..P4, U1..U4)
PAPER_SAMPLE_ORDER = C_SAMPLES + P_SAMPLES + ["U1", "U2", "U3", "U4"]
PSM_COLS = {s: 8 + i for i, s in enumerate(PAPER_SAMPLE_ORDER)}
INT_UNIQUE_COLS = {s: 20 + i for i, s in enumerate(PAPER_SAMPLE_ORDER)}
INT_RAZOR_COLS = {s: 32 + i for i, s in enumerate(PAPER_SAMPLE_ORDER)}

# Phage labels in the paper CSV don't follow "Genus species" form; map them
# to the canonical names we use (see run_ibaq_mock_comm._LABEL_CANONICAL_OVERRIDES).
PAPER_PHAGE_CANON = {
    "Phage M13":  "Enterobacteria phage M13",
    "Phage F2":   "Enterobacteria phage MS2",
    "Phage P22":  "Enterobacteria phage P22",
    "Phage F0":   "Salmonella phage Felix O1",
    "Phage ES18": "Enterobacteria phage ES18",
}


def canonicalize(paper_species: str) -> str | None:
    """Collapse the paper's strain-level names to our 'Genus species' canonical form."""
    if not paper_species:
        return None
    s = paper_species.strip()
    if s in PAPER_PHAGE_CANON:
        return PAPER_PHAGE_CANON[s]
    # Strip " (3 strains combined)" style qualifiers
    s = re.sub(r"\s*\([^)]*\)\s*", "", s).strip()
    # _compose_name_to_species normalizes to Genus species (lowercase epithet)
    norm = _compose_name_to_species(s)
    return norm


# ---------------------------------------------------------------------------
# Load paper CSV
# ---------------------------------------------------------------------------

def _try_float(x: str) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def load_paper_tables() -> dict[str, dict[str, dict[str, float]]]:
    """Return {"psm" | "int_unique" | "int_razor": {sample: {species: value}}}.

    Values are summed across paper rows that collapse to the same canonical
    species (e.g. the two S. aureus strains, the two R. leguminosarum strains).
    """
    psm: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    intu: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    intr: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))

    with PAPER_CSV.open() as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    # Find the first data row (the first block ends at a blank line); paper has
    # two identical blocks — we only take the first (raw values), then normalize
    # ourselves so the comparison is consistent.
    # Header is the row starting with "Label,Species".
    header_idx = None
    for i, row in enumerate(rows):
        if row and row[0].strip() == "Label" and len(row) > 1 and "Species" in row[1]:
            header_idx = i
            break
    if header_idx is None:
        raise RuntimeError("Paper CSV: header row not found")

    n_species = 0
    for row in rows[header_idx + 1:]:
        if not row or not row[0].strip() or not row[1].strip():
            break  # end of first raw-values block
        label = row[0].strip()
        paper_sp = row[1].strip()
        canon = canonicalize(paper_sp)
        if canon is None:
            log.warning("Paper species %r (label %s) could not be canonicalized", paper_sp, label)
            continue
        for s in ALL_SAMPLES:
            psm[s][canon]  += _try_float(row[PSM_COLS[s]])
            intu[s][canon] += _try_float(row[INT_UNIQUE_COLS[s]])
            intr[s][canon] += _try_float(row[INT_RAZOR_COLS[s]])
        n_species += 1

    log.info("Paper CSV parsed: %d species (after strain collapse)", n_species)
    return {"psm": psm, "int_unique": intu, "int_razor": intr}


def normalize_sample(d: dict[str, float]) -> dict[str, float]:
    tot = sum(d.values())
    if tot <= 0:
        return {}
    return {k: v / tot for k, v in d.items()}


# ---------------------------------------------------------------------------
# Load our pipeline output
# ---------------------------------------------------------------------------

def load_ours(method: str, source: str) -> dict[str, dict[str, float]]:
    """Scan per-community CSV folders and build {sample_short: {species: abundance}}."""
    out: dict[str, dict[str, float]] = {}
    suffix = f"__{method}.csv"
    for community in COMMUNITY_PREFIXES:
        cdir = csv_dir(community, source)
        if not cdir.exists():
            continue
        for p in sorted(cdir.glob(f"*{suffix}")):
            sample_run = p.name[:-len(suffix)]           # e.g. "P1_run3_100mM"
            short = sample_run.split("_", 1)[0]           # "P1"
            if short not in ALL_SAMPLES:
                continue
            df = pd.read_csv(p)
            if df.empty:
                continue
            out[short] = dict(zip(df["species"], df["abundance"]))
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metric_pair(est: dict[str, float], ref: dict[str, float]) -> dict[str, float]:
    """R², Pearson r, RMSE, L1 with est aligned to ref species union."""
    species = sorted(set(est) | set(ref))
    if not species:
        return dict(R2=float("nan"), Pearson_r=float("nan"), RMSE=float("nan"),
                    L1=float("nan"), n_species=0)
    y_est = np.array([est.get(s, 0.0) for s in species])
    y_ref = np.array([ref.get(s, 0.0) for s in species])
    ss_res = float(np.sum((y_ref - y_est) ** 2))
    ss_tot = float(np.sum((y_ref - y_ref.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y_ref - y_est) ** 2)))
    l1 = float(np.sum(np.abs(y_ref - y_est)))
    pearson = (float(np.corrcoef(y_ref, y_est)[0, 1])
               if y_ref.std() > 0 and y_est.std() > 0 else float("nan"))
    return dict(R2=r2, Pearson_r=pearson, RMSE=rmse, L1=l1, n_species=len(species))


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_three_way_scatter(
    truth: dict[str, float], paper: dict[str, float], ours: dict[str, float],
    sample: str, dataset: str, ours_label: str, out: Path,
) -> None:
    species = sorted(set(truth) | set(paper) | set(ours))
    y_t = np.array([truth.get(s, 0.0) for s in species]) * 100
    y_p = np.array([paper.get(s, 0.0) for s in species]) * 100
    y_o = np.array([ours.get(s, 0.0) for s in species]) * 100

    mx = max(y_t.max(), y_p.max(), y_o.max()) * 1.05 if len(species) else 1.0

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))
    for ax, (x, y, xlbl, ylbl, title) in zip(axes, [
        (y_t, y_p, "truth (%)", "paper/MaxQuant (%)", "Paper vs truth"),
        (y_t, y_o, "truth (%)", f"ours/{ours_label} (%)", "Ours vs truth"),
        (y_p, y_o, "paper (%)", f"ours/{ours_label} (%)", "Ours vs paper"),
    ]):
        ax.scatter(x, y, alpha=0.7, s=45)
        ax.plot([0, mx], [0, mx], "k--", lw=1)
        ax.set_xlabel(xlbl); ax.set_ylabel(ylbl)
        ax.set_title(title, fontsize=10)
        ax.set_xlim(-0.5, mx); ax.set_ylim(-0.5, mx)

    fig.suptitle(
        f"{sample} ({dataset}) — composition truth vs Kleiner 2017 MaxQuant vs our pipeline",
        fontsize=12,
    )
    plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close(fig)


def plot_r2_comparison(metrics: pd.DataFrame, out: Path) -> None:
    """Grouped bars: R² per sample, with one bar per (estimator, reference)."""
    piv = metrics.pivot_table(index="sample", columns="estimator_vs_ref",
                              values="R2", aggfunc="mean").sort_index()
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(max(14, 0.7 * piv.shape[0]), 5.5))
    piv.plot(kind="bar", ax=ax, edgecolor="black", width=0.85)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_ylabel("R²")
    ax.set_title(
        "Per-sample R² — composition truth as reference\n"
        "PXD006118, 100mM wash, 1% FDR",
        fontsize=11,
    )
    ax.legend(title="", fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out, dpi=150); plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # --- Load truth (Protein abundance % for every community; iBAQ measures
    # protein mass, so using cell % for C samples is methodologically wrong.
    # Paper's Fig. 3 uses protein input as reference for all communities.) ---
    truth_P = truth_by_species(parse_composition(COMP_P, percent_col_idx=3))
    truth_C = truth_by_species(parse_composition(COMP_C, percent_col_idx=3))
    truth_U = truth_by_species(parse_composition(COMP_U, percent_col_idx=3))
    log.info("Truth: %d species (equal_protein), %d (equal_cell), %d (uneven)",
             len(truth_P), len(truth_C), len(truth_U))
    truth_by_dataset = {"equal_protein": truth_P, "equal_cell": truth_C, "uneven": truth_U}

    def truth_for(sample: str) -> dict[str, float]:
        c = sample[0].upper()
        return truth_by_dataset[COMMUNITY_PREFIXES[c]]

    def dataset_for(sample: str) -> str:
        return COMMUNITY_PREFIXES[sample[0].upper()]

    # --- Load paper data (raw values; normalize below) ---
    paper = load_paper_tables()

    # --- Load our outputs: pick the best combo we've identified (ibaq_em + ms1_precursor) ---
    best_method, best_source = "ibaq_em", "ms1_precursor"
    ours_main = load_ours(best_method, best_source)
    log.info(
        "Our pipeline output (%s/%s): %d samples loaded",
        best_method, best_source, len(ours_main),
    )

    # --- Compute metrics in all three comparisons for each paper signal ---
    metrics_rows: list[dict] = []
    paper_signals = [
        ("paper_psm",         "psm",         "MaxQuant PSM %"),
        ("paper_int_unique",  "int_unique",  "MaxQuant intensity (unique)"),
        ("paper_int_razor",   "int_razor",   "MaxQuant intensity (razor+unique)"),
    ]
    for short in ALL_SAMPLES:
        ds = dataset_for(short)
        truth = truth_for(short)
        our_est = ours_main.get(short, {})
        for label, key, _pretty in paper_signals:
            paper_est = normalize_sample(paper[key].get(short, {}))
            # paper vs truth
            m = metric_pair(paper_est, truth)
            m.update(sample=short, dataset=ds,
                     estimator_vs_ref=f"{label} vs truth")
            metrics_rows.append(m)
        # ours vs truth (the "best" combination)
        m = metric_pair(our_est, truth)
        m.update(sample=short, dataset=ds,
                 estimator_vs_ref=f"ours/{best_method}/{best_source} vs truth")
        metrics_rows.append(m)
        # ours vs paper (unique-intensity, the paper's primary quant signal)
        paper_int_u = normalize_sample(paper["int_unique"].get(short, {}))
        m = metric_pair(our_est, paper_int_u)
        m.update(sample=short, dataset=ds,
                 estimator_vs_ref=f"ours vs paper_int_unique")
        metrics_rows.append(m)

    metrics_df = pd.DataFrame(metrics_rows)
    out_csv = CMP_CSV_DIR / "paper_vs_ours_metrics.csv"
    metrics_df.to_csv(out_csv, index=False)
    log.info("Wrote %s (%d rows)", out_csv, len(metrics_df))

    # --- Per-sample species-level comparison table ---
    long_rows: list[dict] = []
    for short in ALL_SAMPLES:
        ds = dataset_for(short)
        truth = truth_for(short)
        our_est = ours_main.get(short, {})
        paper_psm = normalize_sample(paper["psm"].get(short, {}))
        paper_int_u = normalize_sample(paper["int_unique"].get(short, {}))
        paper_int_r = normalize_sample(paper["int_razor"].get(short, {}))
        species = sorted(set(truth) | set(paper_int_u) | set(paper_psm) | set(our_est))
        for sp in species:
            long_rows.append({
                "sample": short, "dataset": ds, "species": sp,
                "truth": truth.get(sp, 0.0),
                "paper_psm": paper_psm.get(sp, 0.0),
                "paper_int_unique": paper_int_u.get(sp, 0.0),
                "paper_int_razor": paper_int_r.get(sp, 0.0),
                "ours_ibaq_em_ms1": our_est.get(sp, 0.0),
            })
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(CMP_CSV_DIR / "paper_vs_ours_long.csv", index=False)

    # --- Per-sample 3-panel scatter (truth vs paper vs ours) ---
    for short in ALL_SAMPLES:
        ds = dataset_for(short)
        truth = truth_for(short)
        our_est = ours_main.get(short, {})
        paper_int_u = normalize_sample(paper["int_unique"].get(short, {}))
        plot_three_way_scatter(
            truth, paper_int_u, our_est, short, ds,
            ours_label=f"{best_method}/{best_source}",
            out=CMP_CHART_DIR / f"compare_three_way_{short}.png",
        )

    plot_r2_comparison(metrics_df, CMP_CHART_DIR / "R2_paper_vs_ours.png")

    # -----------------------------------------------------------------------
    # Figure 3c-style analysis: per-species "measured % / input %" fold
    # deviation.  Drop zero values (species not detected) as the paper does,
    # and aggregate across the 4 replicates per community type.
    # -----------------------------------------------------------------------
    deviation_rows: list[dict] = []
    estimators: dict[str, dict[str, dict[str, float]]] = {
        "paper_psm": {s: normalize_sample(paper["psm"].get(s, {})) for s in ALL_SAMPLES},
        "paper_int_unique": {s: normalize_sample(paper["int_unique"].get(s, {})) for s in ALL_SAMPLES},
        "paper_int_razor": {s: normalize_sample(paper["int_razor"].get(s, {})) for s in ALL_SAMPLES},
    }
    # Add every (method, intensity_source) we ran for ours.
    for method in IBAQ_METHOD_NAMES:
        for source in INTENSITY_SOURCES:
            our_tab = load_ours(method, source)
            if our_tab:
                estimators[f"ours/{method}/{source}"] = our_tab

    for sample in ALL_SAMPLES:
        dataset = dataset_for(sample)
        truth = truth_for(sample)
        for est_name, per_sample in estimators.items():
            est = per_sample.get(sample, {})
            if not est:
                continue
            for sp, truth_val in truth.items():
                if truth_val <= 0:
                    continue
                measured = est.get(sp, 0.0)
                if measured <= 0:
                    continue  # paper convention: drop zeros
                deviation_rows.append({
                    "sample": sample, "dataset": dataset, "estimator": est_name,
                    "species": sp,
                    "measured_over_input": measured / truth_val,
                })

    dev_df = pd.DataFrame(deviation_rows)
    dev_df.to_csv(CMP_CSV_DIR / "figure3c_deviations.csv", index=False)

    # Summarize: median, IQR, and Fig-3c-style "fraction of species within 2x".
    summary = (
        dev_df.groupby(["dataset", "estimator"])["measured_over_input"]
        .agg(["count", "median", "mean",
              lambda s: s.quantile(0.25), lambda s: s.quantile(0.75),
              lambda s: ((s >= 0.5) & (s <= 2.0)).mean(),
              lambda s: ((s >= 0.8) & (s <= 1.25)).mean()])
        .rename(columns={"<lambda_0>": "q25", "<lambda_1>": "q75",
                         "<lambda_2>": "frac_within_2x",
                         "<lambda_3>": "frac_within_1.25x"})
        .round(3)
    )
    summary.to_csv(CMP_CSV_DIR / "figure3c_summary.csv")

    # Box plot per estimator, one row per community type — mimics Fig. 3c.
    for dataset_name in ("equal_protein", "equal_cell", "uneven"):
        sub = dev_df[dev_df["dataset"] == dataset_name]
        if sub.empty:
            continue
        estimators_ordered = sub["estimator"].unique().tolist()
        fig, ax = plt.subplots(figsize=(max(11, 1.1 * len(estimators_ordered)), 5.5))
        data = [sub[sub["estimator"] == e]["measured_over_input"].to_numpy()
                for e in estimators_ordered]
        bp = ax.boxplot(data, labels=estimators_ordered, showfliers=True,
                        whis=(10, 90), patch_artist=True)
        for patch in bp["boxes"]:
            patch.set(facecolor="#cfe8f3", edgecolor="black", linewidth=0.8)
        ax.axhline(1.0, color="#2ca02c", lw=1, label="perfect = 1.0")
        ax.axhline(0.5, color="#999", lw=0.5, ls="--")
        ax.axhline(2.0, color="#999", lw=0.5, ls="--")
        ax.set_ylabel("measured % / input % (per species)")
        ax.set_title(
            f"Figure 3c-style fold deviation — {dataset_name} samples (PXD006118)\n"
            f"green = perfect, grey dashed = 0.5x / 2x; box = Q1-Q3, whiskers = 10th-90th pct",
            fontsize=10,
        )
        ax.set_yscale("log")
        ax.set_ylim(0.05, 20)
        plt.xticks(rotation=40, ha="right", fontsize=8)
        ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(CMP_CHART_DIR / f"figure3c_deviation_{dataset_name}.png", dpi=150)
        plt.close(fig)

    print("\n=== Figure 3c-style fold-deviation summary (measured % / input %) ===")
    for ds in ("equal_protein", "equal_cell", "uneven"):
        if ds in summary.index.get_level_values(0):
            print(f"\n{ds} samples:")
            print(summary.loc[ds].to_string())

    # --- Print summary table ---
    print("\n=== Means by estimator_vs_ref (P samples only, where iBAQ applies) ===")
    print(
        metrics_df[metrics_df["dataset"] == "equal_protein"]
        .groupby("estimator_vs_ref")[["R2", "Pearson_r", "RMSE", "L1"]]
        .mean()
        .round(3)
        .to_string()
    )
    print("\n=== Means by estimator_vs_ref (C samples) ===")
    print(
        metrics_df[metrics_df["dataset"] == "equal_cell"]
        .groupby("estimator_vs_ref")[["R2", "Pearson_r", "RMSE", "L1"]]
        .mean()
        .round(3)
        .to_string()
    )


if __name__ == "__main__":
    main()
