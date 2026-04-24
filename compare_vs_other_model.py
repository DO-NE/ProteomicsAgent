#!/usr/bin/env python3
"""Side-by-side comparison of our pipeline vs. another author's model,
using the three-way bar chart format:

  GT (protein)   — composition-input protein mass %
  GT (PSM)       — Kleiner 2017 supplementary PSM counts (Figure 3 source)
  Model output   — this pipeline's ibaq_em + ms1_precursor output

Pearson r and RMSE are reported against GT (PSM) so the numbers are
directly comparable to the other author's reported figures.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from compare_to_paper import load_paper_tables, normalize_sample  # noqa: E402
from run_ibaq_mock_comm import (  # noqa: E402
    COMP_P, COMP_C, COMP_U, COMMUNITY_PREFIXES, TOP_CHART_DIR, TOP_CSV_DIR,
    csv_dir, parse_composition, truth_by_species, species_to_label,
)

OUT_CHART_DIR = TOP_CHART_DIR
OUT_CHART_DIR.mkdir(parents=True, exist_ok=True)

# Pick a representative sample per community (first replicate).
REPRESENTATIVE_SAMPLE = {"C": "C1", "P": "P1", "U": "U1"}

# Our best pipeline combination (as established in the paper comparison run).
METHOD, SOURCE = "ibaq_em", "ms1_precursor"


def load_our_sample(short: str) -> dict[str, float]:
    """Load {species: abundance} for one short sample (e.g. 'P1').
    The CSVs on disk are named like P1_run3_100mM__ibaq_em.csv."""
    community = short[0]
    cdir = csv_dir(community, SOURCE)
    for p in sorted(cdir.glob(f"{short}_*__{METHOD}.csv")):
        df = pd.read_csv(p)
        if not df.empty:
            return dict(zip(df["species"], df["abundance"]))
    return {}


def metrics(est: dict[str, float], ref: dict[str, float]) -> dict[str, float]:
    species = sorted(set(est) | set(ref))
    if not species:
        return {}
    y_est = np.array([est.get(s, 0.0) for s in species])
    y_ref = np.array([ref.get(s, 0.0) for s in species])
    rmse_pct = float(np.sqrt(np.mean((y_ref * 100 - y_est * 100) ** 2)))
    pearson = (float(np.corrcoef(y_ref, y_est)[0, 1])
               if y_ref.std() > 0 and y_est.std() > 0 else float("nan"))
    ss_res = float(np.sum((y_ref - y_est) ** 2))
    ss_tot = float(np.sum((y_ref - y_ref.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {"Pearson_r": pearson, "RMSE_pct": rmse_pct, "R2": r2}


def three_bar_plot(
    sample: str, community: str,
    gt_protein: dict[str, float], gt_psm: dict[str, float], ours: dict[str, float],
    m_vs_psm: dict[str, float], m_vs_protein: dict[str, float],
    out: Path, top_n: int = 10,
) -> None:
    """Dowon-style three-way bar chart, ranked by GT (protein) descending."""
    # Rank top-N species by GT (protein), per the user request.
    # Species absent from truth but present in ours/paper fall through at 0.
    candidates = set(gt_protein) | set(gt_psm) | set(ours)
    top = sorted(candidates, key=lambda s: -gt_protein.get(s, 0.0))[:top_n]

    labels = [species_to_label(s) for s in top]
    y_prot = np.array([gt_protein.get(s, 0.0) for s in top]) * 100
    y_psm  = np.array([gt_psm.get(s, 0.0)     for s in top]) * 100
    y_our  = np.array([ours.get(s, 0.0)       for s in top]) * 100

    # Match dowon's palette (standard matplotlib C0/C2/C1-ish blues/greens/oranges).
    C_PROT, C_PSM, C_OUR = "#4878D0", "#6ACC64", "#EE854A"

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    x = np.arange(len(top))
    w = 0.27
    ax.bar(x - w, y_prot, w, label="GT (protein)",     color=C_PROT, edgecolor="none")
    ax.bar(x,     y_psm,  w, label="GT (PSM)",         color=C_PSM,  edgecolor="none")
    ax.bar(x + w, y_our,  w, label="Model prediction", color=C_OUR,  edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, fontsize=10)
    ax.set_xlabel("Taxon", fontsize=11)
    ax.set_ylabel("Relative abundance (%)", fontsize=11)
    ax.set_title(
        "GT (protein) vs. GT (PSM) vs. model prediction\n"
        f"Relative taxon abundance — {sample} (top {top_n} by GT protein)",
        fontsize=11,
    )
    ax.legend(loc="upper right", fontsize=9, frameon=True, framealpha=0.95)
    ax.tick_params(axis="y", labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", linestyle="-", alpha=0.15)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    paper = load_paper_tables()
    truth_protein = {
        "C": truth_by_species(parse_composition(COMP_C, percent_col_idx=3)),
        "P": truth_by_species(parse_composition(COMP_P, percent_col_idx=3)),
        "U": truth_by_species(parse_composition(COMP_U, percent_col_idx=3)),
    }

    rows: list[dict] = []
    for community, short_sample in REPRESENTATIVE_SAMPLE.items():
        gt_protein = truth_protein[community]
        gt_psm     = normalize_sample(paper["psm"].get(short_sample, {}))
        ours       = load_our_sample(short_sample)

        m_vs_psm     = metrics(ours, gt_psm)
        m_vs_protein = metrics(ours, gt_protein)
        m_psm_vs_prot = metrics(gt_psm, gt_protein)

        rows.append({
            "community": community, "sample": short_sample,
            "ours_vs_psm_r":      round(m_vs_psm.get("Pearson_r", np.nan), 3),
            "ours_vs_psm_rmse%":  round(m_vs_psm.get("RMSE_pct",  np.nan), 2),
            "ours_vs_psm_R2":     round(m_vs_psm.get("R2",        np.nan), 3),
            "ours_vs_protein_r":  round(m_vs_protein.get("Pearson_r", np.nan), 3),
            "ours_vs_protein_rmse%": round(m_vs_protein.get("RMSE_pct", np.nan), 2),
            "psm_vs_protein_r":   round(m_psm_vs_prot.get("Pearson_r", np.nan), 3),
        })

        out_png = OUT_CHART_DIR / f"threeway_{community}_{short_sample}.png"
        three_bar_plot(
            short_sample, community,
            gt_protein, gt_psm, ours,
            m_vs_psm, m_vs_protein, out_png,
        )
        print(f"wrote {out_png}")

    df = pd.DataFrame(rows)
    out_csv = TOP_CSV_DIR / "three_way_comparison_vs_paperPSM.csv"
    df.to_csv(out_csv, index=False)

    print("\n=== Metrics (our model vs. GT) ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
