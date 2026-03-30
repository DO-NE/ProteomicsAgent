"""Static figure generation utilities for metaproteomics outputs."""

from __future__ import annotations

import itertools
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib.patches import Patch

from taxon.base_plugin import TaxonResult

sns.set_theme(style="whitegrid", font="DejaVu Sans")


def _save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> list[str]:
    """Save a Matplotlib figure to PNG and PDF and return file paths."""

    output_dir.mkdir(parents=True, exist_ok=True)
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [str(png), str(pdf)]


def taxon_bar_chart(results: list[TaxonResult], output_dir: Path, top_n: int = 20) -> list[str]:
    """Create horizontal taxon abundance bar chart for top taxa."""

    selected = sorted(results, key=lambda r: r.abundance, reverse=True)[:top_n]
    if not selected:
        fig, _ax = plt.subplots(figsize=(10, 5))
        return _save_figure(fig, output_dir, "taxon_bar_chart")

    rank_colors = {"species": "#4AAFA0", "genus": "#E07B54", "family": "#888888"}
    names = [r.taxon_name for r in selected]
    vals = [r.abundance * 100 for r in selected]
    colors = [rank_colors.get(r.rank, "#CCCCCC") for r in selected]

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(names, vals, color=colors)
    ax.invert_yaxis()
    ax.set_xlabel("Relative abundance (%)")
    ax.set_title(f"Taxon composition — top {top_n}")

    for bar, val in zip(bars, vals, strict=False):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=8)

    legend_items = [
        Patch(color="#4AAFA0", label="species"),
        Patch(color="#E07B54", label="genus"),
        Patch(color="#888888", label="family"),
        Patch(color="#CCCCCC", label="other"),
    ]
    ax.legend(handles=legend_items, title="Rank", loc="lower right")
    return _save_figure(fig, output_dir, "taxon_bar_chart")


def taxon_pie_chart(results: list[TaxonResult], output_dir: Path, top_n: int = 10) -> list[str]:
    """Create pie chart of top taxon abundances with remainder as Other."""

    selected = sorted(results, key=lambda r: r.abundance, reverse=True)
    top = selected[:top_n]
    other_abundance = max(0.0, 1.0 - sum(r.abundance for r in top))

    labels = [r.taxon_name for r in top]
    values = [r.abundance for r in top]
    if other_abundance > 0:
        labels.append("Other")
        values.append(other_abundance)

    palette = itertools.cycle(sns.color_palette("colorblind", n_colors=8))
    colors = [next(palette) for _ in labels]

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, texts, autotexts = ax.pie(
        values,
        labels=labels,
        autopct=lambda pct: f"{pct:.1f}%",
        pctdistance=1.15,
        labeldistance=1.35,
        colors=colors,
        startangle=90,
    )
    for text in texts + autotexts:
        text.set_fontsize(8)
    ax.set_title("Taxon composition")
    ax.axis("equal")
    return _save_figure(fig, output_dir, "taxon_pie_chart")


def _extract_probabilities(pepxml_path: str) -> list[tuple[str, float]]:
    """Extract peptide sequences and PeptideProphet probabilities from pepXML."""

    path = Path(pepxml_path)
    if not path.exists():
        return []

    tree = ET.parse(path)
    root = tree.getroot()
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0].strip("{")

    def q(name: str) -> str:
        return f"{{{ns}}}{name}" if ns else name

    records: list[tuple[str, float]] = []
    for hit in root.iter(q("search_hit")):
        peptide = hit.attrib.get("peptide", "")
        probability = 0.0
        for ar in hit.findall(q("analysis_result")):
            pp = ar.find(q("peptideprophet_result"))
            if pp is not None:
                probability = float(pp.attrib.get("probability", "0"))
                break
        if peptide:
            records.append((peptide, probability))
    return records


def peptide_heatmap(pepxml_path: str, output_dir: Path) -> list[str]:
    """Generate a single-column heatmap of top 50 peptide probabilities."""

    records = sorted(_extract_probabilities(pepxml_path), key=lambda x: x[1], reverse=True)[:50]
    df = pd.DataFrame(records, columns=["peptide", "probability"]).set_index("peptide")

    fig, ax = plt.subplots(figsize=(8, 12))
    if not df.empty:
        sns.heatmap(df, cmap="YlOrRd", annot=True, fmt=".3f", cbar=True, ax=ax)
    ax.set_title("Top 50 peptides by PeptideProphet probability")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Peptide sequence")
    return _save_figure(fig, output_dir, "peptide_heatmap")


def score_distribution(pepxml_path: str, output_dir: Path) -> list[str]:
    """Generate histogram and KDE of peptide probabilities."""

    probs = [p for _pep, p in _extract_probabilities(pepxml_path)]
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(probs, kde=True, ax=ax, bins=30)
    ax.axvline(0.95, color="red", linestyle="--", linewidth=1.5, label="FDR threshold")
    ax.set_xlabel("PeptideProphet probability")
    ax.set_ylabel("Count")
    ax.set_title("Peptide score distribution")
    ax.legend()
    return _save_figure(fig, output_dir, "score_distribution")


def abundance_boxplot(results_by_condition: dict[str, list[TaxonResult]], output_dir: Path) -> list[str]:
    """Generate grouped taxon abundance boxplot by condition."""

    rows: list[dict[str, object]] = []
    for condition, results in results_by_condition.items():
        for result in results:
            rows.append(
                {
                    "condition": condition,
                    "taxon_name": result.taxon_name,
                    "abundance_pct": result.abundance * 100,
                }
            )

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    if not df.empty:
        sns.boxplot(data=df, x="taxon_name", y="abundance_pct", hue="condition", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_ylabel("Relative abundance (%)")
    ax.set_title("Taxon abundance by condition")
    return _save_figure(fig, output_dir, "abundance_boxplot")
