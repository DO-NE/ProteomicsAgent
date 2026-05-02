"""Bar-chart visualization of the unified abundance results table.

Reads ``abundance_results.tsv`` (the unified output written by
:class:`AbundanceEMPlugin`) and renders a grouped bar chart showing the
PSM-level abundance plus any post-EM correction vectors that were
computed in the run (biomass and / or cell-equivalent abundance).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import matplotlib

# Headless servers commonly lack a display — Agg renders to file without one.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)


_PSM_COLOR = "#4472C4"
_BIOMASS_COLOR = "#E26B0A"
_CELL_COLOR = "#548235"
_BG_COLOR = "#F8F8F6"


def _truncate(label: str, max_len: int = 20) -> str:
    """Return *label* clipped to *max_len* characters with an ellipsis."""

    if len(label) <= max_len:
        return label
    return label[: max(1, max_len - 1)] + "…"


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation that returns ``nan`` instead of raising on edges."""

    if a.size < 2 or b.size < 2 or a.size != b.size:
        return float("nan")
    if np.allclose(a, a[0]) or np.allclose(b, b[0]):
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def plot_abundance_results(
    unified_result_path: str | Path,
    output_path: str | Path,
    top_n: int = 15,
) -> None:
    """Read ``abundance_results.tsv`` and save a bar-chart PNG.

    Parameters
    ----------
    unified_result_path : path to ``abundance_results.tsv`` (the unified
        output written by :class:`AbundanceEMPlugin`).
    output_path : where to save the PNG.
    top_n : number of top taxa (by ``psm_abundance``) to display.
    """

    unified_path = Path(unified_result_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not unified_path.is_file():
        logger.warning(
            "plot_abundance_results: unified result file not found at %s",
            unified_path,
        )
        return

    df = pd.read_csv(unified_path, sep="\t")
    if df.empty:
        logger.warning("plot_abundance_results: unified result file is empty")
        return

    # Defensive: drop CRAP rows (the unified writer already excludes them,
    # but visualisation should not silently break if the file is hand-edited).
    if "taxon_name" in df.columns:
        crap_mask = df["taxon_name"].astype(str).str.contains(
            "crap", case=False, na=False, regex=False
        )
        df = df.loc[~crap_mask].copy()

    if "psm_abundance" not in df.columns:
        logger.warning(
            "plot_abundance_results: column 'psm_abundance' missing from %s",
            unified_path,
        )
        return

    df = df.loc[df["psm_abundance"] > 0].copy()
    if df.empty:
        logger.warning(
            "plot_abundance_results: no taxa with positive psm_abundance to plot"
        )
        return

    df = df.sort_values("psm_abundance", ascending=False).head(int(top_n))

    psm_full = df["psm_abundance"].to_numpy(dtype=float)
    has_biomass = "biomass_abundance" in df.columns and not np.allclose(
        df["biomass_abundance"].to_numpy(dtype=float), psm_full
    )
    has_cell = "cell_abundance" in df.columns and not np.allclose(
        df["cell_abundance"].to_numpy(dtype=float), psm_full
    )

    series: list[tuple[str, np.ndarray, str]] = [
        ("PSM abundance", psm_full * 100.0, _PSM_COLOR),
    ]
    if has_biomass:
        series.append((
            "Biomass abundance",
            df["biomass_abundance"].to_numpy(dtype=float) * 100.0,
            _BIOMASS_COLOR,
        ))
    if has_cell:
        series.append((
            "Cell-equivalent abundance",
            df["cell_abundance"].to_numpy(dtype=float) * 100.0,
            _CELL_COLOR,
        ))

    n_groups = len(df)
    n_series = len(series)
    group_width = 0.8
    bar_width = group_width / n_series
    indices = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(11, 6), dpi=150)
    fig.patch.set_facecolor(_BG_COLOR)
    ax.set_facecolor(_BG_COLOR)

    for i, (label, values, color) in enumerate(series):
        offset = (i - (n_series - 1) / 2.0) * bar_width
        ax.bar(
            indices + offset,
            values,
            width=bar_width,
            color=color,
            label=label,
            edgecolor="white",
            linewidth=0.5,
        )

    tick_labels = [_truncate(name) for name in df["taxon_name"].astype(str).tolist()]
    rotation = 30 if n_groups > 10 else 0
    ax.set_xticks(indices)
    ax.set_xticklabels(
        tick_labels,
        rotation=rotation,
        ha="right" if rotation else "center",
        fontsize=9,
    )

    ax.set_ylabel("Relative abundance (%)", fontsize=11)
    ax.set_title(
        f"Top {n_groups} taxa by PSM abundance",
        fontsize=13,
        loc="left",
        pad=12,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")
    ax.yaxis.grid(True, color="#DDDDDD", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", colors="#444444")
    ax.tick_params(axis="y", colors="#444444")

    ax.legend(loc="upper right", frameon=False, fontsize=9)

    # Stats box: Pearson r between PSM and any correction vector.
    stats_lines: list[str] = []
    for label, values, _color in series[1:]:
        r = _pearson(psm_full * 100.0, values)
        if not np.isnan(r):
            stats_lines.append(f"r(PSM, {label.split()[0].lower()}) = {r:+.3f}")
    if stats_lines:
        ax.text(
            0.985,
            0.74,
            "\n".join(stats_lines),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            family="monospace",
            bbox=dict(facecolor="white", edgecolor="#BBBBBB", boxstyle="round,pad=0.4"),
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=_BG_COLOR)
    plt.close(fig)
    logger.info("Wrote abundance plot: %s", out_path)


def _present_correction_columns(columns: Iterable[str]) -> tuple[bool, bool]:
    """Probe an iterable of column names for biomass / cell columns."""

    cols = set(columns)
    return ("biomass_abundance" in cols, "cell_abundance" in cols)
