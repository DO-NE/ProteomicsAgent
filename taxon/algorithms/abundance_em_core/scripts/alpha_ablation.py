"""Ablation over the genome-scaling exponent α for proteome-mass correction.

Re-runs :func:`taxon.algorithms.abundance_em_core.proteome_mass_correction.
compute_biomass_abundance` for a list of α values against an existing
``abundance_results.tsv`` and writes a comparison table, a grouped
bar-chart plot, and a per-α summary of L1 distance / Bray–Curtis
dissimilarity vs π_hat.

The script reuses the *production* correction function — the formula
``b_t = π_t / W_t^α`` is not duplicated here.

Usage
-----
.. code-block:: bash

    python alpha_ablation.py \\
        --input /path/to/abundance_results.tsv \\
        --alphas 0.0,1.0,2.0,3.6,4.8 \\
        --output alpha_ablation/

Input TSV schema (from ``_write_unified_results``):

    taxon_id  taxon_name  psm_abundance  biomass_abundance  cell_abundance
    proteome_size  marker_families  marker_psms  has_marker_estimate

Outputs (under ``--output``):

* ``alpha_ablation_table.tsv`` — one row per taxon, columns
  ``taxon, pi_hat, W_t, b_t_alpha_<value>`` for each α.
* ``alpha_ablation_plot.png`` — grouped matplotlib bar chart of the
  top-15 taxa by π_hat, with b_t bars side-by-side per α.
* ``alpha_ablation_summary.txt`` — per α, L1 distance and Bray–Curtis
  dissimilarity vs π_hat.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

# Make the script runnable as ``python alpha_ablation.py`` from anywhere
# (without needing PYTHONPATH gymnastics) by walking up to the repo root.
_HERE = Path(__file__).resolve()
_REPO_ROOT = _HERE.parents[4]   # …/scripts/ → core/ → algorithms/ → taxon/ → repo
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from taxon.algorithms.abundance_em_core.proteome_mass_correction import (  # noqa: E402
    compute_biomass_abundance,
)


# ----------------------------------------------------------- I/O helpers


def parse_alphas(spec: str) -> list[float]:
    """Parse a comma-separated alpha specification into floats."""
    out: list[float] = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        out.append(float(tok))
    if not out:
        raise ValueError("--alphas must contain at least one value")
    return out


def read_abundance_results(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Read ``abundance_results.tsv`` and extract the columns we need.

    Returns
    -------
    labels : list[str]
        ``"<taxon_id>|<taxon_name>"`` for each row, matching the
        canonical mapping-matrix label format consumed by
        :func:`compute_biomass_abundance`.
    pi : np.ndarray, shape (T,)
        ``psm_abundance`` column.
    W : np.ndarray, shape (T,)
        ``proteome_size`` column. Zero entries are kept verbatim and
        will be patched to W=1 by the production correction function.
    """
    labels: list[str] = []
    pi_vals: list[float] = []
    W_vals: list[float] = []
    with path.open("r", encoding="utf-8", newline="") as fh:
        # abundance_results.tsv begins with a "#"-prefixed provenance line;
        # csv has no comment-skip option, so filter manually before parsing.
        rows = (line for line in fh if not line.lstrip().startswith("#"))
        reader = csv.DictReader(rows, delimiter="\t")
        required = {"taxon_id", "taxon_name", "psm_abundance", "proteome_size"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{path} is missing required columns: {sorted(missing)}; "
                f"got {reader.fieldnames}"
            )
        for row in reader:
            tid = row["taxon_id"]
            tname = row["taxon_name"]
            labels.append(f"{tid}|{tname}")
            pi_vals.append(float(row["psm_abundance"]))
            W_vals.append(float(row["proteome_size"]))
    pi = np.asarray(pi_vals, dtype=np.float64)
    W = np.asarray(W_vals, dtype=np.float64)

    # Renormalize π in case the input table was post-filtered (e.g.
    # min_abundance / min_psm thresholds) and no longer sums to 1 —
    # without this, b_t would inherit the truncation bias.
    pi_sum = float(pi.sum())
    if pi_sum > 0 and not np.isclose(pi_sum, 1.0):
        pi = pi / pi_sum
    return labels, pi, W


# ------------------------------------------------- ablation core


def run_alpha_sweep(
    pi: np.ndarray,
    W: np.ndarray,
    labels: list[str],
    alphas: list[float],
) -> dict[float, np.ndarray]:
    """Compute b_t for each α via the production function.

    Returns ``{alpha: b_t_vector}``.  Raising ``ValueError`` from
    a negative α propagates to the caller — we do not silently
    swallow misconfiguration here.
    """
    out: dict[float, np.ndarray] = {}
    for a in alphas:
        result = compute_biomass_abundance(
            pi=pi, proteome_sizes=W, taxon_labels=labels, alpha=a,
        )
        out[a] = np.asarray(result.biomass_abundance, dtype=np.float64)
    return out


def l1_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Σ |p_i - q_i|."""
    return float(np.abs(p - q).sum())


def bray_curtis(p: np.ndarray, q: np.ndarray) -> float:
    """Bray–Curtis dissimilarity 1 − 2·Σmin(p,q) / (Σp + Σq).

    For two probability vectors that both sum to 1 this reduces to
    L1/2, but we compute the general form to stay robust to
    pre-normalization quirks.
    """
    s = float(p.sum() + q.sum())
    if s <= 0:
        return float("nan")
    return 1.0 - 2.0 * float(np.minimum(p, q).sum()) / s


def _alpha_label(a: float) -> str:
    """Render α as a stable, filename-friendly token."""
    return f"{a:g}"


# ------------------------------------------------- output writers


def write_table(
    labels: list[str],
    pi: np.ndarray,
    W: np.ndarray,
    bt_per_alpha: dict[float, np.ndarray],
    alphas: list[float],
    out_path: Path,
) -> None:
    """Write the per-taxon comparison table."""
    with out_path.open("w", encoding="utf-8") as fh:
        header = ["taxon", "pi_hat", "W_t"] + [
            f"b_t_alpha_{_alpha_label(a)}" for a in alphas
        ]
        fh.write("\t".join(header) + "\n")
        for i, lbl in enumerate(labels):
            row = [
                lbl,
                f"{float(pi[i]):.6f}",
                f"{int(W[i])}" if float(W[i]).is_integer() else f"{float(W[i]):.6f}",
            ]
            for a in alphas:
                row.append(f"{float(bt_per_alpha[a][i]):.6f}")
            fh.write("\t".join(row) + "\n")


def write_summary(
    pi: np.ndarray,
    bt_per_alpha: dict[float, np.ndarray],
    alphas: list[float],
    out_path: Path,
    input_path: Path,
) -> None:
    """Write per-α L1 / Bray–Curtis summary."""
    lines: list[str] = []
    lines.append("=== Alpha Ablation Summary ===")
    lines.append(f"Input: {input_path}")
    lines.append(f"Taxa: {len(pi)}")
    lines.append("")
    lines.append(f"{'alpha':>8} {'L1(b vs pi)':>14} {'BrayCurtis':>12}")
    for a in alphas:
        b = bt_per_alpha[a]
        lines.append(
            f"{a:>8.4f} {l1_distance(b, pi):>14.6f} {bray_curtis(b, pi):>12.6f}"
        )
    lines.append("")
    lines.append("Notes:")
    lines.append("  - alpha=0 should give b_t == pi_hat; both metrics ~0.")
    lines.append("  - alpha=1 is the original linear TPA form (Pible 2020 / Kleiner 2017).")
    lines.append("  - alpha=4.8 follows Kempes-2016 bacterial genome-volume scaling.")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(
    labels: list[str],
    pi: np.ndarray,
    bt_per_alpha: dict[float, np.ndarray],
    alphas: list[float],
    out_path: Path,
    top_n: int = 15,
) -> None:
    """Grouped bar chart: top-15 taxa by π_hat, b_t bars side-by-side per α."""
    import matplotlib

    # Use a non-interactive backend so this works in headless / CI settings.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    order = np.argsort(-pi)[:top_n]
    short_names = [
        labels[i].split("|", 1)[-1][:30] for i in order
    ]

    n_alpha = len(alphas)
    n_taxa = len(order)
    bar_width = 0.8 / (n_alpha + 1)   # +1 leaves room for the π_hat bar
    x = np.arange(n_taxa)

    fig, ax = plt.subplots(figsize=(max(10, 0.6 * n_taxa + 4), 6))

    # π_hat reference bar.
    ax.bar(
        x - 0.4 + 0.5 * bar_width,
        pi[order],
        width=bar_width,
        label="π_hat",
        color="black",
        alpha=0.85,
    )
    for k, a in enumerate(alphas):
        offset = -0.4 + (k + 1.5) * bar_width
        ax.bar(
            x + offset,
            bt_per_alpha[a][order],
            width=bar_width,
            label=f"α={_alpha_label(a)}",
            alpha=0.85,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right")
    ax.set_ylabel("Relative abundance")
    ax.set_title(f"Genome-scaling exponent ablation (top {n_taxa} taxa by π_hat)")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ----------------------------------------------------------- entry point


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Ablate the genome-scaling exponent α used in the proteome-mass "
            "correction. Reads an existing abundance_results.tsv, recomputes "
            "b_t for each requested α via the production correction function, "
            "and writes a comparison table + plot + summary."
        )
    )
    parser.add_argument(
        "--input", required=True, type=Path,
        help="Path to abundance_results.tsv.",
    )
    parser.add_argument(
        "--alphas", default="0.0,1.0,2.0,3.6,4.8",
        help="Comma-separated α values to evaluate (default: 0.0,1.0,2.0,3.6,4.8).",
    )
    parser.add_argument(
        "--output", required=True, type=Path,
        help="Output directory; created if it does not exist.",
    )
    parser.add_argument(
        "--top-n", type=int, default=15,
        help="Top N taxa (by π_hat) to show in the bar chart (default 15).",
    )
    args = parser.parse_args(argv)

    if not args.input.is_file():
        parser.error(f"--input file does not exist: {args.input}")

    alphas = parse_alphas(args.alphas)
    out_dir: Path = args.output
    out_dir.mkdir(parents=True, exist_ok=True)

    labels, pi, W = read_abundance_results(args.input)
    if len(labels) == 0:
        parser.error(f"--input file {args.input} contains no rows")

    bt_per_alpha = run_alpha_sweep(pi, W, labels, alphas)

    table_path = out_dir / "alpha_ablation_table.tsv"
    plot_path = out_dir / "alpha_ablation_plot.png"
    summary_path = out_dir / "alpha_ablation_summary.txt"

    write_table(labels, pi, W, bt_per_alpha, alphas, table_path)
    write_summary(pi, bt_per_alpha, alphas, summary_path, args.input)
    try:
        make_plot(labels, pi, bt_per_alpha, alphas, plot_path, top_n=args.top_n)
    except Exception as exc:  # noqa: BLE001 — never fail a run on a plot
        print(f"[warning] plot generation failed: {exc}", file=sys.stderr)

    print(f"Wrote {table_path}")
    print(f"Wrote {summary_path}")
    if plot_path.exists():
        print(f"Wrote {plot_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
