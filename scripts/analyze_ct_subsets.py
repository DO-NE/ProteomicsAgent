#!/usr/bin/env python3
"""Post-run analysis of the three c_t subsets from ``abundance_results.tsv``.

Compares ``c_t_all`` / ``c_t_ribo`` / ``c_t_nayfach`` per taxon, flags
qualifying-rule flips between subsets, and (optionally) computes L1
distances against a ground-truth TSV.

Usage::

    python scripts/analyze_ct_subsets.py \\
        --results <path/to/abundance_results.tsv> \\
        [--ground-truth <path/to/ground_truth.tsv>] \\
        [--output <path/to/comparison.md>]

The ground-truth TSV should have columns
``taxon, gt_protein, gt_cell`` (taxon names matched against
``taxon_name`` from abundance_results.tsv, case-insensitive substring
match; rows in the results table that do not match are scored against
zero ground truth).

Output is markdown sent to ``--output`` (defaults to stdout).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


SUBSET_NAMES = ("all", "ribo", "nayfach")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--results", type=Path, required=True,
        help="Path to abundance_results.tsv produced by the pipeline.",
    )
    ap.add_argument(
        "--ground-truth", type=Path, default=None,
        help="Optional TSV with columns taxon, gt_protein, gt_cell.",
    )
    ap.add_argument(
        "--output", type=Path, default=None,
        help="Where to write the markdown report (default: stdout).",
    )
    return ap.parse_args(argv)


def load_results(path: Path) -> pd.DataFrame:
    """Read abundance_results.tsv, skipping the leading ``#`` comment line."""
    return pd.read_csv(path, sep="\t", comment="#")


def load_ground_truth(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    expected = {"taxon", "gt_protein", "gt_cell"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"Ground-truth TSV {path} missing columns: {sorted(missing)}"
        )
    return df


def _match_ground_truth(taxon_name: str, gt: pd.DataFrame) -> tuple[float, float]:
    """Find a GT row whose ``taxon`` is a case-insensitive substring of *taxon_name*."""
    nlow = (taxon_name or "").lower()
    for _, row in gt.iterrows():
        if str(row["taxon"]).lower() in nlow:
            return float(row["gt_protein"]), float(row["gt_cell"])
    return 0.0, 0.0


def l1(predicted: pd.Series, actual: pd.Series) -> float:
    """L1 distance between two same-length series; missing entries treated as 0."""
    p = predicted.fillna(0.0).to_numpy()
    a = actual.fillna(0.0).to_numpy()
    return float(abs(p - a).sum())


def _format_cell(value, is_float: bool) -> str:
    if value is None:
        return ""
    if is_float:
        try:
            return f"{float(value):.4f}"
        except (TypeError, ValueError):
            return str(value)
    return str(value)


def _df_to_markdown(df: pd.DataFrame) -> str:
    """Stdlib-only markdown-table renderer (avoids the tabulate dep)."""
    if df.empty:
        return "_(no rows)_"
    cols = list(df.columns)
    is_float = [pd.api.types.is_float_dtype(df[c]) for c in cols]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = [header, sep]
    for _, r in df.iterrows():
        rows.append(
            "| " + " | ".join(
                _format_cell(r[c], f) for c, f in zip(cols, is_float)
            ) + " |"
        )
    return "\n".join(rows)


def render_per_taxon_table(df: pd.DataFrame) -> str:
    """Sorted-by-psm markdown table of (taxon, pi*, b_t, c_t_*, n_fam_*)."""
    cols = [
        "taxon_id", "taxon_name", "psm_abundance", "biomass_abundance",
        "c_t_all", "c_t_ribo", "c_t_nayfach",
        "n_families_all", "n_families_ribo", "n_families_nayfach",
    ]
    have = [c for c in cols if c in df.columns]
    sorted_df = df.sort_values("psm_abundance", ascending=False)[have]
    return _df_to_markdown(sorted_df)


def flag_has_marker_flips(df: pd.DataFrame) -> list:
    """Return rows where ``has_marker_*`` is True in some subset but False in another."""
    flips: list = []
    cols = ("has_marker_all", "has_marker_ribo", "has_marker_nayfach")
    if not all(c in df.columns for c in cols):
        return flips
    for _, row in df.iterrows():
        vals = {c: bool(row[c]) for c in cols}
        if len(set(vals.values())) > 1:
            flips.append((row["taxon_name"], vals))
    return flips


def flag_nayfach_zero_but_all_qualifies(df: pd.DataFrame) -> list:
    """c_t_nayfach == 0 but has_marker_all is True."""
    if "c_t_nayfach" not in df.columns or "has_marker_all" not in df.columns:
        return []
    rows: list = []
    for _, row in df.iterrows():
        if bool(row["has_marker_all"]) and float(row["c_t_nayfach"]) == 0.0:
            rows.append(row["taxon_name"])
    return rows


def render_report(args: argparse.Namespace) -> str:
    df = load_results(args.results)
    out: list = []
    out.append(f"# c_t subset comparison — `{args.results}`")
    out.append("")
    out.append(f"- Rows: {len(df)}")
    for name in SUBSET_NAMES:
        col = f"has_marker_{name}"
        if col in df.columns:
            n_qual = int(df[col].sum())
            out.append(f"- Qualifying taxa (c_t_{name}): {n_qual}")
    out.append("")
    out.append("## Per-taxon table (sorted by psm_abundance)")
    out.append("")
    out.append(render_per_taxon_table(df))
    out.append("")

    flips = flag_has_marker_flips(df)
    out.append(f"## has_marker_* flips between subsets: {len(flips)}")
    for name, vals in flips:
        out.append(f"- **{name}** — " + ", ".join(f"{k}={v}" for k, v in vals.items()))
    out.append("")

    nay_zero = flag_nayfach_zero_but_all_qualifies(df)
    out.append(
        f"## Taxa qualifying for c_t_all but with c_t_nayfach == 0: {len(nay_zero)}"
    )
    for n in nay_zero:
        out.append(f"- {n}")
    out.append("")

    if args.ground_truth is not None:
        gt = load_ground_truth(args.ground_truth)
        gt_protein: list = []
        gt_cell: list = []
        for _, row in df.iterrows():
            p, c = _match_ground_truth(str(row.get("taxon_name", "")), gt)
            gt_protein.append(p)
            gt_cell.append(c)
        df["gt_protein"] = gt_protein
        df["gt_cell"] = gt_cell
        out.append("## L1 vs ground truth")
        out.append("")
        out.append("| metric | L1 vs gt_protein | L1 vs gt_cell |")
        out.append("|---|---|---|")
        candidates = ("psm_abundance", "biomass_abundance",
                      "c_t_all", "c_t_ribo", "c_t_nayfach")
        for col in candidates:
            if col not in df.columns:
                continue
            out.append(
                f"| {col} | {l1(df[col], df['gt_protein']):.4f} "
                f"| {l1(df[col], df['gt_cell']):.4f} |"
            )

    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.results.is_file():
        print(f"error: results file not found: {args.results}", file=sys.stderr)
        return 2
    report = render_report(args)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"report written to {args.output}")
    else:
        sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
