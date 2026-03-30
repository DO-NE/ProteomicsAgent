"""TSV and text summary export utilities."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from agent.state_manager import RunState
from taxon.base_plugin import TaxonResult


def export_tsv(results: list[TaxonResult], output_dir: Path, filename: str) -> str:
    """Export taxon results to TSV and return output path."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["taxon_id", "taxon_name", "rank", "abundance_pct", "confidence", "peptide_count"])
        for row in results:
            writer.writerow(
                [
                    row.taxon_id,
                    row.taxon_name,
                    row.rank,
                    f"{row.abundance * 100:.4f}",
                    f"{row.confidence:.4f}",
                    row.peptide_count,
                ]
            )
    return str(path)


def export_summary(state: RunState, results: list[TaxonResult], figure_paths: list[str], output_dir: Path) -> str:
    """Export a plain text summary report for the run."""

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.txt"
    top10 = sorted(results, key=lambda x: x.abundance, reverse=True)[:10]
    tsv_path = output_dir / "results.tsv"

    lines = [
        f"Run ID: {state.run_id}",
        f"Date: {datetime.now(timezone.utc).isoformat()}",
        f"Input files: {', '.join(state.input_files)}",
        f"Database: {state.database_path}",
        f"Stages completed: {', '.join(state.completed_stages)}",
        f"Taxon algorithm used: {state.taxon_algorithm}",
        f"Total taxa identified: {len(results)}",
        "Top 10 taxa (name, rank, abundance%):",
    ]
    for row in top10:
        lines.append(f"  - {row.taxon_name}, {row.rank}, {row.abundance * 100:.4f}%")
    lines.append("Output figures:")
    for fig in figure_paths:
        lines.append(f"  - {fig}")
    lines.append(f"TSV report: {tsv_path}")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)
