"""Quantitation stage implementation for spectral counting and ASAPRatio."""

from __future__ import annotations

import csv
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

from .base import PipelineStage


class Quantitation(PipelineStage):
    """Quantify validated peptides and proteins."""

    name = "quantitation"
    tools = ["spectral_counting", "asap_ratio"]

    def run(self, input_path: str, params: dict, dry_run: bool = False) -> str:
        """Run quantitation mode and return generated output path."""

        tool = params.get("tool", "spectral_counting").lower()
        pepxml = Path(input_path)
        run_dir = Path(params["run_dir"])
        outdir = run_dir / "quant"
        outdir.mkdir(parents=True, exist_ok=True)

        if tool == "asap_ratio":
            tpp_bin = Path(params.get("tpp_bin_path", ""))
            cmd = [str(tpp_bin / "ASAPRatio"), str(pepxml), "-F"]
            self.execute(cmd, self.name, "ASAPRatio", dry_run=dry_run)
            return str(pepxml)

        if dry_run:
            return str(outdir / "spectral_counts.tsv")

        tree = ET.parse(pepxml)
        root = tree.getroot()
        ns = ""
        if root.tag.startswith("{"):
            ns = root.tag.split("}")[0].strip("{")

        def q(name: str) -> str:
            return f"{{{ns}}}{name}" if ns else name

        proteins: dict[str, dict[str, object]] = defaultdict(lambda: {"peptides": set(), "count": 0, "probs": []})

        for hit in root.iter(q("search_hit")):
            peptide = hit.attrib.get("peptide", "")
            protein_id = hit.attrib.get("protein", "")
            probability = 0.0
            analysis = hit.find(q("analysis_result"))
            if analysis is not None:
                pp = analysis.find(q("peptideprophet_result"))
                if pp is not None:
                    probability = float(pp.attrib.get("probability", "0"))
            if probability < 0.95 or not protein_id:
                continue
            proteins[protein_id]["count"] = int(proteins[protein_id]["count"]) + 1
            if peptide:
                proteins[protein_id]["peptides"].add(peptide)
            proteins[protein_id]["probs"].append(probability)

        out_path = outdir / "spectral_counts.tsv"
        with out_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle, delimiter="\t")
            writer.writerow(["protein_id", "peptide_count", "spectral_count", "avg_probability"])
            for protein_id, stats in sorted(proteins.items()):
                probs: list[float] = stats["probs"]  # type: ignore[assignment]
                avg_prob = sum(probs) / len(probs) if probs else 0.0
                writer.writerow(
                    [
                        protein_id,
                        len(stats["peptides"]),  # type: ignore[arg-type]
                        stats["count"],
                        f"{avg_prob:.4f}",
                    ]
                )

        return str(out_path)
