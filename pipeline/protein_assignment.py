"""Protein-level inference using ProteinProphet and protXML parsing."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

from .base import PipelineStage


class ProteinAssignment(PipelineStage):
    """Run ProteinProphet to infer protein groups."""

    name = "protein_assignment"
    tools = ["proteinprophet"]

    def run(self, input_path: str, params: dict, dry_run: bool = False) -> str:
        """Run ProteinProphet and return proteins.prot.xml output path."""

        pepxml = Path(input_path)
        run_dir = Path(params["run_dir"])
        outdir = run_dir / "protein"
        outdir.mkdir(parents=True, exist_ok=True)
        output_file = outdir / "proteins.prot.xml"

        tpp_bin = Path(params.get("tpp_bin_path", ""))
        cmd = [str(tpp_bin / "ProteinProphet"), str(pepxml.resolve()), str(output_file.resolve())]
        self.execute(cmd, self.name, "ProteinProphet", dry_run=dry_run)
        return str(output_file)


def parse_protxml(protxml_path: str) -> list[dict]:
    """Parse protXML and extract per-protein summary fields."""

    path = Path(protxml_path)
    if not path.exists():
        return []

    tree = ET.parse(path)
    root = tree.getroot()
    ns = ""
    if root.tag.startswith("{"):
        ns = root.tag.split("}")[0].strip("{")

    def q(name: str) -> str:
        return f"{{{ns}}}{name}" if ns else name

    records: list[dict] = []
    for protein in root.iter(q("protein")):
        records.append(
            {
                "protein_name": protein.attrib.get("protein_name", ""),
                "probability": float(protein.attrib.get("probability", "0")),
                "unique_peptides": int(protein.attrib.get("n_indistinguishable_proteins", "0")),
                "total_peptides": int(protein.attrib.get("total_number_peptides", "0")),
                "percent_coverage": float(protein.attrib.get("percent_coverage", "0")),
            }
        )
    return records
