"""Peptide validation stage using PeptideProphet or Percolator."""

from __future__ import annotations

from pathlib import Path

from config import resolve_tpp_binary
from .base import PipelineStage


class PeptideValidation(PipelineStage):
    """Validate identified peptides with selected validation method."""

    name = "validation"
    tools = ["peptideprophet", "percolator"]

    def run(self, input_path: str, params: dict, dry_run: bool = False) -> str:
        """Run selected validation tool and return output path."""

        tool = params.get("tool", "peptideprophet").lower()
        pepxml = Path(input_path)
        run_dir = Path(params["run_dir"])
        outdir = run_dir / "validation"
        outdir.mkdir(parents=True, exist_ok=True)

        if tool == "peptideprophet":
            tpp_bin_path = params.get("tpp_bin_path", "")
            binary = resolve_tpp_binary(tpp_bin_path, ["PeptideProphetParser", "PeptideProphet"])
            if not binary:
                raise RuntimeError(f"PeptideProphet binary not found in {tpp_bin_path}")
            cmd = [binary, str(pepxml), "DECOY=DECOY_", "NONPARAM", "ACCMASS"]
            self.execute(cmd, self.name, "PeptideProphet", dry_run=dry_run)
            return str(outdir / f"interact-{pepxml.stem}.pep.xml")

        percolator_path = params.get("percolator_path", "")
        output_file = outdir / "percolator_psms.txt"
        cmd = [percolator_path, "--results-psms", str(output_file), str(pepxml)]
        self.execute(cmd, self.name, "percolator", dry_run=dry_run)
        return str(output_file)
