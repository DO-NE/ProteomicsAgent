"""Peptide identification stage using MSFragger or Comet."""

from __future__ import annotations

from pathlib import Path

from .base import PipelineStage


class PeptideIdentification(PipelineStage):
    """Run peptide-spectrum matching with configurable search engine."""

    name = "peptide_id"
    tools = ["msfragger", "comet"]

    def run(self, input_path: str, params: dict, dry_run: bool = False) -> str:
        """Run the selected search engine and return produced pepXML path."""

        tool = params.get("tool", "msfragger").lower()
        input_mzml = Path(input_path)
        run_dir = Path(params["run_dir"])
        params_dir = run_dir / "params"
        params_dir.mkdir(parents=True, exist_ok=True)
        database_path = params.get("database_path", "")

        if tool == "msfragger":
            msfragger_path = params.get("msfragger_path", "")
            param_file = params_dir / "fragger.params"
            param_file.write_text(
                "\n".join(
                    [
                        f"database_name = {database_path}",
                        "precursor_mass_tolerance = 10",
                        "precursor_mass_units = 1",
                        "fragment_mass_tolerance = 0.02",
                        "fragment_mass_units = 1",
                        "num_enzyme_termini = 2",
                        "allowed_missed_cleavage_1 = 1",
                        "search_enzyme_name_1 = trypsin",
                        "minimum_length = 7",
                        "maximum_length = 50",
                        "decoy_prefix = DECOY_",
                    ]
                ),
                encoding="utf-8",
            )
            cmd = ["java", "-jar", msfragger_path, str(param_file), str(input_mzml)]
            self.execute(cmd, self.name, "msfragger", dry_run=dry_run)
            return str(input_mzml.with_suffix(".pepXML"))

        comet_path = params.get("comet_path", "")
        comet_params_path = params.get("comet_params_path", "")
        param_file = params_dir / "comet.params"
        if comet_params_path and Path(comet_params_path).is_file():
            import shutil
            shutil.copy(comet_params_path, param_file)
        else:
            param_file.write_text(
                "\n".join(
                    [
                        f"database_name = {database_path}",
                        "peptide_mass_tolerance = 10.0",
                        "peptide_mass_units = 2",
                        "num_enzyme_termini = 2",
                        "missed_cleavages = 1",
                        "minimum_length = 7",
                    ]
                ),
                encoding="utf-8",
            )
        cmd = [comet_path, f"-P{param_file}", str(input_mzml)]
        self.execute(cmd, self.name, "comet", dry_run=dry_run)
        return str(input_mzml.with_suffix(".pep.xml"))
