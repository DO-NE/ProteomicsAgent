"""Format conversion stage for mzML generation."""

from __future__ import annotations

import shutil
from pathlib import Path

from .base import PipelineStage


class FormatConversion(PipelineStage):
    """Convert vendor RAW files to mzML.

    Thermo ``.raw`` is routed to ThermoRawFileParser because the Thermo
    DLLs that msconvert relies on are Windows-only; everything else still
    goes through msconvert.
    """

    name = "format_conversion"
    tools = ["msconvert", "ThermoRawFileParser"]

    def run(self, input_path: str, params: dict, dry_run: bool = False) -> str:
        """Convert input RAW/mzML into output run mzML directory."""

        source = Path(input_path)
        run_dir = Path(params["run_dir"])
        outdir = run_dir / "mzml"
        outdir.mkdir(parents=True, exist_ok=True)

        target = outdir / f"{source.stem}.mzML"
        suffix = source.suffix.lower()

        if suffix == ".mzml":
            if not dry_run:
                shutil.copy2(source, target)
            return str(target)

        if suffix == ".raw":
            # ThermoRawFileParser writes "<stem>.mzML" into outdir; -f 2
            # selects indexed mzML, matching msconvert's --mzML default.
            cmd = [
                "ThermoRawFileParser",
                "-i", str(source),
                "-o", str(outdir),
                "-f", "2",
            ]
            self.execute(cmd, self.name, "ThermoRawFileParser", dry_run=dry_run)
            return str(target)

        cmd = ["msconvert", str(source), "--mzML", "--outdir", str(outdir)]
        self.execute(cmd, self.name, "msconvert", dry_run=dry_run)
        return str(target)
