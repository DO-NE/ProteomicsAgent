"""Format conversion stage for mzML generation."""

from __future__ import annotations

import shutil
from pathlib import Path

from .base import PipelineStage


class FormatConversion(PipelineStage):
    """Convert RAW files to mzML using msconvert."""

    name = "format_conversion"
    tools = ["msconvert"]

    def run(self, input_path: str, params: dict, dry_run: bool = False) -> str:
        """Convert input RAW/mzML into output run mzML directory."""

        source = Path(input_path)
        run_dir = Path(params["run_dir"])
        outdir = run_dir / "mzml"
        outdir.mkdir(parents=True, exist_ok=True)

        target = outdir / f"{source.stem}.mzML"
        if source.suffix.lower() == ".mzml":
            if not dry_run:
                shutil.copy2(source, target)
            return str(target)

        cmd = ["msconvert", str(source), "--mzML", "--outdir", str(outdir)]
        self.execute(cmd, self.name, "msconvert", dry_run=dry_run)
        return str(target)
