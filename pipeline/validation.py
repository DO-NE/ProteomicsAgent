"""Peptide validation stage using PeptideProphet or Percolator."""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from config import resolve_tpp_binary
from .base import PipelineStage, PipelineError
import shutil

logger = logging.getLogger(__name__)

_FALLBACK_PROBABILITY = 0.90


def _local_tag(tag: str) -> str:
    """Strip XML namespace prefix from a tag name."""
    if "}" in tag:
        return tag.rsplit("}", 1)[1]
    return tag


def extract_fdr_threshold(pepxml_path: str, target_fdr: float = 0.01) -> float:
    """Return the PeptideProphet probability threshold for a target FDR.

    Parses ``<error_point>`` elements from the PeptideProphet ROC curve
    embedded in the pepXML file and returns the ``min_prob`` value
    corresponding to the largest error rate that does not exceed *target_fdr*.
    Uses streaming (iterparse) so it is safe for large pepXML files.

    Falls back to 0.90 with a warning when no ``<error_point>`` elements
    exist (e.g. older TPP versions or non-PeptideProphet output).
    """
    path = Path(pepxml_path)
    if not path.exists():
        logger.warning(
            "pepXML not found: %s — using fallback probability threshold %.2f",
            pepxml_path,
            _FALLBACK_PROBABILITY,
        )
        return _FALLBACK_PROBABILITY

    error_points: list[tuple[float, float]] = []
    for _, elem in ET.iterparse(str(path), events=["end"]):
        if _local_tag(elem.tag) == "error_point":
            try:
                error = float(elem.get("error", ""))
                min_prob = float(elem.get("min_prob", ""))
                error_points.append((error, min_prob))
            except (ValueError, TypeError):
                pass
        elem.clear()

    if not error_points:
        logger.warning(
            "No <error_point> elements found in %s — using fallback probability "
            "threshold %.2f",
            pepxml_path,
            _FALLBACK_PROBABILITY,
        )
        return _FALLBACK_PROBABILITY

    candidates = [(e, p) for e, p in error_points if e <= target_fdr]
    if not candidates:
        # target_fdr is tighter than any available error point; use the most
        # stringent (highest probability) threshold available.
        _, best_prob = max(error_points, key=lambda x: x[1])
        logger.warning(
            "Target FDR %.4f is below all available <error_point> entries; "
            "using most stringent probability threshold %.4f",
            target_fdr,
            best_prob,
        )
        return best_prob

    # Among candidates take the one closest to (but not exceeding) target_fdr.
    _, best_prob = max(candidates, key=lambda x: x[0])
    logger.info(
        "Extracted probability threshold %.4f for %.2f%% FDR from "
        "PeptideProphet ROC curve",
        best_prob,
        target_fdr * 100,
    )
    return best_prob


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

            # PeptideProphet (TPP v7.3.0) modifies the input pepXML file in place.
            dest = outdir / pepxml.name
            shutil.copy2(str(pepxml), str(dest))
            return str(dest)

        percolator_path = params.get("percolator_path", "")
        output_file = outdir / "percolator_psms.txt"
        cmd = [percolator_path, "--results-psms", str(output_file), str(pepxml)]
        self.execute(cmd, self.name, "percolator", dry_run=dry_run)
        return str(output_file)
