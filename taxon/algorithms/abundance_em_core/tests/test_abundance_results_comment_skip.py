"""Regression tests: readers of abundance_results.tsv must skip the leading "#" line.

The unified writer in :class:`AbundanceEMPlugin._write_unified_results`
emits a single ``#``-prefixed provenance line before the header row.
Every downstream reader needs to ignore that line; without ``comment='#'``
(or an equivalent manual filter for stdlib ``csv``) the first
``"# cell_abundance == c_t_all ..."`` line is mistaken for the header
and column lookups like ``psm_abundance`` fail.

Run from the repository root::

    python -m pytest taxon/algorithms/abundance_em_core/tests/test_abundance_results_comment_skip.py -v
"""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd
import pytest


# Single source of truth for the synthetic file body.  Mirrors the exact
# layout produced by AbundanceEMPlugin._write_unified_results (one
# "#"-prefixed comment line, tab-separated header, one data row).
COMMENT_LINE = (
    "# cell_abundance == c_t_all (alias preserved for back-compat); "
    "when marker_compute_subsets=false the ribo/nayfach columns "
    "are zero/false.\n"
)
HEADER_LINE = (
    "taxon_id\ttaxon_name\tpsm_abundance\tbiomass_abundance\t"
    "cell_abundance\tproteome_size\tmarker_families\t"
    "marker_psms\thas_marker_estimate\t"
    "c_t_all\tc_t_ribo\tc_t_nayfach\t"
    "n_families_all\tn_families_ribo\tn_families_nayfach\t"
    "s_t_all\ts_t_ribo\ts_t_nayfach\t"
    "has_marker_all\thas_marker_ribo\thas_marker_nayfach\n"
)
DATA_LINE = (
    "42\tEscherichia coli\t0.600000\t0.500000\t"
    "0.550000\t4000\t10\t"
    "120.0000\t1\t"
    "0.550000\t0.600000\t0.000000\t"
    "10\t5\t3\t"
    "120.0000\t60.0000\t0.0000\t"
    "1\t1\t0\n"
)


@pytest.fixture
def synthetic_results(tmp_path: Path) -> Path:
    p = tmp_path / "abundance_results.tsv"
    p.write_text(COMMENT_LINE + HEADER_LINE + DATA_LINE, encoding="utf-8")
    return p


# --------------------------------------------------------------- pandas reader


def test_pandas_read_csv_with_comment_skips_provenance(synthetic_results: Path):
    """The exact call signature used in visualize_results.plot_abundance_results."""
    df = pd.read_csv(synthetic_results, sep="\t", comment="#")
    expected = {
        "taxon_id", "taxon_name", "psm_abundance", "biomass_abundance",
        "cell_abundance", "proteome_size", "marker_families",
        "marker_psms", "has_marker_estimate",
        "c_t_all", "c_t_ribo", "c_t_nayfach",
        "n_families_all", "n_families_ribo", "n_families_nayfach",
        "s_t_all", "s_t_ribo", "s_t_nayfach",
        "has_marker_all", "has_marker_ribo", "has_marker_nayfach",
    }
    assert expected.issubset(set(df.columns))
    assert len(df) == 1
    assert df.loc[0, "psm_abundance"] == pytest.approx(0.6)
    assert df.loc[0, "c_t_all"] == pytest.approx(0.55)
    assert df.loc[0, "c_t_ribo"] == pytest.approx(0.6)
    assert df.loc[0, "c_t_nayfach"] == pytest.approx(0.0)


def test_pandas_read_csv_without_comment_breaks(synthetic_results: Path):
    """Sanity: without comment='#', psm_abundance disappears (regression repro)."""
    df = pd.read_csv(synthetic_results, sep="\t")
    assert "psm_abundance" not in df.columns


# ------------------------------------------------------- end-to-end plotter


def test_plot_abundance_results_handles_comment_line(synthetic_results: Path, tmp_path: Path):
    """plot_abundance_results must complete without warning about missing columns."""
    from taxon.algorithms.abundance_em_core.visualize_results import plot_abundance_results

    out_png = tmp_path / "plot.png"
    plot_abundance_results(
        unified_result_path=synthetic_results,
        output_path=out_png,
        top_n=5,
    )
    # Successful plot writes the PNG; the regression manifested as an
    # early-return warning and no file.
    assert out_png.is_file()
    assert out_png.stat().st_size > 0


# ------------------------------------------------------ csv DictReader path


def test_csv_dictreader_with_manual_filter(synthetic_results: Path):
    """alpha_ablation.read_abundance_results uses csv.DictReader; verify the manual filter."""
    with synthetic_results.open("r", encoding="utf-8", newline="") as fh:
        rows = (line for line in fh if not line.lstrip().startswith("#"))
        reader = csv.DictReader(rows, delimiter="\t")
        fieldnames = reader.fieldnames or []
        required = {"taxon_id", "taxon_name", "psm_abundance", "proteome_size"}
        assert required.issubset(set(fieldnames))
        rows_out = list(reader)
        assert len(rows_out) == 1
        assert rows_out[0]["psm_abundance"] == "0.600000"
        assert rows_out[0]["proteome_size"] == "4000"


def test_alpha_ablation_read_handles_comment_line(synthetic_results: Path):
    """The production reader in alpha_ablation must not choke on the "#" line."""
    from taxon.algorithms.abundance_em_core.scripts.alpha_ablation import (
        read_abundance_results,
    )

    labels, pi, W = read_abundance_results(synthetic_results)
    assert labels == ["42|Escherichia coli"]
    assert pi.tolist() == pytest.approx([1.0])  # renormalised from 0.6 -> 1.0
    assert W.tolist() == [4000.0]
