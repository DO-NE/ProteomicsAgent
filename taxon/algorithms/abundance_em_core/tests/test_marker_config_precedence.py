"""CLI > YAML > default precedence tests for the marker-subset config keys.

These tests exercise the ``_flatten_yaml_config`` / ``load_config`` chain
in :mod:`main` without spinning up the full pipeline.  No HMMER or FASTA
data is required.

Run from the repository root with::

    python -m pytest taxon/algorithms/abundance_em_core/tests/test_marker_config_precedence.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Make sure the repo root is on sys.path so ``import main`` works
# regardless of where pytest is invoked from.
REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main as main_module  # noqa: E402


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "run_config.yaml"
    p.write_text(body, encoding="utf-8")
    return p


# --------------------------------------------------------------------- tests


def test_yaml_value_used_when_no_cli(tmp_path: Path):
    """YAML-supplied marker-subset keys should land in the merged config."""
    yaml_path = _write_yaml(
        tmp_path,
        """
input: /tmp/in.mzML
db: /tmp/db.fasta
corrections:
  marker:
    enabled: true
    hmm_profile_dir: /tmp/hmms
    compute_subsets: true
    nayfach30_tsv: /tmp/n30.tsv
    evalue: 1.0e-15
    min_family_signal: 0.7
    min_marker_families: 5
    min_marker_psms: 2.5
""",
    )
    config = main_module.load_config(yaml_path, cli_overrides={})
    assert config["marker_compute_subsets"] is True
    assert config["marker_nayfach30_tsv"] == "/tmp/n30.tsv"
    assert config["marker_evalue"] == pytest.approx(1e-15)
    assert config["marker_min_family_signal"] == pytest.approx(0.7)
    assert config["min_marker_families"] == 5
    assert config["min_marker_psms"] == pytest.approx(2.5)


def test_cli_overrides_yaml(tmp_path: Path):
    """A CLI override should win against the YAML value."""
    yaml_path = _write_yaml(
        tmp_path,
        """
corrections:
  marker:
    enabled: true
    compute_subsets: true
    evalue: 1.0e-10
    min_family_signal: 0.5
""",
    )
    config = main_module.load_config(
        yaml_path,
        cli_overrides={
            "marker_compute_subsets": False,
            "marker_evalue": 1e-20,
            "marker_min_family_signal": 1.5,
        },
    )
    assert config["marker_compute_subsets"] is False
    assert config["marker_evalue"] == pytest.approx(1e-20)
    assert config["marker_min_family_signal"] == pytest.approx(1.5)


def test_default_used_when_both_silent(tmp_path: Path):
    """No YAML value, no CLI flag -> key is absent (plugin default kicks in)."""
    yaml_path = _write_yaml(
        tmp_path,
        """
input: /tmp/in.mzML
db: /tmp/db.fasta
""",
    )
    config = main_module.load_config(yaml_path, cli_overrides={})
    assert "marker_compute_subsets" not in config
    assert "marker_nayfach30_tsv" not in config
    assert "marker_evalue" not in config
    assert "marker_min_family_signal" not in config


def test_marker_skip_subsets_flag_disables_subsets(tmp_path: Path):
    """The CLI flag --marker-skip-subsets projects to compute_subsets=False."""
    # The flag is wired in run_cmd as a store_true that is then projected
    # to the flat key marker_compute_subsets=False when set.  Here we
    # simulate that projection through cli_overrides directly.
    yaml_path = _write_yaml(
        tmp_path,
        """
corrections:
  marker:
    compute_subsets: true
""",
    )
    config_no_flag = main_module.load_config(yaml_path, cli_overrides={"marker_compute_subsets": None})
    assert config_no_flag["marker_compute_subsets"] is True

    config_flag = main_module.load_config(yaml_path, cli_overrides={"marker_compute_subsets": False})
    assert config_flag["marker_compute_subsets"] is False


def test_nayfach30_path_round_trip(tmp_path: Path):
    """YAML -> flat -> YAML round-trip preserves nayfach30_tsv."""
    yaml_in = _write_yaml(
        tmp_path,
        """
corrections:
  marker:
    enabled: true
    hmm_profile_dir: /tmp/h
    compute_subsets: true
    nayfach30_tsv: /custom/n30.tsv
    evalue: 5.0e-12
""",
    )
    flat = main_module.load_config(yaml_in, cli_overrides={})
    out_path = tmp_path / "run_config_out.yaml"
    main_module._serialize_run_config(flat, out_path)

    import yaml as _yaml
    parsed = _yaml.safe_load(out_path.read_text(encoding="utf-8"))
    marker = parsed["corrections"]["marker"]
    assert marker["compute_subsets"] is True
    assert marker["nayfach30_tsv"] == "/custom/n30.tsv"
    assert float(marker["evalue"]) == pytest.approx(5e-12)


def test_env_var_map_includes_new_keys():
    """Every new YAML flat key has a corresponding env var entry."""
    assert "marker_compute_subsets" in main_module._ENV_VAR_MAP
    assert "marker_nayfach30_tsv" in main_module._ENV_VAR_MAP
    assert "marker_evalue" in main_module._ENV_VAR_MAP
    assert main_module._ENV_VAR_MAP["marker_compute_subsets"] == "TAXON_MARKER_COMPUTE_SUBSETS"
    assert main_module._ENV_VAR_MAP["marker_nayfach30_tsv"] == "TAXON_MARKER_NAYFACH30_TSV"
    assert main_module._ENV_VAR_MAP["marker_evalue"] == "TAXON_MARKER_EVALUE"
