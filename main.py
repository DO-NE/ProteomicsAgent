"""CLI entry point for the metaproteomics agent system."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any

import click
import requests
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agent.orchestrator import Orchestrator
from agent.state_manager import RunState, StateManager, new_run_state
from config import check_tools, load_settings
from taxon.registry import TaxonRegistry

console = Console()


# Mapping from flat run-config keys to the env vars consumed by the
# orchestrator + abundance_em plugin. Anything not in this table is
# routed through dedicated paths (input/db/output_dir/no_llm) below.
_ENV_VAR_MAP: dict[str, str] = {
    "detectability_mode": "TAXON_DETECTABILITY_MODE",
    "detectability_file": "TAXON_DETECTABILITY_FILE",
    "resolve_uniprot": "TAXON_RESOLVE_UNIPROT",
    "prefix_map_file": "TAXON_PREFIX_MAP_FILE",
    "taxon_level": "TAXON_LEVEL",
    "marker_correction": "TAXON_MARKER_CORRECTION",
    "hmm_profile_dir": "TAXON_HMM_PROFILE_DIR",
    "min_marker_families": "TAXON_MARKER_MIN_FAMILIES",
    "min_marker_psms": "TAXON_MARKER_MIN_PSMS",
    "proteome_mass_correction": "TAXON_PROTEOME_MASS_CORRECTION",
    "alpha": "TAXON_EM_ALPHA",
    "max_iter": "TAXON_EM_MAX_ITER",
    "tol": "TAXON_EM_TOL",
    "n_restarts": "TAXON_EM_N_RESTARTS",
    "init_strategy": "TAXON_EM_INIT",
    "abundance_threshold": "TAXON_EM_ABUNDANCE_THRESHOLD",
    "min_psm_threshold": "TAXON_MIN_PSM_THRESHOLD",
    "generate_plot": "TAXON_GENERATE_PLOT",
    "plot_top_n": "TAXON_PLOT_TOP_N",
    "unified_table": "TAXON_UNIFIED_TABLE",
}


def _flatten_yaml_config(raw: dict) -> dict[str, Any]:
    """Translate the nested YAML schema into the flat parameter dict used internally."""

    flat: dict[str, Any] = {}

    for key in (
        "input", "db", "output_dir", "no_llm", "min_psm_threshold",
        "detectability_mode", "detectability_file",
    ):
        if key in raw and raw[key] is not None:
            flat[key] = raw[key]

    # `prefix_map` in YAML maps to `prefix_map_file` internally
    # (the env var the orchestrator reads is TAXON_PREFIX_MAP_FILE).
    if raw.get("prefix_map") is not None:
        flat["prefix_map_file"] = raw["prefix_map"]

    if "taxon_level" in raw and raw["taxon_level"] is not None:
        flat["taxon_level"] = raw["taxon_level"]
    if "resolve_uniprot" in raw and raw["resolve_uniprot"] is not None:
        flat["resolve_uniprot"] = raw["resolve_uniprot"]

    corrections = raw.get("corrections") or {}
    pm = corrections.get("proteome_mass") or {}
    if "enabled" in pm and pm["enabled"] is not None:
        flat["proteome_mass_correction"] = bool(pm["enabled"])
    marker = corrections.get("marker") or {}
    if "enabled" in marker and marker["enabled"] is not None:
        flat["marker_correction"] = bool(marker["enabled"])
    if marker.get("hmm_profile_dir") is not None:
        flat["hmm_profile_dir"] = marker["hmm_profile_dir"]
    if marker.get("min_marker_families") is not None:
        flat["min_marker_families"] = int(marker["min_marker_families"])
    if marker.get("min_marker_psms") is not None:
        flat["min_marker_psms"] = float(marker["min_marker_psms"])

    em = raw.get("em") or {}
    em_to_flat = {
        "alpha": ("alpha", float),
        "max_iter": ("max_iter", int),
        "tol": ("tol", float),
        "n_restarts": ("n_restarts", int),
        "init_strategy": ("init_strategy", str),
        "abundance_threshold": ("abundance_threshold", float),
    }
    for src, (dst, cast) in em_to_flat.items():
        if em.get(src) is not None:
            flat[dst] = cast(em[src])

    output = raw.get("output") or {}
    if output.get("plot") is not None:
        flat["generate_plot"] = bool(output["plot"])
    if output.get("plot_top_n") is not None:
        flat["plot_top_n"] = int(output["plot_top_n"])
    if output.get("unified_table") is not None:
        flat["unified_table"] = bool(output["unified_table"])

    return flat


def load_config(config_path: Path | None, cli_overrides: dict[str, Any]) -> dict[str, Any]:
    """Load YAML config and merge with CLI overrides.

    CLI overrides take precedence over config-file values. Returns a flat
    dict whose keys match the internal parameter names consumed by the
    orchestrator + abundance_em plugin.
    """

    config: dict[str, Any] = {}
    if config_path is not None:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise click.UsageError(
                "PyYAML is required to load --config files. "
                "Install with: pip install pyyaml"
            ) from exc
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise click.UsageError(
                f"Config file {config_path} must be a mapping (dict) at the top level"
            )
        config = _flatten_yaml_config(raw)

    # CLI overrides — only apply non-None values so unspecified flags do
    # not clobber the YAML defaults.
    for key, value in cli_overrides.items():
        if value is None:
            continue
        config[key] = value

    return config


def _serialize_run_config(config: dict[str, Any], path: Path) -> None:
    """Persist the merged effective config to ``path`` for reproducibility.

    Falls back to a plain key=value text file if PyYAML is missing.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml  # type: ignore
    except ImportError:
        with path.open("w", encoding="utf-8") as fh:
            for k in sorted(config):
                fh.write(f"{k}: {config[k]}\n")
        return

    # Reconstruct the nested schema so the saved file is editable / reusable.
    nested: dict[str, Any] = {}
    flat = dict(config)

    for key in (
        "input", "db", "output_dir", "no_llm", "min_psm_threshold",
        "detectability_mode", "detectability_file",
        "taxon_level", "resolve_uniprot",
    ):
        if key in flat:
            nested[key] = flat.pop(key)
    if "prefix_map_file" in flat:
        nested["prefix_map"] = flat.pop("prefix_map_file")

    corr: dict[str, Any] = {}
    pm: dict[str, Any] = {}
    if "proteome_mass_correction" in flat:
        pm["enabled"] = flat.pop("proteome_mass_correction")
    if pm:
        corr["proteome_mass"] = pm
    marker: dict[str, Any] = {}
    if "marker_correction" in flat:
        marker["enabled"] = flat.pop("marker_correction")
    for k_in, k_out in (
        ("hmm_profile_dir", "hmm_profile_dir"),
        ("min_marker_families", "min_marker_families"),
        ("min_marker_psms", "min_marker_psms"),
    ):
        if k_in in flat:
            marker[k_out] = flat.pop(k_in)
    if marker:
        corr["marker"] = marker
    if corr:
        nested["corrections"] = corr

    em: dict[str, Any] = {}
    for k in (
        "alpha", "max_iter", "tol", "n_restarts",
        "init_strategy", "abundance_threshold",
    ):
        if k in flat:
            em[k] = flat.pop(k)
    if em:
        nested["em"] = em

    output: dict[str, Any] = {}
    if "generate_plot" in flat:
        output["plot"] = flat.pop("generate_plot")
    if "plot_top_n" in flat:
        output["plot_top_n"] = flat.pop("plot_top_n")
    if "unified_table" in flat:
        output["unified_table"] = flat.pop("unified_table")
    if output:
        nested["output"] = output

    # Anything left in `flat` is unrecognised — preserve verbatim under a
    # dedicated key so it is visible without clobbering the schema.
    if flat:
        nested["_extra"] = flat

    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(nested, fh, sort_keys=False, default_flow_style=False)


def _apply_config_to_env(config: dict[str, Any]) -> None:
    """Project the merged config into the env vars the orchestrator reads."""

    if config.get("no_llm"):
        os.environ["NO_LLM_MODE"] = "true"

    for key, env_name in _ENV_VAR_MAP.items():
        if key not in config or config[key] is None:
            continue
        value = config[key]
        if isinstance(value, bool):
            os.environ[env_name] = "true" if value else "false"
        else:
            os.environ[env_name] = str(value)


def _validate_required(config: dict[str, Any]) -> None:
    """Raise UsageError if required input/db are missing or do not exist."""

    for required in ("input", "db"):
        val = config.get(required)
        if not val:
            raise click.UsageError(
                f"Missing required parameter: --{required} (or `{required}:` in YAML)"
            )
        if not Path(str(val)).exists():
            raise click.UsageError(f"{required} path does not exist: {val}")


def _offline_env() -> dict:
    """Return os.environ merged with offline/no-telemetry flags."""
    return {
        **os.environ,
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "LLAMA_CPP_DISABLE_TELEMETRY": "1",
    }


def _llm_server_reachable(url: str) -> bool:
    """Check if local OpenAI-compatible llama server is reachable."""

    try:
        requests.get(url + "/models", timeout=3)
        return True
    except Exception:
        return False


def _find_latest_run(output_dir: Path) -> tuple[Path, RunState] | None:
    """Return latest run directory and state from output tree."""

    states: list[tuple[Path, RunState]] = []
    for state_path in output_dir.glob("*/run_state.json"):
        state = StateManager.load(state_path.parent)
        if state:
            states.append((state_path.parent, state))
    if not states:
        return None
    states.sort(key=lambda item: item[1].started_at, reverse=True)
    return states[0]


def _startup_checks() -> bool:
    """Run tool checks and verify LLM server availability."""

    settings = load_settings()
    statuses = check_tools(settings)
    if not all(statuses.values()):
        console.print("[yellow]Warning: one or more tools are missing; pipeline stages may fail.[/yellow]")

    if settings.no_llm_mode:
        console.print("[yellow]Running in no-LLM mode. LLM server will not be used.[/yellow]")
        return True

    if not _llm_server_reachable(settings.llama_server_url):
        console.print(
            Panel(
                "LLM server not running. Start it first with: python main.py start-server",
                title="LLM Server Error",
                style="red",
            )
        )
        return False
    return True


@click.group()
def cli() -> None:
    """Metaproteomics agent command group."""


@cli.command("run")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to YAML config file. CLI flags override config values.",
)
@click.option("--input", "input_path", required=False, type=click.Path(path_type=Path), default=None)
@click.option("--db", "database_path", required=False, type=click.Path(path_type=Path), default=None)
@click.option("--output-dir", "output_dir", type=click.Path(path_type=Path), default=None,
              help="Override the run output directory (env: OUTPUT_DIR).")
@click.option("--autonomy", type=click.Choice(["full", "balanced", "supervised"]), default=None)
@click.option("--no-llm", "no_llm", is_flag=True, default=None, help="Run without LLM server.")
@click.option(
    "--detectability-mode",
    type=click.Choice(["uniform", "sequence_features", "file"]),
    default=None,
    help="Peptide detectability weighting mode for the abundance_em taxon algorithm.",
)
@click.option(
    "--detectability-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="TSV with pre-computed detectability scores (required when --detectability-mode=file).",
)
@click.option(
    "--resolve-uniprot/--no-resolve-uniprot",
    default=None,
    help="Resolve bare UniProt accession FASTA headers via the UniProt REST API (default: on).",
)
@click.option(
    "--prefix-map",
    "prefix_map",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="TSV mapping accession prefixes to organism names (two columns, no header).",
)
@click.option(
    "--taxon-level",
    type=click.Choice(["species", "strain"]),
    default=None,
    help="Organism-name normalisation depth: species collapses strains to binomials, strain preserves them.",
)
@click.option(
    "--marker-correction/--no-marker-correction",
    default=None,
    help=(
        "Run post-EM marker-based cell-equivalent abundance correction. "
        "Requires HMMER and a directory of GTDB bac120/ar53 HMM profiles "
        "(see --hmm-profile-dir)."
    ),
)
@click.option(
    "--hmm-profile-dir",
    "hmm_profile_dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help=(
        "Directory containing GTDB bac120/ar53 HMM profile bundles "
        "(produced by scripts/download_marker_hmms.py). Required when "
        "--marker-correction is enabled."
    ),
)
@click.option(
    "--proteome-mass-correction/--no-proteome-mass-correction",
    default=None,
    help="Enable proteome-size-weighted protein-biomass abundance correction",
)
@click.option(
    "--min-psm-threshold",
    type=int,
    default=None,
    help="Minimum PSMs for a taxon to be reported.",
)
def run_cmd(
    config_path: Path | None,
    input_path: Path | None,
    database_path: Path | None,
    output_dir: Path | None,
    autonomy: str | None,
    no_llm: bool | None,
    detectability_mode: str | None,
    detectability_file: Path | None,
    resolve_uniprot: bool | None,
    prefix_map: Path | None,
    taxon_level: str | None,
    marker_correction: bool | None,
    hmm_profile_dir: Path | None,
    proteome_mass_correction: bool | None,
    min_psm_threshold: int | None,
) -> None:
    """Start a new run or resume latest if user confirms."""

    cli_overrides = {
        "input": str(input_path) if input_path else None,
        "db": str(database_path) if database_path else None,
        "output_dir": str(output_dir) if output_dir else None,
        "no_llm": no_llm,
        "detectability_mode": detectability_mode,
        "detectability_file": str(detectability_file) if detectability_file else None,
        "resolve_uniprot": resolve_uniprot,
        "prefix_map_file": str(prefix_map) if prefix_map else None,
        "taxon_level": taxon_level,
        "marker_correction": marker_correction,
        "hmm_profile_dir": str(hmm_profile_dir) if hmm_profile_dir else None,
        "proteome_mass_correction": proteome_mass_correction,
        "min_psm_threshold": min_psm_threshold,
    }

    config = load_config(config_path, cli_overrides)
    _validate_required(config)
    if config.get("detectability_mode") == "file" and not config.get("detectability_file"):
        console.print(Panel("--detectability-file is required when detectability_mode=file.", style="red"))
        raise SystemExit(1)
    if config.get("marker_correction") and not config.get("hmm_profile_dir"):
        console.print(Panel(
            "hmm_profile_dir is required when marker_correction is enabled.",
            style="red",
        ))
        raise SystemExit(1)

    if config.get("output_dir"):
        os.environ["OUTPUT_DIR"] = str(config["output_dir"])

    _apply_config_to_env(config)

    settings = load_settings()
    if not _startup_checks():
        raise SystemExit(1)

    base_output = Path(settings.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)
    latest = _find_latest_run(base_output)

    if latest is not None:
        latest_dir, latest_state = latest
        answer = input(
            f"Found existing run from {latest_state.started_at} (completed: {latest_state.completed_stages}). Resume? (y/n) "
        ).strip().lower()
        if answer == "y":
            orchestrator = Orchestrator(settings=settings, run_state=latest_state, run_dir=latest_dir)
            orchestrator.run()
            return

    run_id = str(uuid.uuid4())
    run_state = new_run_state(
        run_id=run_id,
        autonomy_mode=autonomy or settings.default_autonomy_mode,
        input_files=[str(config["input"])],
        database_path=str(config["db"]),
    )
    run_dir = base_output / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    _serialize_run_config(config, run_dir / "run_config.yaml")
    orchestrator = Orchestrator(settings=settings, run_state=run_state, run_dir=run_dir)
    orchestrator.run()


@cli.command("resume")
def resume_cmd() -> None:
    """Resume the most recent run from output checkpoint."""

    settings = load_settings()
    if not _startup_checks():
        raise SystemExit(1)

    output_dir = Path(settings.output_dir)
    latest = _find_latest_run(output_dir)
    if latest is None:
        console.print(Panel("No previous runs found in output directory.", style="red"))
        raise SystemExit(1)

    run_dir, state = latest
    orchestrator = Orchestrator(settings=settings, run_state=state, run_dir=run_dir)
    orchestrator.run()


@cli.command("list-algorithms")
def list_algorithms_cmd() -> None:
    """List available taxon inference plugins."""

    registry = TaxonRegistry()
    table = Table(title="Available Taxon Algorithms")
    table.add_column("Name", style="bold")
    table.add_column("Description")
    table.add_column("Internet")
    for row in registry.list_plugins():
        table.add_row(row["name"], row["description"], "yes" if row["requires_internet"] else "no")
    console.print(table)


@cli.command("check-tools")
def check_tools_cmd() -> None:
    """Print configured bioinformatics tool status."""

    check_tools(load_settings())


@cli.command("start-server")
def start_server_cmd() -> None:
    """Start llama-cpp-python OpenAI-compatible server."""

    settings = load_settings()
    if not settings.model_path:
        console.print(Panel("MODEL_PATH is not configured in .env.", style="red"))
        raise SystemExit(1)

    cmd = [
        sys.executable,
        "-m",
        "llama_cpp.server",
        "--model",
        settings.model_path,
        "--n_gpu_layers",
        "-1",
        "--host",
        "127.0.0.1",
        "--port",
        "8000",
    ]
    console.print(Panel(f"Starting llama server:\n{' '.join(cmd)}", style="green"))
    subprocess.run(cmd, check=False, env=_offline_env())


@cli.command("run-pipeline")
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to YAML config file. CLI flags override config values.",
)
@click.option("--input", "input_path", required=False, type=click.Path(path_type=Path), default=None)
@click.option("--db", "database_path", required=False, type=click.Path(path_type=Path), default=None)
@click.option("--output-dir", "output_dir", type=click.Path(path_type=Path), default=None)
@click.option(
    "--detectability-mode",
    type=click.Choice(["uniform", "sequence_features", "file"]),
    default=None,
    help="Peptide detectability weighting mode for the abundance_em taxon algorithm.",
)
@click.option(
    "--detectability-file",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="TSV with pre-computed detectability scores (required when --detectability-mode=file).",
)
@click.option(
    "--resolve-uniprot/--no-resolve-uniprot",
    default=None,
    help="Resolve bare UniProt accession FASTA headers via the UniProt REST API (default: on).",
)
@click.option(
    "--prefix-map",
    "prefix_map",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="TSV mapping accession prefixes to organism names (two columns, no header).",
)
@click.option(
    "--taxon-level",
    type=click.Choice(["species", "strain"]),
    default=None,
    help="Organism-name normalisation depth: species collapses strains to binomials, strain preserves them.",
)
@click.option(
    "--marker-correction/--no-marker-correction",
    default=None,
    help=(
        "Run post-EM marker-based cell-equivalent abundance correction. "
        "Requires HMMER and --hmm-profile-dir."
    ),
)
@click.option(
    "--hmm-profile-dir",
    "hmm_profile_dir",
    type=click.Path(exists=True, path_type=Path),
    default=None,
    help="Directory with GTDB bac120/ar53 HMM bundles (used with --marker-correction).",
)
@click.option(
    "--proteome-mass-correction/--no-proteome-mass-correction",
    default=None,
    help="Enable proteome-size-weighted protein-biomass abundance correction",
)
@click.option(
    "--min-psm-threshold",
    type=int,
    default=None,
    help="Minimum PSMs for a taxon to be reported.",
)
def run_pipeline_cmd(
    config_path: Path | None,
    input_path: Path | None,
    database_path: Path | None,
    output_dir: Path | None,
    detectability_mode: str | None,
    detectability_file: Path | None,
    resolve_uniprot: bool | None,
    prefix_map: Path | None,
    taxon_level: str | None,
    marker_correction: bool | None,
    hmm_profile_dir: Path | None,
    proteome_mass_correction: bool | None,
    min_psm_threshold: int | None,
) -> None:
    """Run the full pipeline non-interactively in no-LLM mode."""

    cli_overrides = {
        "input": str(input_path) if input_path else None,
        "db": str(database_path) if database_path else None,
        "output_dir": str(output_dir) if output_dir else None,
        "no_llm": True,
        "detectability_mode": detectability_mode,
        "detectability_file": str(detectability_file) if detectability_file else None,
        "resolve_uniprot": resolve_uniprot,
        "prefix_map_file": str(prefix_map) if prefix_map else None,
        "taxon_level": taxon_level,
        "marker_correction": marker_correction,
        "hmm_profile_dir": str(hmm_profile_dir) if hmm_profile_dir else None,
        "proteome_mass_correction": proteome_mass_correction,
        "min_psm_threshold": min_psm_threshold,
    }

    config = load_config(config_path, cli_overrides)
    _validate_required(config)
    if config.get("detectability_mode") == "file" and not config.get("detectability_file"):
        console.print(Panel("--detectability-file is required when detectability_mode=file.", style="red"))
        raise SystemExit(1)
    if config.get("marker_correction") and not config.get("hmm_profile_dir"):
        console.print(Panel(
            "hmm_profile_dir is required when marker_correction is enabled.",
            style="red",
        ))
        raise SystemExit(1)

    if config.get("output_dir"):
        os.environ["OUTPUT_DIR"] = str(config["output_dir"])

    _apply_config_to_env(config)

    settings = load_settings()
    if not _startup_checks():
        raise SystemExit(1)

    base_output = Path(settings.output_dir)
    base_output.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    run_state = new_run_state(
        run_id=run_id,
        autonomy_mode="full",
        input_files=[str(config["input"])],
        database_path=str(config["db"]),
    )
    run_dir = base_output / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "diagnostics").mkdir(parents=True, exist_ok=True)
    _serialize_run_config(config, run_dir / "run_config.yaml")
    orchestrator = Orchestrator(settings=settings, run_state=run_state, run_dir=run_dir)
    orchestrator.run()


if __name__ == "__main__":
    cli()
