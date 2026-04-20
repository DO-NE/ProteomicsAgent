"""CLI entry point for the metaproteomics agent system."""

from __future__ import annotations

import os
import subprocess
import sys
import uuid
from pathlib import Path

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
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--db", "database_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--autonomy", type=click.Choice(["full", "balanced", "supervised"]), default=None)
@click.option("--no-llm", "no_llm", is_flag=True, default=False, help="Run without LLM server.")
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
    "--biomass-mode",
    type=click.Choice(["correct", "none"]),
    default="correct",
    help=(
        "Biomass correction mode. 'correct' applies per-taxon correction "
        "factors to convert PSM-level abundance to biomass-level abundance. "
        "'none' disables correction."
    ),
)
@click.option(
    "--min-psm-threshold",
    type=int,
    default=2,
    help=(
        "Minimum PSM count per protein for proteome coverage calculation "
        "in biomass correction. Default: 2."
    ),
)
def run_cmd(
    input_path: Path,
    database_path: Path,
    autonomy: str | None,
    no_llm: bool,
    detectability_mode: str | None,
    detectability_file: Path | None,
    resolve_uniprot: bool | None,
    prefix_map: Path | None,
    taxon_level: str | None,
    biomass_mode: str,
    min_psm_threshold: int,
) -> None:
    """Start a new run or resume latest if user confirms."""

    if no_llm:
        os.environ["NO_LLM_MODE"] = "true"
    if detectability_mode:
        os.environ["TAXON_DETECTABILITY_MODE"] = detectability_mode
    if detectability_file:
        os.environ["TAXON_DETECTABILITY_FILE"] = str(detectability_file)
    if detectability_mode == "file" and not detectability_file:
        console.print(Panel("--detectability-file is required when --detectability-mode=file.", style="red"))
        raise SystemExit(1)
    if resolve_uniprot is not None:
        os.environ["TAXON_RESOLVE_UNIPROT"] = "true" if resolve_uniprot else "false"
    if prefix_map:
        os.environ["TAXON_PREFIX_MAP_FILE"] = str(prefix_map)
    if taxon_level:
        os.environ["TAXON_LEVEL"] = taxon_level
    os.environ["TAXON_BIOMASS_MODE"] = biomass_mode
    os.environ["TAXON_MIN_PSM_THRESHOLD"] = str(min_psm_threshold)
    settings = load_settings()
    if not _startup_checks():
        raise SystemExit(1)

    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    latest = _find_latest_run(output_dir)

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
        input_files=[str(input_path)],
        database_path=str(database_path),
    )
    run_dir = output_dir / run_id
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
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--db", "database_path", required=True, type=click.Path(exists=True, path_type=Path))
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
    "--biomass-mode",
    type=click.Choice(["correct", "none"]),
    default="correct",
    help=(
        "Biomass correction mode. 'correct' applies per-taxon correction "
        "factors to convert PSM-level abundance to biomass-level abundance. "
        "'none' disables correction."
    ),
)
@click.option(
    "--min-psm-threshold",
    type=int,
    default=2,
    help=(
        "Minimum PSM count per protein for proteome coverage calculation "
        "in biomass correction. Default: 2."
    ),
)
def run_pipeline_cmd(
    input_path: Path,
    database_path: Path,
    detectability_mode: str | None,
    detectability_file: Path | None,
    resolve_uniprot: bool | None,
    prefix_map: Path | None,
    taxon_level: str | None,
    biomass_mode: str,
    min_psm_threshold: int,
) -> None:
    """Run the full pipeline non-interactively in no-LLM mode."""

    os.environ["NO_LLM_MODE"] = "true"
    if detectability_mode:
        os.environ["TAXON_DETECTABILITY_MODE"] = detectability_mode
    if detectability_file:
        os.environ["TAXON_DETECTABILITY_FILE"] = str(detectability_file)
    if detectability_mode == "file" and not detectability_file:
        console.print(Panel("--detectability-file is required when --detectability-mode=file.", style="red"))
        raise SystemExit(1)
    if resolve_uniprot is not None:
        os.environ["TAXON_RESOLVE_UNIPROT"] = "true" if resolve_uniprot else "false"
    if prefix_map:
        os.environ["TAXON_PREFIX_MAP_FILE"] = str(prefix_map)
    if taxon_level:
        os.environ["TAXON_LEVEL"] = taxon_level
    os.environ["TAXON_BIOMASS_MODE"] = biomass_mode
    os.environ["TAXON_MIN_PSM_THRESHOLD"] = str(min_psm_threshold)
    settings = load_settings()
    if not _startup_checks():
        raise SystemExit(1)

    output_dir = Path(settings.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = str(uuid.uuid4())
    run_state = new_run_state(
        run_id=run_id,
        autonomy_mode="full",
        input_files=[str(input_path)],
        database_path=str(database_path),
    )
    run_dir = output_dir / run_id
    orchestrator = Orchestrator(settings=settings, run_state=run_state, run_dir=run_dir)
    orchestrator.run()


if __name__ == "__main__":
    cli()
