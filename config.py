"""Configuration loading and external tool availability checks."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table


@dataclass
class Settings:
    """Application settings loaded from environment variables."""

    llm_backend: str = "llama"
    llama_server_url: str = "http://localhost:8000/v1"
    openai_api_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model_id: str = ""
    anthropic_api_key: str = ""
    anthropic_base_url: str = "https://api.anthropic.com/v1"
    anthropic_model_id: str = "claude-sonnet-4-6"
    msfragger_path: str = ""
    comet_path: str = ""
    comet_params_path: str = ""
    tpp_bin_path: str = ""
    percolator_path: str = ""
    output_dir: str = "./output"
    default_autonomy_mode: str = "balanced"
    database_path: str = ""
    model_path: str = ""
    no_llm_mode: bool = False
    taxon_algorithm: str = os.getenv("TAXON_ALGORITHM", "unipept_api")


def load_settings() -> Settings:
    """Load settings from .env and return a Settings dataclass instance."""

    load_dotenv()
    return Settings(
        llm_backend=os.getenv("LLM_BACKEND", "llama").strip().lower(),
        llama_server_url=os.getenv("LLAMA_SERVER_URL", "http://localhost:8000/v1").strip(),
        openai_api_url=os.getenv("OPENAI_API_URL", "https://api.openai.com/v1").strip(),
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        openai_model_id=os.getenv("OPENAI_MODEL_ID", "gpt-5.4").strip(),
        anthropic_base_url=os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com/v1").strip(),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip(),
        anthropic_model_id=os.getenv("ANTHROPIC_MODEL_ID", "claude-sonnet-4-6").strip(),
        msfragger_path=os.getenv("MSFRAGGER_PATH", "").strip(),
        comet_path=os.getenv("COMET_PATH", "").strip(),
        comet_params_path=os.getenv("COMET_PARAMS_PATH", "").strip(),
        tpp_bin_path=os.getenv("TPP_BIN_PATH", "").strip(),
        percolator_path=os.getenv("PERCOLATOR_PATH", "").strip(),
        output_dir=os.getenv("OUTPUT_DIR", "./output").strip(),
        default_autonomy_mode=os.getenv("DEFAULT_AUTONOMY_MODE", "balanced").strip().lower(),
        database_path=os.getenv("DATABASE_PATH", "").strip(),
        model_path=os.getenv("MODEL_PATH", "").strip(),
        no_llm_mode=os.getenv("NO_LLM_MODE", "false").strip().lower() == "true",
        taxon_algorithm=os.getenv("TAXON_ALGORITHM", "unipept_api").strip(),
    )


def _is_executable(path: Path) -> bool:
    """Return True if path exists, is a file, and is executable."""

    return path.exists() and path.is_file() and os.access(path, os.X_OK)


def resolve_tpp_binary(tpp_bin_path: str, candidates: list[str]) -> str:
    """Return the first executable candidate found in tpp_bin_path, or empty string."""

    if not tpp_bin_path:
        return ""
    tpp_bin = Path(tpp_bin_path)
    for name in candidates:
        p = tpp_bin / name
        if _is_executable(p):
            return str(p)
    return ""


# Candidate binary names for TPP tools across versions (7.x renamed several binaries).
_TPP_CANDIDATES: dict[str, list[str]] = {
    "TPP/PeptideProphet": ["PeptideProphetParser", "PeptideProphet"],
    "TPP/ASAPRatio": ["ASAPRatioPeptideParser", "ASAPRatioProteinParser", "ASAPRatio"],
    "TPP/ProteinProphet": ["ProteinProphet"],
}


def check_tools(settings: Settings | None = None) -> dict[str, bool]:
    """Validate configured tool binaries and print a rich status table."""

    cfg = settings or load_settings()
    console = Console()

    tool_paths: dict[str, str] = {
        "MSFragger": cfg.msfragger_path,
        "Comet": cfg.comet_path,
        **{
            label: resolve_tpp_binary(cfg.tpp_bin_path, names)
            for label, names in _TPP_CANDIDATES.items()
        },
        "Percolator": cfg.percolator_path,
    }

    table = Table(title="Configured Bioinformatics Tools")
    table.add_column("Tool", style="bold")
    table.add_column("Path", overflow="fold")
    table.add_column("Status")

    status: dict[str, bool] = {}
    for tool, raw_path in tool_paths.items():
        path = Path(raw_path).expanduser() if raw_path else Path("")
        ok = _is_executable(path) if raw_path else False
        status[tool] = ok
        table.add_row(tool, str(path) if raw_path else "(not configured)", "✓ found" if ok else "✗ missing")

    console.print(table)
    return status
