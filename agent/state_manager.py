"""Run-state persistence utilities for resumable workflows."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class RunState:
    """Serializable run state stored in run_state.json."""

    run_id: str
    started_at: str
    autonomy_mode: str = "balanced"
    input_files: list[str] = field(default_factory=list)
    database_path: str = ""
    completed_stages: list[str] = field(default_factory=list)
    stage_outputs: dict[str, str] = field(default_factory=dict)
    current_stage: str | None = None
    taxon_algorithm: str = "unipept_api"


class StateManager:
    """Persist and update run state in output/{run_id}/run_state.json."""

    def __init__(self, run_dir: Path) -> None:
        """Initialize manager with a run directory."""

        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.state_path = self.run_dir / "run_state.json"
        self.state: RunState | None = None

    @staticmethod
    def load(run_dir: Path) -> RunState | None:
        """Load run state from the supplied run directory, if present."""

        state_path = run_dir / "run_state.json"
        if not state_path.exists():
            return None
        payload = json.loads(state_path.read_text(encoding="utf-8"))
        loaded_state = RunState(**payload)
        env_algo = os.getenv("TAXON_ALGORITHM")
        if env_algo:
            loaded_state.taxon_algorithm = env_algo
        return loaded_state

    def save(self, state: RunState) -> None:
        """Save provided run state to disk and set as current state."""

        self.state = state
        self.state_path.write_text(json.dumps(asdict(state), indent=2), encoding="utf-8")

    def mark_stage_complete(self, stage_name: str, output_path: str) -> None:
        """Mark a stage as complete and persist its output path."""

        if self.state is None:
            raise RuntimeError("No active state to update.")
        if stage_name not in self.state.completed_stages:
            self.state.completed_stages.append(stage_name)
        self.state.stage_outputs[stage_name] = output_path
        self.state.current_stage = None
        self.save(self.state)

    def is_stage_complete(self, stage_name: str) -> bool:
        """Return whether the specified stage is marked complete."""

        if self.state is None:
            return False
        return stage_name in self.state.completed_stages

    def get_stage_output(self, stage_name: str) -> str | None:
        """Get output path for a completed stage if present."""

        if self.state is None:
            return None
        return self.state.stage_outputs.get(stage_name)


def new_run_state(
    run_id: str,
    autonomy_mode: str,
    input_files: list[str],
    database_path: str,
    taxon_algorithm: str = "unipept_api",
) -> RunState:
    """Create a fresh RunState with current UTC timestamp."""

    effective_algo = os.getenv("TAXON_ALGORITHM") or taxon_algorithm
    return RunState(
        run_id=run_id,
        started_at=datetime.now(timezone.utc).isoformat(),
        autonomy_mode=autonomy_mode,
        input_files=input_files,
        database_path=database_path,
        taxon_algorithm=effective_algo,
    )
