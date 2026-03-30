"""Base classes for pipeline stages and execution error handling."""

from __future__ import annotations

import subprocess
from abc import ABC, abstractmethod


class PipelineError(Exception):
    """Raised when an external pipeline command fails."""

    def __init__(self, stage: str, tool: str, returncode: int, stderr: str):
        self.stage = stage
        self.tool = tool
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(f"[{stage}] {tool} failed (rc={returncode}):\n{stderr}")


class PipelineStage(ABC):
    """Abstract base class for all pipeline stages."""

    name: str
    tools: list[str]

    @abstractmethod
    def run(self, input_path: str, params: dict, dry_run: bool = False) -> str:
        """Run this stage and return output path."""

    def execute(
        self,
        cmd: list[str],
        stage: str,
        tool: str,
        dry_run: bool = False,
    ) -> subprocess.CompletedProcess:
        """Execute a subprocess command and raise PipelineError on failure."""

        if dry_run:
            print(f"[DRY RUN] {' '.join(cmd)}")
            return subprocess.CompletedProcess(cmd, 0, "", "")

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode != 0:
            raise PipelineError(stage, tool, result.returncode, result.stderr)
        return result
