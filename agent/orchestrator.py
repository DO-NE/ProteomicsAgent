"""Interactive orchestrator for metaproteomics agentic pipeline execution."""

from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from agent import llm_client
from agent.prompts import SYSTEM_PROMPT
from agent.state_manager import RunState, StateManager, new_run_state
from config import Settings
from pipeline.base import PipelineError
from pipeline.format_conversion import FormatConversion
from pipeline.peptide_id import PeptideIdentification
from pipeline.protein_assignment import ProteinAssignment
from pipeline.quantitation import Quantitation
from pipeline.validation import PeptideValidation
from taxon.base_plugin import TaxonResult
from taxon.registry import TaxonRegistry
from visualization import figures
from visualization.report import export_summary, export_tsv


class Orchestrator:
    """Conversational orchestrator that delegates actions to pipeline tools."""

    def __init__(
        self,
        settings: Settings,
        run_state: RunState | None = None,
        run_dir: Path | None = None,
        input_files: list[str] | None = None,
        database_path: str | None = None,
        autonomy_mode: str | None = None,
    ) -> None:
        """Initialize orchestrator state, tools, registry, and welcome banner."""

        self.console = Console()
        self.settings = settings
        base_output = Path(settings.output_dir)

        if run_state is None:
            run_id = str(uuid.uuid4())
            selected_autonomy = autonomy_mode or settings.default_autonomy_mode
            run_state = new_run_state(
                run_id=run_id,
                autonomy_mode=selected_autonomy,
                input_files=input_files or [],
                database_path=database_path or settings.database_path,
                taxon_algorithm=settings.taxon_algorithm,
            )
            run_dir = base_output / run_id

        assert run_dir is not None
        self.run_dir = run_dir
        self.state_manager = StateManager(self.run_dir)
        self.state_manager.save(run_state)
        self.state = run_state

        if not self.state.autonomy_mode:
            self.state.autonomy_mode = input("Select autonomy mode (full/balanced/supervised): ").strip().lower() or "balanced"
            self.state_manager.save(self.state)

        self.stages = {
            "format_conversion": FormatConversion(),
            "peptide_id": PeptideIdentification(),
            "validation": PeptideValidation(),
            "quantitation": Quantitation(),
            "protein_assignment": ProteinAssignment(),
        }
        self.taxon_registry = TaxonRegistry()
        self.history: list[dict[str, str]] = []
        self.latest_taxon_results: list[TaxonResult] = []
        self.latest_figures: list[str] = []

        algorithms = ", ".join([p["name"] for p in self.taxon_registry.list_plugins()]) or "none"
        self.console.print(
            Panel(
                (
                    f"Run ID: {self.state.run_id}\n"
                    f"Autonomy: {self.state.autonomy_mode}\n"
                    f"Input files: {', '.join(self.state.input_files) or '(none)'}\n"
                    f"LLM backend: {self.settings.llm_backend}\n"
                    f"Available taxon algorithms: {algorithms}"
                ),
                title="Metaproteomics Agent",
            )
        )

    def _build_system_prompt(self) -> str:
        """Build system prompt with current state and plugin availability."""

        return SYSTEM_PROMPT.format(
            autonomy_mode=self.state.autonomy_mode,
            run_state=json.dumps(self.state.__dict__, indent=2),
            available_algorithms=json.dumps(self.taxon_registry.list_plugins(), indent=2),
            completed_stages=", ".join(self.state.completed_stages),
        )

    def _parse_action(self, response: str) -> dict | None:
        """Extract action dictionary from LLM ACTION block."""

        match = re.search(r"<ACTION>(.*?)</ACTION>", response, flags=re.DOTALL)
        if not match:
            return None
        block = match.group(1)
        action: dict[str, object] = {}
        for line in block.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()
            if key == "params":
                action[key] = json.loads(value) if value else {}
            else:
                action[key] = value
        return action

    def _approve_action(self, tool: str, stage: str | None, params: dict) -> tuple[bool, dict]:
        """Apply autonomy mode gating and optionally allow param edits."""

        mode = self.state.autonomy_mode
        should_prompt = mode == "supervised" or (mode == "balanced" and tool == "run_pipeline_stage")
        if not should_prompt:
            return True, params

        self.console.print(f"[cyan]Proposed action:[/cyan] tool={tool}, stage={stage}, params={params}")
        answer = input("Proceed? (y/n/edit): ").strip().lower()
        if answer == "y":
            return True, params
        if answer == "edit":
            raw = input("Enter new params JSON: ").strip()
            try:
                return True, json.loads(raw)
            except json.JSONDecodeError:
                return False, params
        return False, params

    def run_pipeline_stage(self, stage: str, params: dict) -> dict:
        """Execute a pipeline stage and update persisted state."""

        if stage not in self.stages:
            return {"status": "error", "message": f"Unknown stage: {stage}"}
        if self.state_manager.is_stage_complete(stage):
            return {
                "status": "ok",
                "stage": stage,
                "output": self.state_manager.get_stage_output(stage),
                "message": "Stage already complete; skipped.",
            }

        stage_runner = self.stages[stage]
        input_path = params.get("input_path") or (self.state.input_files[0] if self.state.input_files else "")
        if stage != "format_conversion":
            previous = {
                "peptide_id": "format_conversion",
                "validation": "peptide_id",
                "quantitation": "validation",
                "protein_assignment": "validation",
            }.get(stage)
            if previous:
                input_path = self.state_manager.get_stage_output(previous) or input_path

        merged_params = {
            **params,
            "run_dir": str(self.run_dir),
            "database_path": self.state.database_path,
            "msfragger_path": self.settings.msfragger_path,
            "comet_path": self.settings.comet_path,
            "tpp_bin_path": self.settings.tpp_bin_path,
            "percolator_path": self.settings.percolator_path,
        }
        self.state.current_stage = stage
        self.state_manager.save(self.state)
        try:
            output_path = stage_runner.run(str(input_path), merged_params)
        except PipelineError as exc:
            self.state.current_stage = None
            self.state_manager.save(self.state)
            return {
                "status": "error",
                "type": "PipelineError",
                "stage": exc.stage,
                "tool": exc.tool,
                "returncode": exc.returncode,
                "stderr": exc.stderr,
            }

        self.state_manager.mark_stage_complete(stage, output_path)
        self.state = self.state_manager.state or self.state
        return {"status": "ok", "output": output_path, "stage": stage}

    def run_taxon_inference(self, algorithm: str, params: dict) -> dict:
        """Run taxonomic inference using peptides from validation pepXML."""

        # --- get peptide sequences from validation pepXML ---
        pepxml_path = self.state_manager.get_stage_output("validation")
        peptides: list[str] = []
        spectral_counts: dict[str, int] = {}

        if pepxml_path and Path(pepxml_path).exists():
            import xml.etree.ElementTree as ET
            tree = ET.parse(pepxml_path)
            root = tree.getroot()
            ns = root.tag.split("}")[0].strip("{") if root.tag.startswith("{") else ""
            def q(name):
                return f"{{{ns}}}{name}" if ns else name
            for hit in root.iter(q("search_hit")):
                pep = hit.attrib.get("peptide", "")
                if pep:
                    peptides.append(pep)
                    spectral_counts[pep] = spectral_counts.get(pep, 0) + 1

        if not peptides:
            return {
                "status": "error",
                "message": "No peptides found in validation output. "
                        "Ensure validation stage completed successfully."
            }

        # --- build config for plugin ---
        config = {
            "fasta_path": params.get("fasta_path", self.state.database_path),
            "database_path": params.get("database_path", self.state.database_path),
            "spectral_counts": spectral_counts,
        }

        detectability_mode = params.get("detectability_mode") or os.getenv("TAXON_DETECTABILITY_MODE")
        if detectability_mode:
            config["detectability_mode"] = detectability_mode
        detectability_file = params.get("detectability_file") or os.getenv("TAXON_DETECTABILITY_FILE")
        if detectability_file:
            config["detectability_file"] = detectability_file
        resolve_uniprot_env = os.getenv("TAXON_RESOLVE_UNIPROT")
        if "resolve_uniprot" in params:
            config["resolve_uniprot"] = bool(params["resolve_uniprot"])
        elif resolve_uniprot_env is not None:
            config["resolve_uniprot"] = resolve_uniprot_env.lower() not in ("0", "false", "no")
        prefix_map_file = params.get("prefix_map_file") or os.getenv("TAXON_PREFIX_MAP_FILE")
        if prefix_map_file:
            config["prefix_map_file"] = prefix_map_file
        taxon_level = params.get("taxon_level") or os.getenv("TAXON_LEVEL")
        if taxon_level:
            config["taxon_level"] = taxon_level
        marker_env = os.getenv("TAXON_MARKER_CORRECTION")
        if "marker_correction" in params:
            config["marker_correction"] = bool(params["marker_correction"])
        elif marker_env is not None:
            config["marker_correction"] = marker_env.lower() not in ("0", "false", "no")
        hmm_profile_dir = (
            params.get("hmm_profile_dir") or os.getenv("TAXON_HMM_PROFILE_DIR")
        )
        if hmm_profile_dir:
            config["hmm_profile_dir"] = hmm_profile_dir
        config["output_dir"] = str(self.run_dir)

        results = self.taxon_registry.run(algorithm, peptides, config)
        self.latest_taxon_results = results
        self.state.taxon_algorithm = algorithm
        self.state_manager.save(self.state)

        taxon_dir = self.run_dir / "taxon"
        tsv_path = export_tsv(results, taxon_dir, "results.tsv")

        # mark complete so resume works
        self.state_manager.mark_stage_complete("taxon_inference", str(tsv_path))
        self.state = self.state_manager.state or self.state

        return {
            "status": "ok",
            "taxon_count": len(results),
            "top_taxon": results[0].taxon_name if results else "none",
            "output": str(tsv_path),
        }

    def show_state(self) -> dict:
        """Return current run state dictionary."""

        return self.state.__dict__

    def generate_figures(self, types: list[str], data_path: str) -> dict:
        """Generate requested figures from taxon or pepXML data."""

        fig_dir = self.run_dir / "figures"
        saved: list[str] = []
        for fig_type in types:
            if fig_type == "taxon_bar_chart":
                saved.extend(figures.taxon_bar_chart(self.latest_taxon_results, fig_dir))
            elif fig_type == "taxon_pie_chart":
                saved.extend(figures.taxon_pie_chart(self.latest_taxon_results, fig_dir))
            elif fig_type == "peptide_heatmap":
                saved.extend(figures.peptide_heatmap(data_path, fig_dir))
            elif fig_type == "score_distribution":
                saved.extend(figures.score_distribution(data_path, fig_dir))
        self.latest_figures = saved
        return {"status": "ok", "saved_to": saved}

    def export_report(self, fmt: str) -> dict:
        """Export summary report in text format."""

        if fmt != "txt":
            return {"status": "error", "message": "Only txt export is currently supported."}
        report_dir = self.run_dir / "report"
        path = export_summary(self.state, self.latest_taxon_results, self.latest_figures, report_dir)
        return {"status": "ok", "path": path}

    def _execute_action(self, action: dict) -> dict:
        """Dispatch action to tool method."""

        tool = str(action.get("tool", ""))
        stage = action.get("stage")
        params = action.get("params", {})
        if not isinstance(params, dict):
            params = {}

        approved, params = self._approve_action(tool, str(stage) if stage else None, params)
        if not approved:
            return {"status": "cancelled", "message": "Action cancelled by user."}

        if tool == "run_pipeline_stage":
            return self.run_pipeline_stage(str(stage), params)
        if tool == "run_taxon_inference":
            return self.run_taxon_inference(str(params.get("algorithm", self.state.taxon_algorithm)), params)
        if tool == "show_state":
            return self.show_state()
        if tool == "generate_figures":
            return self.generate_figures(list(params.get("types", [])), str(params.get("data_path", "")))
        if tool == "export_report":
            return self.export_report(str(params.get("format", "txt")))
        return {"status": "error", "message": f"Unknown tool: {tool}"}

    # Ordered pipeline stages for no-LLM automatic execution.
    PIPELINE_STAGES = [
        "format_conversion",
        "peptide_id",
        "validation",
        "quantitation",
        "protein_assignment",
        "taxon_inference",
    ]

    def _next_incomplete_stage(self) -> str | None:
        """Return the name of the next incomplete pipeline stage, or None."""
        for stage in self.PIPELINE_STAGES:
            if stage not in self.state.completed_stages:
                return stage
        return None

    def _run_no_llm(self) -> None:
        """Run pipeline stages sequentially without LLM, prompting between stages."""

        algorithm = os.getenv("TAXON_ALGORITHM") or self.state.taxon_algorithm or "abundance_em"
        self.console.print(
            f"[cyan]Taxon algorithm: {algorithm} "
            f"(from {'env' if os.getenv('TAXON_ALGORITHM') else 'state'})[/cyan]"
        )
        self.console.print("[cyan]Running in no-LLM mode. Stages will execute automatically.[/cyan]")
        while True:
            stage = self._next_incomplete_stage()
            if stage is None:
                self.console.print(Panel("All pipeline stages complete!", style="green"))
                self.console.print(f"[bold]Completed stages:[/bold] {', '.join(self.state.completed_stages)}")
                for s, out in self.state.stage_outputs.items():
                    self.console.print(f"  {s}: {out}")
                break

            self.console.print(f"\n[bold cyan]>>> Running stage: {stage}[/bold cyan]")
            if stage == "taxon_inference":
                algorithm = os.getenv("TAXON_ALGORITHM") or self.state.taxon_algorithm or "abundance_em"
                self.console.print(f"[cyan]Using taxon algorithm: {algorithm}[/cyan]")
                result = self.run_taxon_inference(algorithm, {})
            else:
                result = self.run_pipeline_stage(stage, {})
            self.console.print(Panel(json.dumps(result, indent=2), title=f"Stage Result: {stage}"))

            if result.get("status") != "ok":
                self.console.print(f"[red]Stage {stage} failed. Stopping.[/red]")
                break

            next_stage = self._next_incomplete_stage()
            if next_stage is not None:
                answer = input("Continue to next stage? (y/n) ").strip().lower()
                if answer != "y":
                    self.console.print("[green]Session ended by user.[/green]")
                    break

    def run(self) -> None:
        """Start interactive conversation loop and process LLM-directed actions."""

        if self.settings.no_llm_mode:
            self._run_no_llm()
            return

        self.console.print("[green]Type 'exit' or 'quit' to end the session.[/green]")
        while True:
            user_text = input("> ").strip()
            if user_text.lower() in {"exit", "quit"}:
                self.console.print("[green]Session ended.[/green]")
                break

            self.history.append({"role": "user", "content": user_text})
            system_prompt = self._build_system_prompt()
            response = llm_client.chat(self.history, system_prompt)
            self.console.print(Markdown(response))
            self.history.append({"role": "assistant", "content": response})

            action = self._parse_action(response)
            if action:
                result = self._execute_action(action)
                tool_msg = f"Tool result: {json.dumps(result, indent=2)}"
                self.console.print(Panel(tool_msg, title="Action Result"))
                self.history.append({"role": "assistant", "content": tool_msg})
