"""Interactive orchestrator for metaproteomics agentic pipeline execution."""

from __future__ import annotations

import datetime
import json
import os
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

        self.log_path = self.run_dir / "chat.log"
        self._turn = 0
        self.run_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.log_path.open("a", encoding="utf-8") as _f:
            _f.write(
                f"\n{'═' * 80}\n"
                f"SESSION STARTED: {ts} | Run ID: {self.state.run_id} | Backend: {settings.llm_backend}\n"
                f"{'═' * 80}\n"
            )

        self.stages = {
            "format_conversion": FormatConversion(),
            "peptide_id": PeptideIdentification(),
            "validation": PeptideValidation(),
            "quantitation": Quantitation(),
            "protein_assignment": ProteinAssignment(),
        }
        self.taxon_registry = TaxonRegistry()
        self.history: list[dict] = []
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

    def _log_event(self, label: str, content: str) -> None:
        """Append a labelled, timestamped event block to the chat log file."""
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sep = "─" * 80
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(f"\n{sep}\n[{ts}] {label}\n{sep}\n{content}\n")

    def _approve_action(self, tool_name: str, tool_input: dict) -> tuple[bool, dict]:
        """Apply autonomy mode gating and optionally allow full input edits."""

        mode = self.state.autonomy_mode
        should_prompt = mode == "supervised" or (
            mode == "balanced" and tool_name == "run_pipeline_stage"
        )
        if not should_prompt:
            return True, tool_input

        self.console.print(
            f"[cyan]Proposed tool call:[/cyan] {tool_name}({json.dumps(tool_input)})"
        )
        answer = input("Proceed? (y/n/edit): ").strip().lower()
        if answer == "y":
            return True, tool_input
        if answer == "edit":
            raw = input("Enter new tool input JSON: ").strip()
            try:
                return True, json.loads(raw)
            except json.JSONDecodeError:
                return False, tool_input
        return False, tool_input

    def _dispatch_tool(self, tool_name: str, tool_input: dict) -> dict:
        """Execute a named tool with its input and return the result dict."""

        if tool_name == "run_pipeline_stage":
            return self.run_pipeline_stage(
                str(tool_input.get("stage", "")),
                tool_input.get("params") or {},
            )
        if tool_name == "run_taxon_inference":
            algorithm = str(tool_input.get("algorithm", self.state.taxon_algorithm))
            return self.run_taxon_inference(algorithm, tool_input)
        if tool_name == "show_state":
            return self.show_state()
        if tool_name == "generate_figures":
            return self.generate_figures(
                list(tool_input.get("types", [])),
                str(tool_input.get("data_path", "")),
            )
        if tool_name == "export_report":
            return self.export_report(str(tool_input.get("format", "txt")))
        return {"status": "error", "message": f"Unknown tool: {tool_name}"}

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
            "comet_params_path": self.settings.comet_params_path,
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

    _OTHER_SENTINEL = "__other__"

    def ask_question(self, params: dict) -> dict:
        """Present an interactive question with arrow-key navigation."""

        import questionary
        from questionary import Choice, Style

        question = str(params.get("question", "What would you like to do?"))
        q_type = str(params.get("type", "decision"))
        options = params.get("options", [])
        if not isinstance(options, list):
            options = []

        type_labels = {
            "clarification": "Clarification Needed",
            "decision": "Decision",
            "parameter": "Parameter Input",
        }
        type_label = type_labels.get(q_type, q_type.title())

        style = Style([
            ("qmark", "fg:cyan bold"),
            ("question", "bold"),
            ("answer", "fg:green bold"),
            ("pointer", "fg:cyan bold"),
            ("highlighted", "fg:cyan bold"),
            ("selected", "fg:green"),
        ])

        # Build questionary choices
        choices: list[Choice] = []
        for opt in options:
            if isinstance(opt, str):
                label, desc = opt, ""
            else:
                label = opt.get("label", "")
                desc = opt.get("description", "")
            title = f"{label} — {desc}" if desc else label
            choices.append(Choice(title=title, value=label))
        choices.append(Choice(
            title="Other (type your own response)",
            value=self._OTHER_SENTINEL,
        ))

        # Show context panel
        self.console.print(Panel(
            f"[bold]{question}[/bold]",
            title=type_label,
            border_style="cyan",
        ))

        # Selection loop — user can go back from "Other" to the option list
        while True:
            selected = questionary.select(
                "Select an option:",
                choices=choices,
                style=style,
                instruction="(arrow keys to move, enter to select)",
            ).ask()

            if selected is None:
                # User pressed Ctrl-C — treat as cancel
                return {"status": "cancelled", "message": "Question cancelled by user."}

            if selected != self._OTHER_SENTINEL:
                self.console.print(f"\n  [green]Selected:[/green] {selected}\n")
                return {"status": "ok", "answer": selected, "type": q_type}

            # "Other" selected — prompt for free text
            custom = questionary.text(
                "Your response:",
                style=style,
            ).ask()

            if custom is None or not custom.strip():
                # Ctrl-C or empty — go back to option list
                self.console.print("[dim]  Going back to options...[/dim]\n")
                continue

            custom = custom.strip()

            # Confirm or go back
            confirm = questionary.confirm(
                f'Send "{custom}"?',
                default=True,
                style=style,
            ).ask()

            if confirm:
                self.console.print(f"\n  [green]Selected:[/green] {custom}\n")
                return {"status": "ok", "answer": custom, "type": q_type}

            # User declined — back to option list
            self.console.print("[dim]  Going back to options...[/dim]\n")
            continue


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
        """Start interactive conversation loop and process LLM-directed tool calls."""

        if self.settings.no_llm_mode:
            self._run_no_llm()
            return

        self.console.print("[green]Type 'exit' or 'quit' to end the session.[/green]")
        while True:
            # ── Outer loop: wait for user input ──────────────────────────────
            user_text = input("> ").strip()
            if user_text.lower() in {"exit", "quit"}:
                self.console.print("[green]Session ended.[/green]")
                break

            self._turn += 1
            self.history.append({"role": "user", "content": user_text})
            self._log_event(f"TURN {self._turn} — USER MESSAGE", user_text)

            # ── Inner agentic loop: keep querying until no tool call ──────────
            while True:
                system_prompt = self._build_system_prompt()
                self._log_event(f"TURN {self._turn} — SYSTEM PROMPT", system_prompt)
                self._log_event(
                    f"TURN {self._turn} — CONVERSATION HISTORY (sent to LLM)",
                    json.dumps(self.history, indent=2, ensure_ascii=False),
                )

                response = llm_client.chat(self.history, system_prompt, self.settings)

                if response.text:
                    self.console.print(Markdown(response.text))
                    self._log_event(f"TURN {self._turn} — LLM RESPONSE (text)", response.text)

                if response.tool_call is None:
                    # Pure text reply — add to history and hand control back to user.
                    if response.text:
                        self.history.append({"role": "assistant", "content": response.text})
                    break

                tool_call = response.tool_call
                tool_name = tool_call["name"]
                tool_input = tool_call["input"]

                self._log_event(
                    f"TURN {self._turn} — TOOL CALL [PRE-EXECUTE]",
                    json.dumps(tool_call, indent=2, ensure_ascii=False),
                )

                # Record the assistant turn (text + tool_use) in history.
                self.history.append(
                    {"role": "assistant_with_tool", "text": response.text, "tool_call": tool_call}
                )

                # ask_question skips the approval gate — it *is* asking the user.
                if tool_name == "ask_question":
                    result = self.ask_question(tool_input)
                else:
                    approved, tool_input = self._approve_action(tool_name, tool_input)
                    if not approved:
                        result = {"status": "cancelled", "message": "Action cancelled by user."}
                    else:
                        result = self._dispatch_tool(tool_name, tool_input)
                    self.console.print(
                        Panel(json.dumps(result, indent=2), title=f"Tool Result: {tool_name}")
                    )

                self._log_event(
                    f"TURN {self._turn} — TOOL RESULT",
                    json.dumps(result, indent=2, ensure_ascii=False),
                )

                # Append the tool result so the LLM sees it on the next inner iteration.
                self.history.append(
                    {
                        "role": "tool_result",
                        "tool_use_id": tool_call["id"],
                        "content": json.dumps(result),
                    }
                )

                # If ask_question was cancelled, stop the inner loop and let the user type.
                if tool_name == "ask_question" and result.get("status") != "ok":
                    self.console.print("[dim]Question cancelled.[/dim]")
                    break
                # Otherwise continue the inner loop — the LLM reacts to the result.
