"""Tool definitions for the metaproteomics agent.

Tools are defined in Anthropic's tool_use schema format and converted to
OpenAI's function-calling format when needed.
"""

from __future__ import annotations


TOOLS_ANTHROPIC: list[dict] = [
    {
        "name": "run_pipeline_stage",
        "description": (
            "Execute a named pipeline stage. "
            "Stages must run in order: format_conversion → peptide_id → validation "
            "→ quantitation → protein_assignment. "
            "Skip format_conversion if the input files are already in mzML format."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stage": {
                    "type": "string",
                    "enum": [
                        "format_conversion",
                        "peptide_id",
                        "validation",
                        "quantitation",
                        "protein_assignment",
                    ],
                    "description": "The pipeline stage to execute.",
                },
                "params": {
                    "type": "object",
                    "description": "Stage-specific parameters (all optional).",
                    "properties": {
                        "input_path": {
                            "type": "string",
                            "description": (
                                "Override input file path. Auto-resolved from the "
                                "previous stage's output if omitted."
                            ),
                        },
                        "tool": {
                            "type": "string",
                            "enum": ["msfragger", "comet"],
                            "description": (
                                "Search engine for the peptide_id stage. "
                                "Defaults to 'msfragger'. "
                                "Use 'comet' to run Comet instead "
                                "(uses COMET_PARAMS_PATH if configured)."
                            ),
                        },
                    },
                },
            },
            "required": ["stage"],
        },
    },
    {
        "name": "run_taxon_inference",
        "description": (
            "Run taxonomic inference on peptides identified during the validation stage. "
            "Requires the validation stage to be complete."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "algorithm": {
                    "type": "string",
                    "description": "Taxon inference algorithm name (see available_algorithms in run state).",
                },
                "fasta_path": {
                    "type": "string",
                    "description": "Override the protein FASTA database path.",
                },
                "database_path": {
                    "type": "string",
                    "description": "Override the local taxonomy database path.",
                },
            },
            "required": ["algorithm"],
        },
    },
    {
        "name": "show_state",
        "description": "Return the current run state: completed stages, output file paths, and settings.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "generate_figures",
        "description": "Generate visualization figures from pipeline results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "taxon_bar_chart",
                            "taxon_pie_chart",
                            "peptide_heatmap",
                            "score_distribution",
                        ],
                    },
                    "description": "List of figure types to generate.",
                    "minItems": 1,
                },
                "data_path": {
                    "type": "string",
                    "description": "Path to the input data file (e.g. pepXML or TSV results).",
                },
            },
            "required": ["types"],
        },
    },
    {
        "name": "export_report",
        "description": "Export a summary report of the completed run.",
        "input_schema": {
            "type": "object",
            "properties": {
                "format": {
                    "type": "string",
                    "enum": ["txt"],
                    "description": "Output format. Only 'txt' is currently supported.",
                },
            },
        },
    },
    {
        "name": "ask_question",
        "description": (
            "Ask the user a question to get clarification, a decision, or a parameter value "
            "before proceeding. The user will see an interactive list and can choose with "
            "arrow keys or type a custom free-text response."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Clear, specific question for the user.",
                },
                "type": {
                    "type": "string",
                    "enum": ["clarification", "decision", "parameter"],
                    "description": (
                        "'clarification': need more information to understand intent. "
                        "'decision': user must choose between valid alternatives. "
                        "'parameter': need a specific value (threshold, path, algorithm name, etc.)."
                    ),
                },
                "options": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "label": {"type": "string"},
                            "description": {"type": "string"},
                        },
                        "required": ["label"],
                    },
                    "description": "1–4 selectable options presented to the user.",
                    "minItems": 1,
                },
            },
            "required": ["question", "type", "options"],
        },
    },
]


def tools_for_openai() -> list[dict]:
    """Convert Anthropic-format tool definitions to OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["input_schema"],
            },
        }
        for tool in TOOLS_ANTHROPIC
    ]
