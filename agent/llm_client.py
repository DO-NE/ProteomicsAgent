"""LLM chat client supporting llama-cpp, OpenAI-compatible, and Anthropic backends."""

from __future__ import annotations

import re
import time
from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from config import Settings


def _call_llama(
    messages: list[dict[str, str]],
    temperature: float,
    base_url: str,
) -> str:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="none")
    response = client.chat.completions.create(
        model="local",
        temperature=temperature,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def _call_openai_compatible(
    messages: list[dict[str, str]],
    temperature: float,
    base_url: str,
    api_key: str,
    model: str,
) -> str:
    from openai import OpenAI

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in .env or environment variables."
        )
    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def _call_claude(
    messages: list[dict[str, str]],
    system_prompt: str,
    temperature: float,
    base_url: str,
    api_key: str,
    model: str,
) -> str:
    import anthropic

    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Set it in .env or environment variables."
        )
    client = anthropic.Anthropic(base_url=base_url, api_key=api_key)
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=temperature,
        system=system_prompt,
        messages=[m for m in messages if m.get("role") in {"user", "assistant"}],
    )
    if not response.content:
        return ""
    return "".join(getattr(block, "text", "") for block in response.content)


def _next_incomplete_stage(system_prompt: str) -> str:
    """Parse completed stages from the system prompt and return the next ACTION."""

    stages = [
        "format_conversion", "peptide_id", "validation",
        "quantitation", "protein_assignment",
    ]
    completed: list[str] = []
    # Extract completed_stages from the run_state JSON embedded in the prompt
    match = re.search(r'"completed_stages":\s*\[([^\]]*)\]', system_prompt)
    if match:
        completed = [s.strip().strip('"') for s in match.group(1).split(",") if s.strip()]

    for stage in stages:
        if stage not in completed:
            return (
                f"Running next pipeline stage automatically (no-LLM mode).\n"
                f"<ACTION>\ntool: run_pipeline_stage\nstage: {stage}\nparams: {{}}\n</ACTION>"
            )
    return (
        "All pipeline stages are complete.\n"
        "<ACTION>\ntool: show_state\nparams: {}\n</ACTION>"
    )


def chat(messages: list[dict], system_prompt: str, settings: Settings) -> str:
    """Send chat messages to configured backend with retries and return text output."""

    if settings.no_llm_mode:
        return _next_incomplete_stage(system_prompt)

    backend = settings.llm_backend
    console = Console()
    retries = [1, 2, 4]
    prepared_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]

    for attempt, backoff in enumerate(retries, start=1):
        try:
            if backend == "claude":
                return _call_claude(
                    prepared_messages,
                    system_prompt,
                    temperature=0.2,
                    base_url=settings.anthropic_base_url,
                    api_key=settings.anthropic_api_key,
                    model=settings.anthropic_model_id,
                )
            if backend == "openai":
                return _call_openai_compatible(
                    prepared_messages,
                    temperature=0.2,
                    base_url=settings.openai_api_url,
                    api_key=settings.openai_api_key,
                    model=settings.openai_model_id,
                )
            return _call_llama(
                prepared_messages,
                temperature=0.2,
                base_url=settings.llama_server_url,
            )
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]LLM request failed (attempt {attempt}/3): {exc}[/yellow]")
            if attempt < len(retries):
                time.sleep(backoff)

    backend_hints = {
        "llama": "Check that the llama-cpp-python server is running.",
        "openai": "Check OPENAI_API_URL, OPENAI_API_KEY, and OPENAI_MODEL_ID in .env.",
        "claude": "Check ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, and ANTHROPIC_MODEL_ID in .env.",
    }
    raise RuntimeError(
        f"LLM request failed after 3 attempts ({backend} backend). "
        f"{backend_hints.get(backend, 'Check your LLM configuration.')}"
    )
