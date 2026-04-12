"""LLM chat client supporting llama-cpp, OpenAI-compatible, and Anthropic backends.

All backends return an LLMResponse containing separate text and tool_call fields,
using each provider's native tool/function-calling API.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rich.console import Console

from agent.tools import TOOLS_ANTHROPIC, tools_for_openai

if TYPE_CHECKING:
    from config import Settings


# Injected into the system prompt only when a local llama model doesn't support
# native function calling.  Claude and OpenAI backends never see this block.
_ACTION_FALLBACK_INSTRUCTIONS = """
You do not have native tool-calling support. Instead, when you need to call a tool,
emit exactly one ACTION block in your response using this format:

<ACTION>
tool: <run_pipeline_stage | run_taxon_inference | show_state | generate_figures | export_report | ask_question>
stage: <stage_name>   (only for run_pipeline_stage)
params: {"key": "value"}   (valid JSON; use {} if no params)
</ACTION>

Available tools and their params are described in the system prompt above.
Never emit more than one ACTION block per response.
"""


@dataclass
class LLMResponse:
    """Structured response from the LLM."""

    text: str
    tool_call: dict | None = None  # {"id": str, "name": str, "input": dict}


# ---------------------------------------------------------------------------
# History conversion helpers
# ---------------------------------------------------------------------------

def _to_claude_messages(history: list[dict]) -> list[dict]:
    """Convert internal history to Anthropic messages format."""
    messages = []
    for msg in history:
        role = msg["role"]
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": msg["content"]})
        elif role == "assistant_with_tool":
            content: list[dict] = []
            if msg.get("text"):
                content.append({"type": "text", "text": msg["text"]})
            tc = msg["tool_call"]
            content.append(
                {"type": "tool_use", "id": tc["id"], "name": tc["name"], "input": tc["input"]}
            )
            messages.append({"role": "assistant", "content": content})
        elif role == "tool_result":
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg["tool_use_id"],
                            "content": msg["content"],
                        }
                    ],
                }
            )
    return messages


def _to_openai_messages(history: list[dict], system_prompt: str) -> list[dict]:
    """Convert internal history to OpenAI messages format."""
    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    for msg in history:
        role = msg["role"]
        if role in {"user", "assistant"}:
            messages.append({"role": role, "content": msg["content"]})
        elif role == "assistant_with_tool":
            tc = msg["tool_call"]
            messages.append(
                {
                    "role": "assistant",
                    "content": msg.get("text") or None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["input"]),
                            },
                        }
                    ],
                }
            )
        elif role == "tool_result":
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": msg["tool_use_id"],
                    "content": msg["content"],
                }
            )
    return messages


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _call_claude(
    history: list[dict],
    system_prompt: str,
    temperature: float,
    base_url: str,
    api_key: str,
    model: str,
) -> LLMResponse:
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
        messages=_to_claude_messages(history),
        tools=TOOLS_ANTHROPIC,
    )

    text = ""
    tool_call = None
    for block in response.content:
        if block.type == "text":
            text += block.text
        elif block.type == "tool_use":
            tool_call = {"id": block.id, "name": block.name, "input": block.input}

    return LLMResponse(text=text, tool_call=tool_call)


def _call_openai_compatible(
    history: list[dict],
    system_prompt: str,
    temperature: float,
    base_url: str,
    api_key: str,
    model: str,
) -> LLMResponse:
    from openai import OpenAI

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Set it in .env or environment variables."
        )

    client = OpenAI(base_url=base_url, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        temperature=temperature,
        messages=_to_openai_messages(history, system_prompt),
        tools=tools_for_openai(),
        tool_choice="auto",
    )

    message = response.choices[0].message
    text = message.content or ""
    tool_call = None
    if message.tool_calls:
        tc = message.tool_calls[0]
        tool_call = {
            "id": tc.id,
            "name": tc.function.name,
            "input": json.loads(tc.function.arguments),
        }

    return LLMResponse(text=text, tool_call=tool_call)


def _parse_text_action(text: str) -> dict | None:
    """Extract a tool call from a plain-text <ACTION> block (llama fallback only)."""
    import re

    match = re.search(r"<ACTION>(.*?)</ACTION>", text, flags=re.DOTALL)
    if not match:
        return None
    action: dict = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key, value = key.strip(), value.strip()
        if key == "params":
            try:
                action[key] = json.loads(value) if value else {}
            except json.JSONDecodeError:
                return None
        else:
            action[key] = value
    tool_name = action.pop("tool", None)
    if not tool_name:
        return None
    return {
        "id": f"text-{hash(text) & 0xFFFFFF:06x}",
        "name": tool_name,
        "input": {k: v for k, v in action.items()},
    }


def _call_llama(
    history: list[dict],
    system_prompt: str,
    temperature: float,
    base_url: str,
) -> LLMResponse:
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="none")
    messages = _to_openai_messages(history, system_prompt)

    # Try native function calling first.
    try:
        response = client.chat.completions.create(
            model="local",
            temperature=temperature,
            messages=messages,
            tools=tools_for_openai(),
            tool_choice="auto",
        )
        message = response.choices[0].message
        text = message.content or ""
        tool_call = None
        if message.tool_calls:
            tc = message.tool_calls[0]
            tool_call = {
                "id": tc.id,
                "name": tc.function.name,
                "input": json.loads(tc.function.arguments),
            }
        return LLMResponse(text=text, tool_call=tool_call)
    except Exception:  # noqa: BLE001
        # Model does not support function calling — inject <ACTION> instructions
        # and parse the text response.
        fallback_system = system_prompt + _ACTION_FALLBACK_INSTRUCTIONS
        fallback_messages = _to_openai_messages(history, fallback_system)
        response = client.chat.completions.create(
            model="local",
            temperature=temperature,
            messages=fallback_messages,
        )
        text = response.choices[0].message.content or ""
        return LLMResponse(text=text, tool_call=_parse_text_action(text))


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def chat(history: list[dict], system_prompt: str, settings: "Settings") -> LLMResponse:
    """Send the conversation history to the configured LLM backend and return a structured response."""

    backend = settings.llm_backend
    console = Console()
    retries = [1, 2, 4]

    for attempt, backoff in enumerate(retries, start=1):
        try:
            if backend == "claude":
                return _call_claude(
                    history,
                    system_prompt,
                    temperature=0.2,
                    base_url=settings.anthropic_base_url,
                    api_key=settings.anthropic_api_key,
                    model=settings.anthropic_model_id,
                )
            if backend == "openai":
                return _call_openai_compatible(
                    history,
                    system_prompt,
                    temperature=0.2,
                    base_url=settings.openai_api_url,
                    api_key=settings.openai_api_key,
                    model=settings.openai_model_id,
                )
            return _call_llama(
                history,
                system_prompt,
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
