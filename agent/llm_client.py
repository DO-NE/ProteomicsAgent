"""LLM chat client supporting llama-cpp OpenAI mode and Anthropic fallback."""

from __future__ import annotations

import os
import time

from rich.console import Console


def _call_llama(messages: list[dict[str, str]], temperature: float) -> str:
    from openai import OpenAI

    client = OpenAI(
        base_url=os.getenv("LLAMA_SERVER_URL", "http://localhost:8000/v1"),
        api_key="none",
    )
    response = client.chat.completions.create(
        model="local",
        temperature=temperature,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def _call_claude(messages: list[dict[str, str]], system_prompt: str, temperature: float) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        temperature=temperature,
        system=system_prompt,
        messages=[m for m in messages if m.get("role") in {"user", "assistant"}],
    )
    if not response.content:
        return ""
    return "".join(getattr(block, "text", "") for block in response.content)


def chat(messages: list[dict], system_prompt: str) -> str:
    """Send chat messages to configured backend with retries and return text output."""

    backend = os.getenv("LLM_BACKEND", "llama").strip().lower()
    console = Console()
    retries = [1, 2, 4]
    prepared_messages = [{"role": "system", "content": system_prompt}] + [
        {"role": m["role"], "content": m["content"]} for m in messages
    ]

    for attempt, backoff in enumerate(retries, start=1):
        try:
            if backend == "claude":
                return _call_claude(prepared_messages, system_prompt, temperature=0.2)
            return _call_llama(prepared_messages, temperature=0.2)
        except Exception as exc:  # noqa: BLE001
            console.print(f"[yellow]LLM request failed (attempt {attempt}/3): {exc}[/yellow]")
            if attempt < len(retries):
                time.sleep(backoff)

    raise RuntimeError(
        "LLM request failed after 3 attempts. Please check that the llama-cpp-python "
        "server is running on port 8000."
    )
