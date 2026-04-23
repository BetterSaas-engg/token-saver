"""
Thin wrapper around the Anthropic SDK.

Single responsibility: send a prompt to Claude, return the text response
plus usage metadata. No retries, no streaming, no business logic.
If we ever need to swap providers or add instrumentation, this is the
only file that changes.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv


# Load .env from the backend directory regardless of where this is run from
_ENV_PATH = Path(__file__).parent / ".env"
load_dotenv(_ENV_PATH)

_API_KEY = os.getenv("ANTHROPIC_API_KEY")
_MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")

if not _API_KEY:
    raise RuntimeError(
        f"ANTHROPIC_API_KEY not found. Expected it in {_ENV_PATH}"
    )

_client = Anthropic(api_key=_API_KEY)


@dataclass
class LLMResponse:
    """Result of a Claude API call."""
    text: str
    input_tokens: int
    output_tokens: int
    model: str


def send(prompt: str, max_tokens: int = 1024) -> LLMResponse:
    """
    Send a prompt to Claude and return the response.

    The prompt is sent as a single user message. No system prompt,
    no conversation history. Callers that need those should build
    them into the prompt string for now.
    """
    if not prompt or not prompt.strip():
        raise ValueError("Prompt cannot be empty")

    response = _client.messages.create(
        model=_MODEL,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )

    # Claude returns a list of content blocks; v1 only uses text blocks
    text = "".join(
        block.text for block in response.content if block.type == "text"
    )

    return LLMResponse(
        text=text,
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        model=response.model,
    )


# Quick self-test
# Run this file directly to verify: python backend\llm_client.py

if __name__ == "__main__":
    print("LLM Client -- Self Test\n")
    print(f"Using model: {_MODEL}")
    print(f"API key loaded: {_API_KEY[:15]}...{_API_KEY[-4:]}\n")

    result = send("Reply with exactly three words: token saver works")

    print(f"Response: {result.text}")
    print(f"Input tokens (reported by API): {result.input_tokens}")
    print(f"Output tokens (reported by API): {result.output_tokens}")
    print(f"Model confirmed: {result.model}")
