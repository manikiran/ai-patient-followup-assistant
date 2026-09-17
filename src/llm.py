"""Shared Claude client for all agents, via the official Anthropic SDK.

Reads ANTHROPIC_API_KEY (and optionally CLAUDE_MODEL) from the environment --
loaded from a local .env file via python-dotenv. Never hardcode the key here
or log it; anthropic.Anthropic() picks ANTHROPIC_API_KEY up from the
environment on its own.

Model selection is dynamic, not baked in at import time: DEFAULT_MODEL is
just the fallback when a caller doesn't ask for a specific one. Every agent
call accepts an optional `model` (see agents.py's FollowUpState["model"]),
so the model can be switched per-request -- e.g. from Open WebUI's own model
picker -- with no server restart.
"""
import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODEL = os.getenv("CLAUDE_MODEL", "claude-opus-5")

# Keep this in sync with what /v1/models advertises to Open WebUI. Anything
# not in this set falls back to DEFAULT_MODEL rather than hitting the API
# with an unvalidated string.
AVAILABLE_MODELS = ["claude-opus-5", "claude-sonnet-5", "claude-haiku-4-5"]

_client = anthropic.Anthropic()


def resolve_model(requested: str | None) -> str:
    if requested and requested in AVAILABLE_MODELS:
        return requested
    return DEFAULT_MODEL


def ask(prompt: str, max_tokens: int = 600, effort: str = "medium", model: str | None = None) -> str:
    """Sends one prompt to Claude and returns the text response."""
    response = _client.messages.create(
        model=resolve_model(model),
        max_tokens=max_tokens,
        output_config={"effort": effort},
        messages=[{"role": "user", "content": prompt}],
    )
    return next((block.text for block in response.content if block.type == "text"), "")
