"""Shared LLM access for all three agents.

LLM_PROVIDER picks the backend:
  * "anthropic" (default) -- Claude via the official `anthropic` SDK, using
    your commercial ANTHROPIC_API_KEY. Model: CLAUDE_MODEL (claude-opus-5).
  * "ollama" -- free open-source model served locally by Ollama
    (https://ollama.com). No key, no per-call cost.
  * "gemini" -- Google Gemini API free tier (GOOGLE_API_KEY).

Every agent asks for *structured* output: it passes a Pydantic model and gets
a validated instance back, so downstream code never string-parses LLM text.

Resilience (plan T4.3 / risk table):
  * calls are retried with exponential backoff (Anthropic SDK `max_retries`;
    LangChain `with_retry` for the free providers);
  * any failure that survives the retries -- or a refusal -- is raised as
    `LLMError`, which the workflow catches so one bad call never crashes a run;
  * LLM_CACHE=true caches responses on disk by prompt hash, to keep cost and
    rate limits down while iterating on the same test patients.

Keys come from the environment (on EC2: fed from SSM Parameter Store). They
are never hardcoded, logged or written to disk.
"""
import hashlib
import json
import os
from functools import lru_cache
from typing import TypeVar

from pydantic import BaseModel

from . import config

T = TypeVar("T", bound=BaseModel)

CACHE_DIR = os.path.join(config.LOCAL_DATA_DIR, "..", ".cache", "llm")


class LLMError(RuntimeError):
    """An LLM call failed after retries, was refused, or returned no valid structured output."""


def model_name() -> str:
    return {
        "anthropic": config.CLAUDE_MODEL,
        "gemini": config.GEMINI_MODEL,
        "ollama": config.OLLAMA_MODEL,
    }.get(config.LLM_PROVIDER, "?")


# --- Claude (commercial key) -----------------------------------------------------


@lru_cache(maxsize=1)
def get_anthropic_client():
    import anthropic

    # Reads ANTHROPIC_API_KEY from the environment.
    return anthropic.Anthropic(max_retries=4)


def _ask_claude(prompt: str, schema: type[T], system: str, effort: str) -> T:
    import anthropic

    request = dict(
        model=config.CLAUDE_MODEL,
        max_tokens=16000,
        system=system,
        output_config={"effort": effort},
        output_format=schema,
        messages=[{"role": "user", "content": prompt}],
    )
    client = get_anthropic_client()
    try:
        if config.CLAUDE_FALLBACKS == "default":
            # Server-side refusal fallback: if the model's safety classifiers
            # decline (e.g. a false positive on clinical wording), the API
            # re-runs the request on Anthropic's recommended fallback model.
            response = client.beta.messages.parse(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
            )
        else:
            response = client.messages.parse(**request)
    except anthropic.APIError as exc:
        raise LLMError(f"{type(exc).__name__}: {exc}") from exc

    if response.stop_reason == "refusal":
        raise LLMError("the model declined this request")
    if response.stop_reason == "max_tokens" or response.parsed_output is None:
        raise LLMError(f"no valid structured output (stop_reason={response.stop_reason})")
    return response.parsed_output


# --- Free providers (LangChain) -----------------------------------------------------


@lru_cache(maxsize=1)
def get_chat_model():
    if config.LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Reads GOOGLE_API_KEY from the environment.
        return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, temperature=0)
    if config.LLM_PROVIDER == "ollama":
        from langchain_ollama import ChatOllama

        return ChatOllama(model=config.OLLAMA_MODEL, base_url=config.OLLAMA_BASE_URL, temperature=0)
    raise ValueError(f"LLM_PROVIDER must be anthropic, ollama or gemini; got {config.LLM_PROVIDER!r}")


def _ask_langchain(prompt: str, schema: type[T], system: str) -> T:
    chain = get_chat_model().with_structured_output(schema).with_retry(
        stop_after_attempt=3, wait_exponential_jitter=True
    )
    try:
        result = chain.invoke([("system", system), ("human", prompt)])
    except Exception as exc:  # provider SDKs raise unrelated exception types
        raise LLMError(f"{type(exc).__name__}: {exc}") from exc
    if not isinstance(result, schema):
        raise LLMError(f"no valid structured output from {config.LLM_PROVIDER}:{model_name()}")
    return result


# --- Entry point ---------------------------------------------------------------------


def _cache_path(system: str, prompt: str, schema: type[BaseModel], effort: str) -> str:
    key = json.dumps([config.LLM_PROVIDER, model_name(), effort, system, prompt, schema.model_json_schema()],
                     sort_keys=True)
    return os.path.join(CACHE_DIR, hashlib.sha256(key.encode()).hexdigest() + ".json")


def ask_structured(prompt: str, schema: type[T], *, system: str, effort: str = "medium") -> T:
    """Sends one prompt to the configured LLM and returns a validated `schema` instance.

    `effort` (low | medium | high) applies to Claude only."""
    cache_file = _cache_path(system, prompt, schema, effort) if config.LLM_CACHE else None
    if cache_file and os.path.exists(cache_file):
        with open(cache_file) as f:
            return schema.model_validate_json(f.read())

    if config.LLM_PROVIDER == "anthropic":
        result = _ask_claude(prompt, schema, system, effort)
    else:
        result = _ask_langchain(prompt, schema, system)

    if cache_file:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_file, "w") as f:
            f.write(result.model_dump_json())
    return result
