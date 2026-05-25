"""
LLM Factory — Single switching point for all agent LLM backends.

To change the LLM provider, edit only your .env file:

    LLM_PROVIDER=groq          → uses Groq  (your gsk_ key)
    LLM_PROVIDER=openai        → uses OpenAI (your sk- key)

No code changes are needed in advocate.py, judge.py, or pipeline.py.

Supported providers:
    groq   : llama-3.3-70b-versatile (default), llama-3.1-8b-instant
    openai : gpt-4o (default), gpt-4-turbo, gpt-3.5-turbo

.env example:
    LLM_PROVIDER=groq
    GROQ_API_KEY=gsk_...
    LLM_MODEL=llama3-70b-8192    # optional, uses provider default if unset
"""

import os
import logging
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ── Default model per provider ──────────────────────────────────────
_PROVIDER_DEFAULTS = {
    "groq":   "llama-3.3-70b-versatile",
    "openai": "gpt-4o",
}


def get_llm(model_name: str = None,
            temperature: float = 0.3,
            api_key: str = None):
    """
    Return the correct LangChain chat LLM based on LLM_PROVIDER in .env.

    Args:
        model_name  : Override model name. If None, uses LLM_MODEL env var,
                      then falls back to the provider default.
        temperature : LLM temperature (lower = more deterministic).
        api_key     : Explicit API key override. If None, reads from env
                      (GROQ_API_KEY or OPENAI_API_KEY depending on provider).

    Returns:
        A LangChain chat model instance (ChatGroq or ChatOpenAI).

    Raises:
        EnvironmentError : if the required API key is missing.
        ValueError       : if LLM_PROVIDER is unrecognised.
    """
    provider = os.getenv("LLM_PROVIDER", "groq").strip().lower()

    # Resolve model name: arg > env var > provider default
    resolved_model = (
        model_name
        or os.getenv("LLM_MODEL")
        or _PROVIDER_DEFAULTS.get(provider)
    )

    logger.info(f"[LLM Factory] provider={provider}, model={resolved_model}, "
                f"temperature={temperature}")

    # ── Groq ────────────────────────────────────────────────────────
    if provider == "groq":
        try:
            from langchain_groq import ChatGroq
        except ImportError:
            raise ImportError(
                "langchain-groq not installed. Run: pip install langchain-groq"
            )
        key = api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to .env or pass api_key= directly."
            )
        return ChatGroq(model=resolved_model, temperature=temperature, api_key=key)

    # ── OpenAI ──────────────────────────────────────────────────────
    elif provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError(
                "langchain-openai not installed. Run: pip install langchain-openai"
            )
        key = api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise EnvironmentError(
                "OPENAI_API_KEY not set. Add it to .env or pass api_key= directly."
            )
        return ChatOpenAI(model=resolved_model, temperature=temperature, api_key=key)

    # ── Unknown ─────────────────────────────────────────────────────
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER: '{provider}'. "
            f"Supported values: 'groq', 'openai'."
        )


def get_provider_info() -> dict:
    """Return current provider configuration (safe to log — no keys)."""
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    model = os.getenv("LLM_MODEL") or _PROVIDER_DEFAULTS.get(provider, "unknown")
    return {"provider": provider, "model": model}
