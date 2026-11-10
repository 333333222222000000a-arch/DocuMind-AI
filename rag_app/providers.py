"""Optional Groq and OpenAI provider adapters."""

import os
import re

from .config import MAX_OUTPUT_TOKENS

try:
    from langchain_groq import ChatGroq
    GROQ_AVAILABLE = True
except ImportError:
    ChatGroq = None
    GROQ_AVAILABLE = False

try:
    from langchain_community.chat_models import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    ChatOpenAI = None
    OPENAI_AVAILABLE = False


def get_api_key(key_name, environ=None):
    """Read an API key from the environment, then optional Streamlit secrets.

    Accessing ``st.secrets`` can raise when no secrets file or configured
    secret store exists, so that case is treated as an unset key.
    """
    environment = os.environ if environ is None else environ
    key = environment.get(key_name)
    if key:
        return key

    try:
        import streamlit as st
        return st.secrets.get(key_name)
    except Exception:
        # Streamlit raises when secrets.toml is absent or invalid/unavailable.
        return None


def get_groq_llm_instance(api_key, model_name):
    if not GROQ_AVAILABLE:
        raise RuntimeError("Groq integration is unavailable")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not configured")
    return ChatGroq(groq_api_key=api_key, model_name=model_name,
                    temperature=0.0, max_tokens=MAX_OUTPUT_TOKENS,
                    max_retries=2)


def get_openai_llm_instance(api_key, model_name="gpt-3.5-turbo"):
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI integration is unavailable")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not configured")
    return ChatOpenAI(model_name=model_name, temperature=0.0, openai_api_key=api_key)


def provider_error_summary(error):
    """Return actionable provider diagnostics without exposing raw response data."""
    status = provider_error_status(error)
    response = getattr(error, "response", None)

    body = getattr(error, "body", None)
    if body is None and response is not None:
        try:
            body = response.json()
        except Exception:
            body = None
    details = body.get("error", body) if isinstance(body, dict) else {}
    code = details.get("code") if isinstance(details, dict) else None
    provider_message = details.get("message", "") if isinstance(details, dict) else ""

    if status == 413:
        usage = re.search(r"Limit\s+([\d,]+).*?Requested\s+([\d,]+)", provider_message, re.IGNORECASE)
        is_tpm_error = code == "rate_limit_exceeded" or "tokens per minute" in provider_message.lower()
        if is_tpm_error and usage:
            limit, requested = (value.replace(",", "") for value in usage.groups())
            return f"Provider token limit exceeded (TPM limit {limit} tokens/minute; request estimated at {requested} tokens). Context is capped locally; retry after the current rate window."
        return "Provider returned HTTP 413 (request payload too large). Context is capped locally; retry with a shorter question if it persists."
    if status == 429:
        return "The provider rate limit or quota was reached. Wait briefly or check the account limits."
    if status in (401, 403):
        return "The provider rejected the configured API key. Check the provider key configuration."
    if status == 404:
        return "The selected provider model is unavailable. Check the configured model ID."
    if type(error).__name__ in {"APIConnectionError", "ConnectError", "TimeoutException", "APITimeoutError"}:
        return "Could not connect to the provider. Check network access and retry."
    if status is not None:
        return f"Provider request failed with HTTP {status} ({type(error).__name__}). Check provider configuration and retry."
    return f"Provider request failed ({type(error).__name__}). Check provider configuration and retry."


def provider_error_status(error):
    """Read an HTTP status without formatting or exposing provider response data."""
    response = getattr(error, "response", None)
    return getattr(error, "status_code", None) or getattr(response, "status_code", None)
