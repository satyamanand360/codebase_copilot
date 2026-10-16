import os
from typing import Literal

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Model names can be overridden in .env without touching code.
GEMINI_MAIN = os.environ.get("GEMINI_MAIN_MODEL", "gemini-3.1-flash-lite")
GEMINI_FAST = os.environ.get("GEMINI_FAST_MODEL", GEMINI_MAIN)
CLAUDE_MAIN = os.environ.get("CLAUDE_MAIN_MODEL", "claude-sonnet-5-5")
CLAUDE_FAST = os.environ.get("CLAUDE_FAST_MODEL", "claude-haiku-5-5")
RETRIES = 3
PLACEHOLDERS = {"your_google_api_key_here", "your_anthropic_api_key_here", ""}


def _key(name: str):
    value = os.environ.get(name)
    return None if value is None or value in PLACEHOLDERS else value


def get_llm(model_class: Literal["main", "fast"]):
    """Gemini as primary, Claude as fallback.

    Each client retries transient errors (rate limits, 5xx) itself via
    max_retries. If the primary still fails, ``with_fallbacks`` re-sends the
    same input to the fallback provider. Returns None when no key is set, and
    the agents report that instead of crashing.
    """
    google_key, anthropic_key = _key("GOOGLE_API_KEY"), _key("ANTHROPIC_API_KEY")

    primary = None
    if google_key:
        primary = ChatGoogleGenerativeAI(
            model=GEMINI_MAIN if model_class == "main" else GEMINI_FAST,
            temperature=0, max_retries=RETRIES, api_key=google_key,
        )
    fallback = None
    if anthropic_key:
        fallback = ChatAnthropic(
            model=CLAUDE_MAIN if model_class == "main" else CLAUDE_FAST,
            temperature=0, max_retries=RETRIES, api_key=anthropic_key,
        )

    if primary and fallback:
        return primary.with_fallbacks([fallback])
    return primary or fallback
