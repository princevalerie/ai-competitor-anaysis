"""Application configuration loader and URL normalization helpers.

API keys are loaded strictly from environment variables / `.env` file so the
Streamlit UI never asks for or persists credentials.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = PROJECT_ROOT / ".env"

DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_GEMINI_FALLBACK_MODEL = "gemini-3.5-flash-lite"
DEFAULT_OPENROUTER_MODEL = "openrouter/free"
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MAX_CONTEXT_CHARS = 120_000

_PLACEHOLDER_PREFIXES = ("your_", "<your", "changeme", "replace_")


def _is_configured_secret(value: str | None) -> bool:
    """Return True if an API key is non-empty and not an example placeholder."""
    if not value:
        return False
    cleaned = value.strip()
    if not cleaned:
        return False
    lowered = cleaned.lower()
    if any(lowered.startswith(prefix) for prefix in _PLACEHOLDER_PREFIXES):
        return False
    return not lowered.endswith("_here")


def _parse_positive_int(raw: str | None, default: int, minimum: int = 1, maximum: int = 100) -> int:
    """Safely parse an integer environment variable within bounds."""
    if not raw:
        return default
    try:
        parsed = int(raw.strip())
    except ValueError:
        return default
    return max(minimum, min(maximum, parsed))


@dataclass(frozen=True)
class AppSettings:
    """Immutable application settings loaded from `.env` and environment variables."""

    gemini_api_key: str = ""
    scrapegraph_api_key: str = ""
    scrapegraph_api_key_2: str = ""
    scrapegraph_api_key_3: str = ""
    openrouter_api_key: str = ""
    gemini_model: str = DEFAULT_GEMINI_MODEL
    gemini_fallback_model: str = DEFAULT_GEMINI_FALLBACK_MODEL
    openrouter_model: str = DEFAULT_OPENROUTER_MODEL
    max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS

    @property
    def has_gemini_key(self) -> bool:
        return _is_configured_secret(self.gemini_api_key)

    @property
    def has_openrouter_key(self) -> bool:
        return _is_configured_secret(self.openrouter_api_key)

    @property
    def has_ai_provider(self) -> bool:
        return self.has_gemini_key or self.has_openrouter_key

    @property
    def scrapegraph_keys(self) -> list[str]:
        """Return configured ScrapeGraphAI API keys in fallback priority order (deduplicated)."""
        ordered: list[str] = []
        for raw_key in (
            self.scrapegraph_api_key,
            self.scrapegraph_api_key_2,
            self.scrapegraph_api_key_3,
        ):
            if _is_configured_secret(raw_key):
                cleaned = raw_key.strip()
                if cleaned not in ordered:
                    ordered.append(cleaned)
        return ordered

    @property
    def scrapegraph_slots(self) -> list[tuple[int, str, bool]]:
        """Return `(slot_number, key_or_empty, is_configured)` for all 3 ScrapeGraphAI slots."""
        raw_slots = (
            self.scrapegraph_api_key,
            self.scrapegraph_api_key_2,
            self.scrapegraph_api_key_3,
        )
        return [(idx, val.strip(), _is_configured_secret(val)) for idx, val in enumerate(raw_slots, start=1)]

    @property
    def has_scrapegraph_key(self) -> bool:
        return bool(self.scrapegraph_keys)


def load_settings(env_path: Path | None = None, *, reload_env: bool = True) -> AppSettings:
    """Load settings from `.env` and OS environment variables."""
    target_env = env_path or DEFAULT_ENV_PATH
    if reload_env and target_env.exists():
        load_dotenv(dotenv_path=target_env, override=True)

    raw_gemini = os.getenv("GEMINI_API_KEY", "").strip()
    raw_openrouter = os.getenv("OPENROUTER_API_KEY", "").strip()
    raw_sgai_1 = (
        os.getenv("SGAI_API_KEY")
        or os.getenv("SGAI_API_KEY_1")
        or os.getenv("SCRAPEGRAPH_API_KEY")
        or ""
    ).strip()
    raw_sgai_2 = (os.getenv("SGAI_API_KEY_2") or os.getenv("SCRAPEGRAPH_API_KEY_2") or "").strip()
    raw_sgai_3 = (os.getenv("SGAI_API_KEY_3") or os.getenv("SCRAPEGRAPH_API_KEY_3") or "").strip()

    gemini_model = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    gemini_fallback = (
        os.getenv("GEMINI_FALLBACK_MODEL", DEFAULT_GEMINI_FALLBACK_MODEL).strip()
        or DEFAULT_GEMINI_FALLBACK_MODEL
    )
    openrouter_model = (
        os.getenv("OPENROUTER_MODEL", DEFAULT_OPENROUTER_MODEL).strip() or DEFAULT_OPENROUTER_MODEL
    )

    return AppSettings(
        gemini_api_key=raw_gemini if _is_configured_secret(raw_gemini) else "",
        scrapegraph_api_key=raw_sgai_1 if _is_configured_secret(raw_sgai_1) else "",
        scrapegraph_api_key_2=raw_sgai_2 if _is_configured_secret(raw_sgai_2) else "",
        scrapegraph_api_key_3=raw_sgai_3 if _is_configured_secret(raw_sgai_3) else "",
        openrouter_api_key=raw_openrouter if _is_configured_secret(raw_openrouter) else "",
        gemini_model=gemini_model,
        gemini_fallback_model=gemini_fallback,
        openrouter_model=openrouter_model,
        max_context_chars=_parse_positive_int(
            os.getenv("MAX_CONTEXT_CHARS"), DEFAULT_MAX_CONTEXT_CHARS, minimum=5_000, maximum=400_000
        ),
    )


def normalize_url(raw_url: str) -> str:
    """Normalize a user-supplied website URL (e.g. `www.collabox.id` -> `https://www.collabox.id`).

    Raises:
        ValueError: If the URL is empty, uses an unsupported scheme, or lacks a valid host.
    """
    cleaned = (raw_url or "").strip()
    if not cleaned:
        raise ValueError("URL cannot be empty.")

    if "://" not in cleaned:
        cleaned = f"https://{cleaned}"

    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    if scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme '{scheme}'. Use http or https.")

    netloc = parsed.netloc.strip()
    if not netloc or " " in netloc or ("." not in netloc and not netloc.startswith("localhost")):
        raise ValueError(f"Invalid website URL: '{raw_url}'")

    return urlunparse((scheme, netloc.lower(), parsed.path or "", parsed.params, parsed.query, ""))
