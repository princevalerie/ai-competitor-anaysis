"""Gemini AI extractor and strategic analyzer with OpenRouter fallback.

Primary AI uses the modern `google-genai` SDK (`gemini-3.8-flash` with fallback
to `gemini-3.5-flash-lite` on rate-limit errors). When Gemini fails or hits quota
limits, analysis automatically falls back to OpenRouter (`openrouter/free`) with
reasoning enabled, and builds the competitor comparison DataFrame
deterministically without wasting an extra LLM call.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable
from typing import Any

import pandas as pd
import requests
from google import genai
from google.genai import types
from google.genai.errors import APIError
from pydantic import BaseModel, Field, ValidationError

from config import DEFAULT_OPENROUTER_URL, AppSettings
from crawler import CrawlResultData

logger = logging.getLogger(__name__)

_RETRYABLE_ERROR_MARKERS = ("429", "resource_exhausted", "503", "unavailable", "rate limit")

_ENGINE_LABELS = {
    "scrapegraph": "ScrapeGraphAI",
    "scrapegraph_1": "ScrapeGraphAI (Akun #1)",
    "scrapegraph_2": "ScrapeGraphAI (Akun #2)",
    "scrapegraph_3": "ScrapeGraphAI (Akun #3)",
}


class CompetitorDataSchema(BaseModel):
    """Structured intelligence profile extracted from crawled website Markdown."""

    company_name: str = Field(description="Official company or product brand name.")
    value_proposition: str = Field(
        description="Concise 1-2 sentence summary of core value proposition and target audience."
    )
    pricing: str = Field(
        description="Pricing tiers, plans, free trial availability, or 'Contact sales / N/A' if unlisted."
    )
    key_features: list[str] = Field(
        description="Top 3 to 5 key product or service features."
    )
    tech_stack: list[str] = Field(
        description="Notable technologies, integrations, APIs, or platforms mentioned (or ['N/A'])."
    )
    marketing_focus: str = Field(
        description="Primary positioning, messaging angle, or target industry segment."
    )
    customer_feedback: str = Field(
        description="Summary or brief quotes of testimonials, client logos, or social proof (or 'N/A')."
    )


def create_gemini_client(api_key: str) -> genai.Client | None:
    """Create a configured `google.genai.Client` instance if an API key is provided."""
    cleaned = (api_key or "").strip()
    if not cleaned:
        return None
    return genai.Client(api_key=cleaned)


def _is_retryable_gemini_error(exc: Exception) -> bool:
    lower = str(exc).lower()
    return any(marker in lower for marker in _RETRYABLE_ERROR_MARKERS)


def _build_model_candidates(settings: AppSettings) -> list[str]:
    models = [settings.gemini_model]
    if settings.gemini_fallback_model and settings.gemini_fallback_model not in models:
        models.append(settings.gemini_fallback_model)
    return models


def _generate_with_resilience(
    client: Any,
    settings: AppSettings,
    contents: str,
    config: types.GenerateContentConfig,
    *,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> Any:
    """Call Gemini with short backoff retries and fallback to the lighter Free Tier model."""
    last_error: Exception | None = None
    candidates = _build_model_candidates(settings)

    for model_name in candidates:
        for attempt in range(2):
            try:
                return client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config,
                )
            except (APIError, RuntimeError, TimeoutError, ConnectionError, ValueError) as exc:
                last_error = exc
                if not _is_retryable_gemini_error(exc):
                    raise
                logger.warning(
                    "Gemini call failed on model=%s (attempt=%d): %s",
                    model_name,
                    attempt + 1,
                    exc,
                )
                if attempt == 0:
                    sleep_fn(2.0)

    if last_error is not None:
        raise last_error
    raise RuntimeError("No Gemini model candidates configured.")


def _post_openrouter_chat(
    settings: AppSettings,
    messages: list[dict[str, Any]],
    *,
    http_session: requests.Session | None = None,
    timeout_seconds: int = 90,
) -> dict[str, Any]:
    """Send a chat completion request to OpenRouter with reasoning enabled."""
    if not settings.has_openrouter_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured in .env.")

    payload = {
        "model": settings.openrouter_model,
        "messages": messages,
        "reasoning": {"enabled": True},
    }
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
    }
    poster = http_session.post if http_session is not None else requests.post
    response = poster(
        url=DEFAULT_OPENROUTER_URL,
        headers=headers,
        data=json.dumps(payload),
        timeout=timeout_seconds,
    )
    if getattr(response, "status_code", 200) >= 400:
        detail = (getattr(response, "text", "") or "")[:200]
        raise RuntimeError(f"OpenRouter HTTP {response.status_code}: {detail}")

    body = response.json()
    choices = body.get("choices") or []
    if not choices or not isinstance(choices[0], dict):
        raise RuntimeError(f"OpenRouter returned invalid response payload: {body}")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise TypeError("OpenRouter choice missing 'message' object.")
    return message


def _strip_json_fences(raw_text: str) -> str:
    cleaned = (raw_text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").strip()
        if cleaned.lower().startswith("json"):
            cleaned = cleaned[4:].strip()
    return cleaned


def _parse_json_object(raw_text: str) -> dict[str, Any]:
    """Parse a JSON object from model text, tolerating Markdown code fences or surrounding prose."""
    cleaned = _strip_json_fences(raw_text)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end > start:
        parsed = json.loads(cleaned[start : end + 1])
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Model response did not contain a valid JSON object.")


def _call_openrouter_structured(
    settings: AppSettings,
    prompt: str,
    *,
    http_session: requests.Session | None = None,
) -> CompetitorDataSchema:
    """Extract `CompetitorDataSchema` via OpenRouter, preserving reasoning across a retry turn if needed."""
    schema_hint = json.dumps(CompetitorDataSchema.model_json_schema(), indent=2)
    structured_prompt = (
        f"{prompt}\n\n"
        "Return ONLY a valid JSON object matching this exact JSON Schema (no Markdown fences or extra prose):\n"
        f"{schema_hint}"
    )

    messages: list[dict[str, Any]] = [{"role": "user", "content": structured_prompt}]
    assistant_msg = _post_openrouter_chat(settings, messages, http_session=http_session)
    content = (assistant_msg.get("content") or "").strip()

    try:
        return CompetitorDataSchema.model_validate(_parse_json_object(content))
    except (ValueError, ValidationError) as first_err:
        logger.warning(
            "OpenRouter initial structured response invalid (%s); continuing reasoning turn.",
            first_err,
        )
        preserved_assistant: dict[str, Any] = {
            "role": "assistant",
            "content": assistant_msg.get("content"),
        }
        if "reasoning_details" in assistant_msg:
            preserved_assistant["reasoning_details"] = assistant_msg.get("reasoning_details")

        follow_up_messages = [
            *messages,
            preserved_assistant,
            {
                "role": "user",
                "content": (
                    "Your previous answer was not valid JSON matching the required schema. "
                    "Respond with ONLY a single raw JSON object containing keys: "
                    "company_name, value_proposition, pricing, key_features (array of strings), "
                    "tech_stack (array of strings), marketing_focus, customer_feedback."
                ),
            },
        ]
        second_msg = _post_openrouter_chat(settings, follow_up_messages, http_session=http_session)
        second_content = (second_msg.get("content") or "").strip()
        return CompetitorDataSchema.model_validate(_parse_json_object(second_content))


def _call_openrouter_text(
    settings: AppSettings,
    prompt: str,
    *,
    http_session: requests.Session | None = None,
) -> str:
    """Generate free-form Markdown analysis via OpenRouter with reasoning enabled."""
    messages = [{"role": "user", "content": prompt}]
    assistant_msg = _post_openrouter_chat(settings, messages, http_session=http_session)
    content = (assistant_msg.get("content") or "").strip()
    if not content:
        raise RuntimeError("OpenRouter returned empty content.")
    return content


def _language_instruction(output_language: str) -> str:
    if output_language.lower().startswith("id"):
        return "Write all extracted descriptions and analysis in clear, natural Bahasa Indonesia."
    return "Write all extracted descriptions and analysis in clear, professional English."


def _normalize_list_field(values: list[str] | None, limit: int) -> list[str]:
    cleaned = [str(item).strip() for item in (values or []) if str(item).strip()]
    return cleaned[:limit] if cleaned else ["N/A"]


def extract_structured_profile(
    client: Any,
    settings: AppSettings,
    crawl_data: CrawlResultData,
    *,
    output_language: str = "id",
    http_session: requests.Session | None = None,
) -> dict[str, Any]:
    """Extract structured competitor data via Gemini with automatic OpenRouter fallback."""
    lang_rule = _language_instruction(output_language)
    prompt = f"""
You are a senior competitive intelligence analyst.
Extract structured company intelligence strictly from the crawled website pages below.
Do not invent pricing numbers, features, or testimonials not supported by the pages.
If a field is not present in the content, use "N/A" (or ["N/A"] for list fields).
{lang_rule}

Target Website URL: {crawl_data.target_url}
Crawled Pages Count: {len(crawl_data.pages)}

Website Content (Markdown):
{crawl_data.combined_markdown}
""".strip()

    schema: CompetitorDataSchema | None = None
    ai_provider = "gemini"
    ai_fallback_note: str | None = None

    if client is not None and settings.has_gemini_key:
        gen_config = types.GenerateContentConfig(
            temperature=0.2,
            response_mime_type="application/json",
            response_schema=CompetitorDataSchema,
        )
        try:
            response = _generate_with_resilience(client, settings, prompt, gen_config)
            parsed_obj = getattr(response, "parsed", None)
            if isinstance(parsed_obj, CompetitorDataSchema):
                schema = parsed_obj
            elif isinstance(parsed_obj, dict):
                schema = CompetitorDataSchema.model_validate(parsed_obj)
            else:
                raw_text = _strip_json_fences(getattr(response, "text", "") or "")
                schema = CompetitorDataSchema.model_validate(json.loads(raw_text))
        except (APIError, RuntimeError, TimeoutError, ConnectionError, ValueError, OSError) as exc:
            if not settings.has_openrouter_key:
                raise
            ai_fallback_note = f"Gemini failed ({str(exc)[:120]}); fallback to OpenRouter ({settings.openrouter_model})"
            logger.warning("%s", ai_fallback_note)
    elif not settings.has_openrouter_key:
        raise RuntimeError("Neither GEMINI_API_KEY nor OPENROUTER_API_KEY is configured in .env.")
    else:
        ai_fallback_note = f"GEMINI_API_KEY not set; using OpenRouter ({settings.openrouter_model})"

    if schema is None:
        schema = _call_openrouter_structured(settings, prompt, http_session=http_session)
        ai_provider = "openrouter"

    return {
        "url": crawl_data.target_url,
        "company_name": schema.company_name.strip() or crawl_data.target_url,
        "value_proposition": schema.value_proposition.strip() or "N/A",
        "pricing": schema.pricing.strip() or "N/A",
        "key_features": _normalize_list_field(schema.key_features, limit=5),
        "tech_stack": _normalize_list_field(schema.tech_stack, limit=5),
        "marketing_focus": schema.marketing_focus.strip() or "N/A",
        "customer_feedback": schema.customer_feedback.strip() or "N/A",
        "engine_used": crawl_data.engine_used,
        "ai_provider": ai_provider,
        "ai_fallback_note": ai_fallback_note,
        "fallback_notes": crawl_data.fallback_notes,
        "pages_crawled": [page.url for page in crawl_data.pages],
        "pages_count": len(crawl_data.pages),
    }


def build_comparison_dataframe(
    company_profile: dict[str, Any],
    competitor_profiles: list[dict[str, Any]],
    *,
    output_language: str = "id",
) -> pd.DataFrame:
    """Build a structured comparison DataFrame directly from extracted profiles (0 LLM calls)."""
    valid_competitors = [c for c in competitor_profiles if isinstance(c, dict) and c]
    all_profiles = [company_profile] + valid_competitors
    is_id = output_language.lower().startswith("id")

    rows = []
    for idx, item in enumerate(all_profiles):
        role_label = (
            ("Perusahaan Anda" if is_id else "Your Company")
            if idx == 0
            else (f"Kompetitor #{idx}" if is_id else f"Competitor #{idx}")
        )
        engine_label = _ENGINE_LABELS.get(item.get("engine_used", ""), "Unknown")
        features = "; ".join(item.get("key_features", [])[:3]) or "N/A"
        tech = ", ".join(item.get("tech_stack", [])[:3]) or "N/A"
        pages_count = item.get("pages_count", 0)

        if is_id:
            rows.append(
                {
                    "Peran": role_label,
                    "Perusahaan": f"{item.get('company_name', 'N/A')} ({item.get('url', '')})",
                    "Value Proposition": item.get("value_proposition", "N/A"),
                    "Harga & Paket": item.get("pricing", "N/A"),
                    "Fitur Utama": features,
                    "Tech Stack": tech,
                    "Fokus Pemasaran": item.get("marketing_focus", "N/A"),
                    "Umpan Balik Pelanggan": item.get("customer_feedback", "N/A"),
                    "Sumber Crawl": f"{engine_label} ({pages_count} hal.)",
                }
            )
        else:
            rows.append(
                {
                    "Role": role_label,
                    "Company": f"{item.get('company_name', 'N/A')} ({item.get('url', '')})",
                    "Value Proposition": item.get("value_proposition", "N/A"),
                    "Pricing": item.get("pricing", "N/A"),
                    "Key Features": features,
                    "Tech Stack": tech,
                    "Marketing Focus": item.get("marketing_focus", "N/A"),
                    "Customer Feedback": item.get("customer_feedback", "N/A"),
                    "Crawl Source": f"{engine_label} ({pages_count} pages)",
                }
            )

    return pd.DataFrame(rows)


def generate_strategic_analysis(
    client: Any,
    settings: AppSettings,
    company_profile: dict[str, Any],
    competitor_profiles: list[dict[str, Any]],
    *,
    output_language: str = "id",
    http_session: requests.Session | None = None,
) -> str:
    """Generate actionable competitive strategy recommendations via Gemini with OpenRouter fallback."""
    valid_competitors = [c for c in competitor_profiles if isinstance(c, dict) and c]
    if not valid_competitors:
        raise ValueError("At least one valid competitor profile is required for strategic analysis.")

    lang_rule = _language_instruction(output_language)

    # Strip internal crawl metadata before sending to LLM to minimize token usage
    def _compact_profile(profile: dict[str, Any]) -> dict[str, Any]:
        return {
            "url": profile.get("url"),
            "company_name": profile.get("company_name"),
            "value_proposition": profile.get("value_proposition"),
            "pricing": profile.get("pricing"),
            "key_features": profile.get("key_features"),
            "tech_stack": profile.get("tech_stack"),
            "marketing_focus": profile.get("marketing_focus"),
            "customer_feedback": profile.get("customer_feedback"),
        }

    company_json = json.dumps(_compact_profile(company_profile), indent=2, ensure_ascii=False)
    competitors_json = json.dumps(
        [_compact_profile(c) for c in valid_competitors], indent=2, ensure_ascii=False
    )

    prompt = f"""
You are a senior product & growth strategist advising the user's company based on verified competitor data.
{lang_rule}
Format the response cleanly with Markdown headings and bullet points. Ground every insight in the data below.

Your Company Data:
{company_json}

Competitor Data:
{competitors_json}

Required Sections:
1. Executive Positioning Summary (how your company compares right now)
2. Market Gaps & Unserved Needs
3. Competitor Weaknesses & Blind Spots
4. Feature & Tech Stack Differentiation Opportunities
5. Pricing & Packaging Recommendations
6. Prioritized 30-60-90 Day Growth Action Plan
""".strip()

    if client is not None and settings.has_gemini_key:
        gen_config = types.GenerateContentConfig(temperature=0.3)
        try:
            response = _generate_with_resilience(client, settings, prompt, gen_config)
            text = (getattr(response, "text", "") or "").strip()
            if text:
                return text
            raise RuntimeError("Gemini returned empty strategic analysis.")
        except (APIError, RuntimeError, TimeoutError, ConnectionError, ValueError, OSError) as exc:
            if not settings.has_openrouter_key:
                raise
            logger.warning(
                "Gemini strategic analysis failed (%s); falling back to OpenRouter (%s).",
                exc,
                settings.openrouter_model,
            )
    elif not settings.has_openrouter_key:
        raise RuntimeError("Neither GEMINI_API_KEY nor OPENROUTER_API_KEY is configured in .env.")

    return _call_openrouter_text(settings, prompt, http_session=http_session)
