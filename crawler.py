"""ScrapeGraphAI website crawler with 3-account API key fallback.

Crawling runs without an artificial max-page limit across up to 3 configured
ScrapeGraphAI accounts (`SGAI_API_KEY`, `SGAI_API_KEY_2`, `SGAI_API_KEY_3`).
Fallback to the next account occurs only when the active account fails,
hits quota/rate limits, or returns no usable content.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlparse

from config import AppSettings

logger = logging.getLogger(__name__)

_SUBPAGE_PRIORITY_KEYWORDS: tuple[str, ...] = (
    "pricing",
    "price",
    "harga",
    "paket",
    "plan",
    "feature",
    "fitur",
    "product",
    "produk",
    "solution",
    "solusi",
    "service",
    "layanan",
    "about",
    "tentang",
    "company",
    "client",
    "customer",
    "pelanggan",
    "testimoni",
    "review",
    "case-stud",
    "integrat",
    "tech",
    "faq",
)

_LIMIT_ERROR_MARKERS = (
    "402",
    "429",
    "rate limit",
    "ratelimit",
    "credit",
    "quota",
    "limit exceeded",
    "payment required",
    "insufficient",
    "too many requests",
)


@dataclass
class CrawledPage:
    """Normalized representation of a single crawled webpage."""

    url: str
    title: str
    markdown: str


@dataclass
class CrawlResultData:
    """Aggregated crawl output across one website (homepage + subpages)."""

    target_url: str
    engine_used: str  # "scrapegraph_1" | "scrapegraph_2" | "scrapegraph_3" | "failed"
    pages: list[CrawledPage] = field(default_factory=list)
    combined_markdown: str = ""
    fallback_notes: list[str] = field(default_factory=list)
    error_message: str | None = None

    @property
    def success(self) -> bool:
        return bool(self.pages) and bool(self.combined_markdown.strip())


def _score_url_priority(url: str, target_url: str) -> tuple[int, int]:
    """Higher-priority pages sort first: homepage > pricing/features > others."""
    parsed = urlparse(url)
    target_parsed = urlparse(target_url)
    path = (parsed.path or "/").rstrip("/") or "/"
    target_path = (target_parsed.path or "/").rstrip("/") or "/"

    if path == target_path:
        return (0, len(path))

    lower_url = url.lower()
    for idx, kw in enumerate(_SUBPAGE_PRIORITY_KEYWORDS):
        if kw in lower_url:
            return (1 + idx // 6, len(path))
    return (10, len(path))


def aggregate_markdown(
    pages: Iterable[CrawledPage],
    target_url: str,
    max_context_chars: int = 120_000,
) -> tuple[list[CrawledPage], str]:
    """Deduplicate, prioritize, and aggregate crawled pages within a character budget."""
    seen_urls: set[str] = set()
    unique_pages: list[CrawledPage] = []

    for page in pages:
        cleaned_md = (page.markdown or "").strip()
        if not cleaned_md:
            continue
        norm_key = page.url.rstrip("/")
        if norm_key in seen_urls:
            continue
        seen_urls.add(norm_key)
        unique_pages.append(
            CrawledPage(
                url=page.url,
                title=(page.title or "").strip() or page.url,
                markdown=cleaned_md,
            )
        )

    unique_pages.sort(key=lambda p: _score_url_priority(p.url, target_url))

    sections: list[str] = []
    remaining = max_context_chars
    included_pages: list[CrawledPage] = []

    for idx, page in enumerate(unique_pages, start=1):
        if remaining <= 400:
            break
        header = f"=== PAGE {idx}: {page.title} ===\nURL: {page.url}\n\n"
        body_budget = remaining - len(header) - 2
        if body_budget <= 200:
            break

        content = page.markdown
        if len(content) > body_budget:
            content = content[:body_budget].rstrip() + "\n...[truncated due to context budget]"

        section = f"{header}{content}\n\n"
        sections.append(section)
        included_pages.append(page)
        remaining -= len(section)

    return included_pages, "".join(sections).strip()


def _extract_attr_or_key(obj: Any, *names: str, default: Any = None) -> Any:
    """Safely read an attribute or dictionary key supporting snake_case & camelCase."""
    if obj is None:
        return default
    for name in names:
        if isinstance(obj, dict) and name in obj and obj[name] is not None:
            return obj[name]
        if hasattr(obj, name):
            val = getattr(obj, name)
            if val is not None:
                return val
    return default


def _classify_provider_error(provider: str, exc: Exception) -> str:
    """Build a concise human-readable fallback note for provider failures."""
    msg = str(exc)
    lower = msg.lower()
    if any(marker in lower for marker in _LIMIT_ERROR_MARKERS):
        return f"{provider} quota/rate limit reached ({msg[:140]})"
    if "401" in lower or "403" in lower or "unauthorized" in lower or "api key" in lower:
        return f"{provider} API key rejected or unauthorized"
    if "timeout" in lower or "timed out" in lower:
        return f"{provider} request timed out"
    return f"{provider} failed ({msg[:140]})"


def _coerce_markdown_text(value: Any) -> str:
    """Normalize a ScrapeGraphAI markdown payload (`str`, `list[str]`, or nested `dict`) into text."""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = [_coerce_markdown_text(item) for item in value]
        non_empty = [part for part in parts if part]
        return "\n\n".join(non_empty).strip()
    if isinstance(value, dict):
        for key in ("data", "markdown", "content", "text"):
            if key in value:
                extracted = _coerce_markdown_text(value[key])
                if extracted:
                    return extracted
    return ""


def _extract_scrapegraph_markdown(page: Any) -> str:
    """Extract Markdown string from a ScrapeGraphAI v2 `CrawlPage` item."""
    direct_md = _coerce_markdown_text(_extract_attr_or_key(page, "markdown"))
    if direct_md:
        return direct_md

    scrape_obj = _extract_attr_or_key(page, "scrape")
    if scrape_obj is None:
        return ""

    results = _extract_attr_or_key(scrape_obj, "results", default={})
    if not isinstance(results, dict):
        return ""

    return _coerce_markdown_text(results.get("markdown"))


def _crawl_with_scrapegraph(
    target_url: str,
    api_key: str,
    scrapegraph_factory: Callable[[str], Any] | None = None,
    *,
    poll_interval_seconds: float = 2.0,
    max_wait_seconds: float = 180.0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[CrawledPage]:
    """Execute ScrapeGraphAI v2 multi-page crawl without a max_pages limit."""
    if scrapegraph_factory is not None:
        sgai = scrapegraph_factory(api_key)
        formats: list[Any] = [{"type": "markdown", "mode": "normal"}]
    else:
        from scrapegraph_py import MarkdownFormatConfig, ScrapeGraphAI

        sgai = ScrapeGraphAI(api_key=api_key)
        formats = [MarkdownFormatConfig(mode="normal")]

    result = sgai.crawl.start(
        target_url,
        formats=formats,
    )

    if _extract_attr_or_key(result, "status") != "success":
        err = _extract_attr_or_key(result, "error", default="Unknown ScrapeGraphAI crawl error")
        raise RuntimeError(str(err))

    crawl_data = _extract_attr_or_key(result, "data")
    if crawl_data is None:
        return []

    crawl_id = _extract_attr_or_key(crawl_data, "id")
    crawl_status = _extract_attr_or_key(crawl_data, "status", default="completed")

    # Poll async crawl job until completion if status is "running"
    elapsed = 0.0
    while crawl_status == "running" and crawl_id and elapsed < max_wait_seconds:
        sleep_fn(poll_interval_seconds)
        elapsed += poll_interval_seconds
        poll_res = sgai.crawl.get(crawl_id)
        if _extract_attr_or_key(poll_res, "status") != "success":
            err = _extract_attr_or_key(poll_res, "error", default="Failed polling ScrapeGraphAI job")
            raise RuntimeError(str(err))
        crawl_data = _extract_attr_or_key(poll_res, "data")
        crawl_status = _extract_attr_or_key(crawl_data, "status", default="completed")

    if crawl_status in ("failed", "deleted"):
        reason = _extract_attr_or_key(crawl_data, "reason", default=f"Crawl status: {crawl_status}")
        raise RuntimeError(str(reason))

    raw_pages = list(_extract_attr_or_key(crawl_data, "pages", default=[]) or [])

    # Fetch `/crawl/{id}/pages` whenever `pages` is empty, partial, or lacks populated `scrape` content
    # (ScrapeGraphAI's `/crawl/{id}` status endpoint returns `scrape=None` and places results in `/pages`).
    total_count = int(_extract_attr_or_key(crawl_data, "total", default=0) or 0)
    has_embedded_markdown = any(_extract_scrapegraph_markdown(item) for item in raw_pages)
    needs_pages_endpoint = (
        not raw_pages
        or not has_embedded_markdown
        or (total_count > 0 and len(raw_pages) < total_count)
    )

    if crawl_id and hasattr(sgai.crawl, "pages") and needs_pages_endpoint:
        collected: list[Any] = []
        cursor: int | None = None
        while True:
            pages_res = (
                sgai.crawl.pages(crawl_id)
                if cursor is None
                else sgai.crawl.pages(crawl_id, cursor=cursor)
            )
            if _extract_attr_or_key(pages_res, "status") != "success":
                break
            pages_payload = _extract_attr_or_key(pages_res, "data")
            batch = _extract_attr_or_key(pages_payload, "data", "pages", default=[]) or []
            collected.extend(batch)
            pagination = _extract_attr_or_key(pages_payload, "pagination")
            next_cursor = _extract_attr_or_key(pagination, "next_cursor", "nextCursor")
            if next_cursor is None or next_cursor == "":
                break
            cursor = int(next_cursor)
        if collected:
            raw_pages = collected

    pages: list[CrawledPage] = []
    for item in raw_pages:
        markdown = _extract_scrapegraph_markdown(item)
        if not markdown:
            continue
        page_url = str(_extract_attr_or_key(item, "url", default=target_url))
        title = str(_extract_attr_or_key(item, "title", default="") or page_url)
        pages.append(CrawledPage(url=page_url, title=title, markdown=markdown))

    return pages


def crawl_website(
    target_url: str,
    settings: AppSettings,
    *,
    scrapegraph_factory: Callable[[str], Any] | None = None,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> CrawlResultData:
    """Crawl a target website using ScrapeGraphAI with automatic fallback across up to 3 API keys.

    Key fallback order:
    1. `SGAI_API_KEY`
    2. `SGAI_API_KEY_2`
    3. `SGAI_API_KEY_3`
    """
    fallback_notes: list[str] = []
    configured_slots = [
        (slot_num, api_key)
        for slot_num, api_key, is_ok in settings.scrapegraph_slots
        if is_ok
    ]

    if not configured_slots:
        msg = "Tidak ada SGAI_API_KEY yang dikonfigurasi di .env."
        return CrawlResultData(
            target_url=target_url,
            engine_used="failed",
            fallback_notes=[msg],
            error_message=msg,
        )

    # Deduplicate keys if user pasted the same key into multiple slots
    seen_keys: set[str] = set()
    for slot_num, api_key in configured_slots:
        if api_key in seen_keys:
            continue
        seen_keys.add(api_key)

        provider_label = f"ScrapeGraphAI Akun #{slot_num}"
        try:
            sg_pages = _crawl_with_scrapegraph(
                target_url,
                api_key,
                scrapegraph_factory=scrapegraph_factory,
                sleep_fn=sleep_fn,
            )
            included, combined = aggregate_markdown(
                sg_pages, target_url, max_context_chars=settings.max_context_chars
            )
            if included and combined:
                return CrawlResultData(
                    target_url=target_url,
                    engine_used=f"scrapegraph_{slot_num}",
                    pages=included,
                    combined_markdown=combined,
                    fallback_notes=fallback_notes,
                )
            note = f"{provider_label} returned empty crawl data"
            fallback_notes.append(note)
            logger.warning("%s for %s; trying next key if available.", note, target_url)
        except Exception as exc:  # noqa: BLE001 - any SDK/quota/network error falls back to next ScrapeGraphAI key
            note = _classify_provider_error(provider_label, exc)
            fallback_notes.append(note)
            logger.warning("%s for %s (%s); trying next key if available.", provider_label, target_url, exc)

    return CrawlResultData(
        target_url=target_url,
        engine_used="failed",
        fallback_notes=fallback_notes,
        error_message="; ".join(fallback_notes) or "All ScrapeGraphAI accounts failed.",
    )
