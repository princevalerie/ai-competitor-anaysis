"""Unit tests for config, 3-key ScrapeGraphAI crawler, analyzer (Gemini + OpenRouter), and UI."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from analyzer import (
    CompetitorDataSchema,
    build_comparison_dataframe,
    extract_structured_profile,
    generate_strategic_analysis,
)
from config import AppSettings, load_settings, normalize_url
from crawler import CrawledPage, CrawlResultData, aggregate_markdown, crawl_website

# ---------------------------------------------------------------------------
# Config & URL normalization tests
# ---------------------------------------------------------------------------


def test_normalize_url_adds_https_and_cleans_fragment() -> None:
    assert normalize_url("www.collabox.id") == "https://www.collabox.id"
    assert normalize_url("https://Collabox.ID/pricing/#pro") == "https://collabox.id/pricing/"
    assert normalize_url("http://localhost:8501") == "http://localhost:8501"


@pytest.mark.parametrize("bad_url", ["", "   ", "ftp://example.com", "not a domain", "javascript:alert(1)"])
def test_normalize_url_rejects_invalid_inputs(bad_url: str) -> None:
    with pytest.raises(ValueError):
        normalize_url(bad_url)


def test_load_settings_reads_three_sgai_keys_and_openrouter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in (
        "GEMINI_API_KEY",
        "OPENROUTER_API_KEY",
        "OPENROUTER_MODEL",
        "SGAI_API_KEY",
        "SGAI_API_KEY_1",
        "SGAI_API_KEY_2",
        "SGAI_API_KEY_3",
        "SCRAPEGRAPH_API_KEY",
        "SCRAPEGRAPH_API_KEY_2",
        "SCRAPEGRAPH_API_KEY_3",
        "GEMINI_MODEL",
        "GEMINI_FALLBACK_MODEL",
    ):
        monkeypatch.delenv(key, raising=False)

    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=your_gemini_api_key_here\n"
        "OPENROUTER_API_KEY=sk-or-v1-test-key\n"
        "SGAI_API_KEY=sgai-key-1\n"
        "SGAI_API_KEY_2=sgai-key-2\n"
        "SGAI_API_KEY_3=your_scrapegraph_api_key_3_here\n",
        encoding="utf-8",
    )

    settings = load_settings(env_path=env_file, reload_env=True)
    assert settings.has_gemini_key is False
    assert settings.has_openrouter_key is True
    assert settings.openrouter_api_key == "sk-or-v1-test-key"
    assert settings.openrouter_model == "openrouter/free"
    assert settings.has_ai_provider is True
    assert settings.has_scrapegraph_key is True
    assert settings.scrapegraph_keys == ["sgai-key-1", "sgai-key-2"]


# ---------------------------------------------------------------------------
# ScrapeGraphAI Crawler & 3-Key Fallback tests
# ---------------------------------------------------------------------------


def test_crawl_website_uses_pages_endpoint_and_list_markdown_without_fallback() -> None:
    """Verify real SDK shape: `/crawl/{id}` returns `scrape=None` and `/pages` returns `list[str]`."""
    captured: dict[str, Any] = {}
    used_keys: list[str] = []

    class FakeCrawlResource:
        def start(self, url: str, **kwargs: Any) -> Any:
            captured["url"] = url
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                status="success",
                error=None,
                data=SimpleNamespace(
                    id="job-real-shape",
                    status="running",
                    total=2,
                    finished=0,
                    pages=[],
                ),
            )

        def get(self, job_id: str) -> Any:
            # Simulates real API where `/crawl/{id}` omits `scrape` payloads
            return SimpleNamespace(
                status="success",
                error=None,
                data=SimpleNamespace(
                    id=job_id,
                    status="completed",
                    total=2,
                    finished=2,
                    pages=[
                        SimpleNamespace(url="https://www.collabox.id/", title="Home", scrape=None),
                        SimpleNamespace(url="https://www.collabox.id/pricing", title="Pricing", scrape=None),
                    ],
                ),
            )

        def pages(self, job_id: str, cursor: int | None = None) -> Any:
            return SimpleNamespace(
                status="success",
                data=SimpleNamespace(
                    data=[
                        SimpleNamespace(
                            url="https://www.collabox.id/pricing",
                            title="Pricing",
                            scrape=SimpleNamespace(results={"markdown": {"data": ["Rp 199.000/mo"]}}),
                        ),
                        SimpleNamespace(
                            url="https://www.collabox.id/",
                            title="Collabox Home",
                            scrape=SimpleNamespace(results={"markdown": {"data": ["# Collabox AI", "Workspace"]}}),
                        ),
                    ],
                    pagination=SimpleNamespace(next_cursor=None),
                ),
            )

    def factory(key: str) -> Any:
        used_keys.append(key)
        return SimpleNamespace(crawl=FakeCrawlResource())

    settings = AppSettings(
        gemini_api_key="gem-key",
        scrapegraph_api_key="sgai-key-1",
        scrapegraph_api_key_2="sgai-key-2",
        scrapegraph_api_key_3="sgai-key-3",
    )

    result = crawl_website(
        "https://www.collabox.id",
        settings,
        scrapegraph_factory=factory,
        sleep_fn=lambda _: None,
    )

    assert result.success is True
    assert result.engine_used == "scrapegraph_1"
    assert result.fallback_notes == []
    assert used_keys == ["sgai-key-1"]
    assert captured["url"] == "https://www.collabox.id"
    assert "max_pages" not in captured["kwargs"]
    assert [p.url for p in result.pages] == [
        "https://www.collabox.id/",
        "https://www.collabox.id/pricing",
    ]
    assert "Workspace" in result.combined_markdown


def test_crawl_website_falls_back_across_three_scrapegraph_accounts() -> None:
    attempted_keys: list[str] = []

    class MultiKeyCrawlResource:
        def __init__(self, key: str) -> None:
            self.key = key

        def start(self, url: str, **kwargs: Any) -> Any:
            if self.key == "sgai-key-1":
                raise RuntimeError("402 Payment Required: insufficient credits")
            if self.key == "sgai-key-2":
                return SimpleNamespace(status="error", data=None, error="429 rate limit exceeded")
            return SimpleNamespace(
                status="success",
                error=None,
                data=SimpleNamespace(
                    id="job-3",
                    status="completed",
                    total=1,
                    finished=1,
                    pages=[
                        SimpleNamespace(
                            url="https://www.collabox.id",
                            title="Home",
                            scrape=SimpleNamespace(results={"markdown": {"data": ["# Success via Key 3"]}}),
                        )
                    ],
                ),
            )

    def factory(key: str) -> Any:
        attempted_keys.append(key)
        return SimpleNamespace(crawl=MultiKeyCrawlResource(key))

    settings = AppSettings(
        gemini_api_key="gem-key",
        scrapegraph_api_key="sgai-key-1",
        scrapegraph_api_key_2="sgai-key-2",
        scrapegraph_api_key_3="sgai-key-3",
    )

    result = crawl_website(
        "https://www.collabox.id",
        settings,
        scrapegraph_factory=factory,
        sleep_fn=lambda _: None,
    )

    assert attempted_keys == ["sgai-key-1", "sgai-key-2", "sgai-key-3"]
    assert result.success is True
    assert result.engine_used == "scrapegraph_3"
    assert len(result.fallback_notes) == 2
    assert "akun #1" in result.fallback_notes[0].lower()
    assert "akun #2" in result.fallback_notes[1].lower()


def test_aggregate_markdown_enforces_budget() -> None:
    pages = [
        CrawledPage(url="https://example.com", title="Home", markdown="A" * 4000),
        CrawledPage(url="https://example.com/pricing", title="Pricing", markdown="B" * 4000),
    ]
    included, combined = aggregate_markdown(pages, "https://example.com", max_context_chars=5200)
    assert len(included) == 2
    assert len(combined) <= 5250
    assert "truncated due to context budget" in combined


# ---------------------------------------------------------------------------
# Analyzer & Gemini + OpenRouter resilience tests
# ---------------------------------------------------------------------------


def test_extract_structured_profile_retries_and_falls_back_to_lite_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("analyzer.time.sleep", lambda _: None)
    calls: list[str] = []

    class FakeModels:
        def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
            calls.append(model)
            if model == "gemini-3.8-flash":
                raise RuntimeError("429 RESOURCE_EXHAUSTED: Quota exceeded")
            schema = CompetitorDataSchema(
                company_name="Collabox",
                value_proposition="All-in-one collaboration suite",
                pricing="Rp 199.000/month",
                key_features=["Task Board", "Docs", "AI Assistant"],
                tech_stack=["React", "Python"],
                marketing_focus="Indonesian SMEs & agencies",
                customer_feedback="Saves 5 hours/week",
            )
            return SimpleNamespace(parsed=schema, text=schema.model_dump_json())

    fake_client = SimpleNamespace(models=FakeModels())
    settings = AppSettings(
        gemini_api_key="gem-key",
        scrapegraph_api_key="sgai-key-1",
        gemini_model="gemini-3.8-flash",
        gemini_fallback_model="gemini-3.5-flash-lite",
    )
    crawl_data = CrawlResultData(
        target_url="https://www.collabox.id",
        engine_used="scrapegraph_1",
        pages=[CrawledPage(url="https://www.collabox.id", title="Home", markdown="# Collabox")],
        combined_markdown="# Collabox",
    )

    profile = extract_structured_profile(fake_client, settings, crawl_data, output_language="id")
    assert calls == ["gemini-3.8-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite"]
    assert profile["company_name"] == "Collabox"
    assert profile["tech_stack"] == ["React", "Python"]
    assert profile["pages_count"] == 1


def test_build_comparison_dataframe_and_strategic_analysis() -> None:
    company = {
        "url": "https://www.collabox.id",
        "company_name": "Collabox",
        "value_proposition": "Team OS",
        "pricing": "Rp 199k/mo",
        "key_features": ["Tasks", "Docs", "Chat"],
        "tech_stack": ["Next.js", "FastAPI"],
        "marketing_focus": "SMBs",
        "customer_feedback": "Fast onboarding",
        "engine_used": "scrapegraph_1",
        "pages_count": 6,
    }
    competitor = {
        "url": "https://competitor.example",
        "company_name": "RivalApp",
        "value_proposition": "Enterprise PM",
        "pricing": "$25/user",
        "key_features": ["Gantt", "Sprints", "Reporting"],
        "tech_stack": ["Java"],
        "marketing_focus": "Enterprise",
        "customer_feedback": "Steep learning curve",
        "engine_used": "scrapegraph_2",
        "pages_count": 4,
    }

    df = build_comparison_dataframe(company, [competitor, None], output_language="id")  # type: ignore[list-item]
    assert len(df) == 2
    assert list(df["Peran"]) == ["Perusahaan Anda", "Kompetitor #1"]
    assert "ScrapeGraphAI (Akun #1) (6 hal.)" in df.iloc[0]["Sumber Crawl"]
    assert "ScrapeGraphAI (Akun #2) (4 hal.)" in df.iloc[1]["Sumber Crawl"]

    captured_prompt: dict[str, str] = {}

    class FakeModels:
        def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
            captured_prompt["contents"] = contents
            return SimpleNamespace(text="## 1. Executive Positioning Summary\nSolid gap found.")

    fake_client = SimpleNamespace(models=FakeModels())
    settings = AppSettings(gemini_api_key="gem-key", scrapegraph_api_key="sgai-key-1")
    report = generate_strategic_analysis(fake_client, settings, company, [competitor], output_language="id")
    assert "Solid gap found" in report
    assert "pages_crawled" not in captured_prompt["contents"]


def test_extract_structured_profile_falls_back_to_openrouter_with_reasoning_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("analyzer.time.sleep", lambda _: None)
    captured_posts: list[dict[str, Any]] = []

    class FailingGeminiModels:
        def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
            raise RuntimeError("429 RESOURCE_EXHAUSTED: Quota exceeded")

    class FakeOpenRouterResponse:
        def __init__(self, status_code: int, payload: dict[str, Any]) -> None:
            self.status_code = status_code
            self._payload = payload
            self.text = json.dumps(payload)

        def json(self) -> dict[str, Any]:
            return self._payload

    class FakeOpenRouterSession:
        def post(self, url: str, headers: dict[str, str], data: str, timeout: int) -> FakeOpenRouterResponse:
            body = json.loads(data)
            captured_posts.append({"url": url, "headers": headers, "body": body, "timeout": timeout})
            if len(captured_posts) == 1:
                return FakeOpenRouterResponse(
                    200,
                    {
                        "choices": [
                            {
                                "message": {
                                    "role": "assistant",
                                    "content": "Let me analyze the website first...",
                                    "reasoning_details": [{"type": "reasoning.text", "text": "step 1"}],
                                }
                            }
                        ]
                    },
                )
            schema_json = json.dumps(
                {
                    "company_name": "Collabox",
                    "value_proposition": "Workspace kolaborasi tim terpadu",
                    "pricing": "Rp 199.000/bulan",
                    "key_features": ["Board", "Docs", "AI"],
                    "tech_stack": ["Python", "Next.js"],
                    "marketing_focus": "UMKM & agensi",
                    "customer_feedback": "Hemat waktu tim",
                }
            )
            return FakeOpenRouterResponse(
                200,
                {"choices": [{"message": {"role": "assistant", "content": schema_json}}]},
            )

    settings = AppSettings(
        gemini_api_key="gem-key",
        scrapegraph_api_key="sgai-key-1",
        openrouter_api_key="sk-or-test-key",
        openrouter_model="openrouter/free",
    )
    crawl_data = CrawlResultData(
        target_url="https://www.collabox.id",
        engine_used="scrapegraph_1",
        pages=[CrawledPage(url="https://www.collabox.id", title="Home", markdown="# Collabox")],
        combined_markdown="# Collabox",
    )

    profile = extract_structured_profile(
        SimpleNamespace(models=FailingGeminiModels()),
        settings,
        crawl_data,
        output_language="id",
        http_session=FakeOpenRouterSession(),  # type: ignore[arg-type]
    )

    assert profile["ai_provider"] == "openrouter"
    assert profile["company_name"] == "Collabox"
    assert "openrouter" in (profile["ai_fallback_note"] or "").lower()
    assert len(captured_posts) == 2
    assert captured_posts[0]["url"] == "https://openrouter.ai/api/v1/chat/completions"
    assert captured_posts[0]["headers"]["Authorization"] == "Bearer sk-or-test-key"
    assert captured_posts[0]["body"]["model"] == "openrouter/free"
    assert captured_posts[0]["body"]["reasoning"] == {"enabled": True}
    second_messages = captured_posts[1]["body"]["messages"]
    assert second_messages[1]["role"] == "assistant"
    assert second_messages[1]["reasoning_details"] == [{"type": "reasoning.text", "text": "step 1"}]


def test_generate_strategic_analysis_falls_back_to_openrouter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("analyzer.time.sleep", lambda _: None)
    captured_payloads: list[dict[str, Any]] = []

    class FailingGeminiModels:
        def generate_content(self, *, model: str, contents: str, config: Any) -> Any:
            raise RuntimeError("503 UNAVAILABLE")

    class FakeOpenRouterSession:
        def post(self, url: str, headers: dict[str, str], data: str, timeout: int) -> Any:
            captured_payloads.append(json.loads(data))
            return SimpleNamespace(
                status_code=200,
                text="",
                json=lambda: {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "## 1. Executive Positioning Summary\nFallback strategy via OpenRouter.",
                            }
                        }
                    ]
                },
            )

    settings = AppSettings(
        gemini_api_key="gem-key",
        scrapegraph_api_key="sgai-key-1",
        openrouter_api_key="sk-or-test-key",
    )
    company = {"url": "https://www.collabox.id", "company_name": "Collabox"}
    competitor = {"url": "https://rival.example", "company_name": "Rival"}

    report = generate_strategic_analysis(
        SimpleNamespace(models=FailingGeminiModels()),
        settings,
        company,
        [competitor],
        output_language="id",
        http_session=FakeOpenRouterSession(),  # type: ignore[arg-type]
    )

    assert "Fallback strategy via OpenRouter" in report
    assert captured_payloads[0]["model"] == "openrouter/free"
    assert captured_payloads[0]["reasoning"] == {"enabled": True}


def test_streamlit_ui_renders_cleanly_without_api_key_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    monkeypatch.setenv("SGAI_API_KEY", "sgai-test-key")
    app_path = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_path))
    at.run()
    assert len(at.exception) == 0
    labels = [inp.label for inp in at.text_input]
    assert not any("api key" in lbl.lower() for lbl in labels)
    assert any("perusahaan" in lbl.lower() for lbl in labels)
    assert len(at.slider) == 0
