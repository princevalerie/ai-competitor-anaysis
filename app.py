"""Streamlit UI for AI Competitor Intelligence.

All API credentials are loaded strictly from `.env` via `config.py`.
Crawling runs without an artificial max-page limit using ScrapeGraphAI
with automatic fallback across up to 3 API key accounts (`SGAI_API_KEY`,
`SGAI_API_KEY_2`, `SGAI_API_KEY_3`).
"""

from __future__ import annotations

import time
from typing import Any

import pandas as pd
import streamlit as st

from analyzer import (
    build_comparison_dataframe,
    create_gemini_client,
    extract_structured_profile,
    generate_strategic_analysis,
)
from config import load_settings, normalize_url
from crawler import CrawlResultData, crawl_website

st.set_page_config(
    page_title="AI Competitor Intelligence",
    page_icon="🧲",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Custom CSS — profile cards & animated crawl indicators
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }
    .profile-card {
        padding: 1rem 1.15rem;
        border-radius: 0.75rem;
        border: 1px solid rgba(128, 128, 128, 0.22);
        margin-bottom: 0.9rem;
    }

    /* ---- animated crawl process styles ---- */
    @keyframes pulseGlow {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.15); }
    }
    @keyframes spinIcon {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    @keyframes slideInUp {
        from { opacity: 0; transform: translateY(12px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes progressFill {
        from { width: 0%; }
        to   { width: 100%; }
    }
    @keyframes dotBounce {
        0%, 80%, 100% { transform: translateY(0); }
        40% { transform: translateY(-6px); }
    }

    .crawl-step {
        animation: slideInUp 0.4s ease-out;
        padding: 0.6rem 0.9rem;
        border-radius: 0.6rem;
        margin-bottom: 0.5rem;
        border-left: 4px solid transparent;
    }
    .crawl-step-active {
        background: linear-gradient(135deg, rgba(59,130,246,0.08), rgba(147,51,234,0.08));
        border-left-color: #3b82f6;
    }
    .crawl-step-success {
        background: rgba(34,197,94,0.08);
        border-left-color: #22c55e;
    }
    .crawl-step-warn {
        background: rgba(234,179,8,0.08);
        border-left-color: #eab308;
    }
    .crawl-step-error {
        background: rgba(239,68,68,0.08);
        border-left-color: #ef4444;
    }

    .crawl-spinner {
        display: inline-block;
        animation: spinIcon 1s linear infinite;
        margin-right: 0.4rem;
    }
    .crawl-pulse {
        display: inline-block;
        animation: pulseGlow 1.5s ease-in-out infinite;
    }

    .crawl-progress-bar {
        height: 4px;
        border-radius: 2px;
        background: rgba(128,128,128,0.15);
        margin: 0.3rem 0 0.2rem;
        overflow: hidden;
    }
    .crawl-progress-fill {
        height: 100%;
        border-radius: 2px;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        animation: progressFill 2s ease-out forwards;
    }

    .tier-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 1rem;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    .tier-1 { background: #fef3c7; color: #92400e; }
    .tier-2 { background: #dbeafe; color: #1e40af; }
    .tier-3 { background: #e0e7ff; color: #3730a3; }

    .page-counter {
        font-size: 1.5rem;
        font-weight: 700;
        color: #3b82f6;
        margin: 0.3rem 0;
        font-variant-numeric: tabular-nums;
    }

    .dots-loading span {
        display: inline-block;
        width: 6px;
        height: 6px;
        border-radius: 50%;
        background: #3b82f6;
        margin: 0 2px;
        animation: dotBounce 1.2s ease-in-out infinite;
    }
    .dots-loading span:nth-child(2) { animation-delay: 0.15s; }
    .dots-loading span:nth-child(3) { animation-delay: 0.3s; }

    .phase-timeline {
        display: flex;
        align-items: center;
        gap: 0;
        margin: 0.8rem 0;
        flex-wrap: wrap;
    }
    .phase-node {
        display: flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.35rem 0.7rem;
        border-radius: 1.5rem;
        font-size: 0.82rem;
        font-weight: 500;
        white-space: nowrap;
    }
    .phase-node-pending {
        background: rgba(128,128,128,0.1);
        color: rgba(128,128,128,0.6);
    }
    .phase-node-active {
        background: linear-gradient(135deg, #3b82f6, #8b5cf6);
        color: white;
        box-shadow: 0 2px 8px rgba(59,130,246,0.3);
    }
    .phase-node-done {
        background: rgba(34,197,94,0.15);
        color: #15803d;
    }
    .phase-node-error {
        background: rgba(239,68,68,0.15);
        color: #b91c1c;
    }
    .phase-arrow {
        color: rgba(128,128,128,0.4);
        font-size: 0.9rem;
        margin: 0 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

settings = load_settings()

_ENGINE_BADGE = {
    "scrapegraph": "🥇 ScrapeGraphAI",
    "scrapegraph_1": "🥇 ScrapeGraphAI (Akun #1)",
    "scrapegraph_2": "🥈 ScrapeGraphAI (Akun #2)",
    "scrapegraph_3": "🥉 ScrapeGraphAI (Akun #3)",
}


# ---------------------------------------------------------------------------
# Animated UI helper functions
# ---------------------------------------------------------------------------
def _render_phase_timeline(phases: list[tuple[str, str]]) -> str:
    """Build an HTML phase-timeline bar.

    Each phase is ``(label, state)`` where *state* is one of
    ``"pending"``, ``"active"``, ``"done"``, ``"error"``.
    """
    nodes: list[str] = []
    for idx, (label, state) in enumerate(phases):
        icon_map = {"pending": "⏳", "active": "⚡", "done": "✅", "error": "❌"}
        icon = icon_map.get(state, "⏳")
        css_cls = f"phase-node phase-node-{state}"
        if state == "active":
            icon = f'<span class="crawl-pulse">{icon}</span>'
        nodes.append(f'<span class="{css_cls}">{icon} {label}</span>')
        if idx < len(phases) - 1:
            nodes.append('<span class="phase-arrow">→</span>')
    return f'<div class="phase-timeline">{"".join(nodes)}</div>'


def _render_crawl_step(html_content: str, step_class: str = "active") -> str:
    """Wrap content in an animated crawl-step div."""
    return f'<div class="crawl-step crawl-step-{step_class}">{html_content}</div>'


def _render_crawl_active(url: str, tier_label: str, tier_num: int) -> str:
    """Render an actively-crawling step with spinner + dots."""
    return _render_crawl_step(
        f'<span class="crawl-spinner">🔄</span>'
        f'<span class="tier-badge tier-{tier_num}">{tier_label}</span> '
        f'Meng-crawl <code>{url}</code>'
        f'<div class="crawl-progress-bar"><div class="crawl-progress-fill"></div></div>'
        f'<div class="dots-loading"><span></span><span></span><span></span></div>',
        step_class="active",
    )


def _render_crawl_success(url: str, engine: str, page_count: int) -> str:
    """Render a completed crawl step."""
    badge = _ENGINE_BADGE.get(engine, engine)
    return _render_crawl_step(
        f'✅ <code>{url}</code> selesai via <strong>{badge}</strong> '
        f'— <span class="page-counter">{page_count}</span> halaman ditemukan',
        step_class="success",
    )


def _render_crawl_warning(url: str, message: str) -> str:
    """Render a warning/skip step."""
    return _render_crawl_step(
        f"⚠️ <code>{url}</code> — {message}",
        step_class="warn",
    )


def _render_fallback_note(note: str) -> str:
    """Render a fallback cascade note."""
    return _render_crawl_step(
        f"↪️ <em>{note}</em>",
        step_class="warn",
    )


def _render_extraction_active(url: str) -> str:
    """Render AI extraction in-progress."""
    return _render_crawl_step(
        f'<span class="crawl-spinner">🧠</span> Mengekstrak profil AI dari <code>{url}</code>'
        f'<div class="crawl-progress-bar"><div class="crawl-progress-fill" '
        f'style="animation-duration: 4s;"></div></div>',
        step_class="active",
    )


def _render_extraction_done(
    url: str,
    company_name: str,
    ai_provider: str = "gemini",
    fallback_note: str | None = None,
) -> str:
    """Render completed AI extraction."""
    provider_label = "OpenRouter Fallback" if ai_provider == "openrouter" else "Gemini AI"
    note_html = f'<br><small>↪️ <em>{fallback_note}</em></small>' if fallback_note else ""
    return _render_crawl_step(
        f'🧠 Profil <strong>{company_name}</strong> (<code>{url}</code>) berhasil diekstrak '
        f'via <strong>{provider_label}</strong>{note_html}',
        step_class="success",
    )


def _render_analysis_active() -> str:
    """Render strategic analysis in-progress."""
    return _render_crawl_step(
        '<span class="crawl-spinner">🎯</span> Menyusun analisis strategis AI (Gemini / OpenRouter fallback)'
        '<div class="crawl-progress-bar"><div class="crawl-progress-fill" '
        'style="animation-duration: 6s;"></div></div>',
        step_class="active",
    )


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    """Render a Pandas DataFrame as a GFM pipe table without optional `tabulate`."""
    headers = [str(col).replace("|", "\\|") for col in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in df.iterrows():
        cells = [
            str(val).replace("\r\n", " ").replace("\n", " ").replace("|", "\\|")
            for val in row.tolist()
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _format_markdown_report(
    company_profile: dict[str, Any],
    competitor_profiles: list[dict[str, Any]],
    comparison_df: pd.DataFrame,
    strategic_analysis: str,
) -> str:
    """Render a downloadable Markdown report combining summary, table, and strategy."""
    lines = [
        "# AI Competitor Intelligence Report",
        "",
        f"**Target Company:** {company_profile.get('company_name')} ({company_profile.get('url')})",
        "",
        "## Competitor Comparison",
        "",
        _dataframe_to_markdown(comparison_df),
        "",
        "## Strategic Analysis & Recommendations",
        "",
        strategic_analysis,
        "",
        "## Crawl Coverage & Fallback Log",
        "",
    ]
    for profile in [company_profile] + competitor_profiles:
        engine = _ENGINE_BADGE.get(profile.get("engine_used", ""), "Unknown")
        lines.append(
            f"- **{profile.get('company_name')}** ({profile.get('url')}) — {engine}, {profile.get('pages_count', 0)} pages"
        )
        for note in profile.get("fallback_notes", []):
            lines.append(f"  - Fallback note: {note}")
    return "\n".join(lines)


def _validate_inputs(
    raw_company_url: str, raw_competitor_urls: list[str]
) -> tuple[str | None, list[str], list[str]]:
    """Normalize URLs and collect validation errors without aborting on first issue."""
    errors: list[str] = []
    norm_company: str | None = None
    norm_competitors: list[str] = []
    seen: set[str] = set()

    if not raw_company_url.strip():
        errors.append("Masukkan URL website perusahaan Anda terlebih dahulu.")
    else:
        try:
            norm_company = normalize_url(raw_company_url)
            seen.add(norm_company.rstrip("/"))
        except ValueError as exc:
            errors.append(f"URL Perusahaan tidak valid: {exc}")

    for idx, raw_url in enumerate(raw_competitor_urls, start=1):
        if not raw_url.strip():
            continue
        try:
            norm_url = normalize_url(raw_url)
            key = norm_url.rstrip("/")
            if key not in seen:
                seen.add(key)
                norm_competitors.append(norm_url)
        except ValueError as exc:
            errors.append(f"URL Kompetitor #{idx} tidak valid: {exc}")

    if not norm_competitors and not any("Kompetitor" in err for err in errors):
        errors.append("Masukkan minimal 1 URL website kompetitor.")

    return norm_company, norm_competitors, errors


# ---------------------------------------------------------------------------
# Sidebar: read-only .env status + report language
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("🧲 Competitor Intel")
    st.caption("Semua API key dibaca otomatis dari file `.env`.")

    st.subheader("Status Koneksi (.env)")
    if settings.has_gemini_key:
        st.success(f"✅ Gemini AI siap (`{settings.gemini_model}`)")
    else:
        st.warning("⚠️ `GEMINI_API_KEY` belum diisi di `.env`")

    if settings.has_openrouter_key:
        st.success(f"✅ OpenRouter Fallback siap (`{settings.openrouter_model}`)")
    else:
        st.warning("⚠️ `OPENROUTER_API_KEY` belum diset (Fallback Gemini nonaktif)")

    for slot_num, _slot_key, is_ok in settings.scrapegraph_slots:
        env_name = "SGAI_API_KEY" if slot_num == 1 else f"SGAI_API_KEY_{slot_num}"
        slot_role = "Primary" if slot_num == 1 else f"Fallback {slot_num - 1}"
        if is_ok:
            st.success(f"✅ ScrapeGraphAI Akun #{slot_num} siap ({slot_role})")
        elif slot_num == 1 and not settings.has_scrapegraph_key:
            st.error(f"❌ `{env_name}` belum diset")
        else:
            st.info(f"ℹ️ `{env_name}` kosong ({slot_role} opsional)")

    st.divider()
    st.subheader("Preferensi Laporan")
    language_label = st.selectbox(
        "Bahasa Output Analisis",
        options=["Bahasa Indonesia", "English"],
        index=0,
    )
    output_lang = "id" if language_label == "Bahasa Indonesia" else "en"

    st.caption(
        "Crawl berjalan **tanpa batas max page** (mengikuti kuota/limit provider, "
        "lalu otomatis fallback jika terkena limit)."
    )


# ---------------------------------------------------------------------------
# Main Workspace
# ---------------------------------------------------------------------------
st.title("🧲 AI Competitor Intelligence Agent")

if not settings.has_scrapegraph_key:
    st.error(
        "Konfigurasi `.env` belum lengkap: isi minimal 1 key ScrapeGraphAI (`SGAI_API_KEY`) "
        "di file `.env` (lalu refresh halaman ini) sebelum memulai crawl."
    )

if not settings.has_ai_provider:
    st.error(
        "Konfigurasi `.env` belum lengkap: isi minimal `GEMINI_API_KEY` atau `OPENROUTER_API_KEY` "
        "di file `.env` (lalu refresh halaman ini) sebelum memulai analisis."
    )
elif not settings.has_gemini_key and settings.has_openrouter_key:
    st.info(
        f"`GEMINI_API_KEY` belum diisi; analisis akan langsung menggunakan fallback **OpenRouter** (`{settings.openrouter_model}`)."
    )

with st.container(border=True):
    col_comp, col_count = st.columns([3, 1])
    with col_comp:
        company_url_input = st.text_input(
            "URL Website Perusahaan Anda",
            placeholder="contoh: www.collabox.id atau https://example.com",
        )
    with col_count:
        num_competitors = st.number_input(
            "Jumlah Kompetitor",
            min_value=1,
            max_value=5,
            value=1,
        )

    st.markdown("**URL Website Kompetitor**")
    comp_cols = st.columns(2)
    competitor_inputs: list[str] = []
    for i in range(int(num_competitors)):
        with comp_cols[i % 2]:
            competitor_inputs.append(
                st.text_input(
                    f"Kompetitor #{i + 1}",
                    key=f"comp_{i}",
                    placeholder="contoh: www.competitor.com",
                )
            )

    submitted = st.button(
        "🚀 Analyze Competitors",
        type="primary",
        use_container_width=True,
        disabled=not (settings.has_ai_provider and settings.has_scrapegraph_key),
    )


# ---------------------------------------------------------------------------
# Processing pipeline with rich animations
# ---------------------------------------------------------------------------
if submitted:
    norm_company, norm_competitors, validation_errors = _validate_inputs(
        company_url_input, competitor_inputs
    )

    if validation_errors:
        for err in validation_errors:
            st.error(err)
    elif norm_company:
        gemini_client = create_gemini_client(settings.gemini_api_key)
        company_profile: dict[str, Any] | None = None
        competitor_profiles: list[dict[str, Any]] = []

        all_urls = [norm_company] + norm_competitors
        total_sites = len(all_urls)

        # Build initial phase timeline
        phase_labels = [
            ("Crawl & Ekstrak", "active"),
            ("Perbandingan", "pending"),
            ("Analisis Strategis", "pending"),
        ]

        with st.status("🚀 Memulai proses crawl & analisis AI...", expanded=True) as status_box:
            # --- Phase timeline ---
            timeline_placeholder = st.empty()
            timeline_placeholder.markdown(
                _render_phase_timeline(phase_labels), unsafe_allow_html=True
            )

            # === PHASE 1: Crawl & Extract ===
            st.markdown(
                f"##### 🕷️ Fase 1: Crawling & Ekstraksi ({total_sites} website)",
                unsafe_allow_html=True,
            )

            # 1. Crawl & extract company website
            anim_slot = st.empty()
            anim_slot.markdown(
                _render_crawl_active(norm_company, "Perusahaan Anda", 1),
                unsafe_allow_html=True,
            )

            t0 = time.perf_counter()
            company_crawl: CrawlResultData = crawl_website(norm_company, settings)
            crawl_elapsed = time.perf_counter() - t0

            if not company_crawl.success:
                anim_slot.markdown(
                    _render_crawl_warning(
                        norm_company,
                        f"Gagal mengambil konten: {company_crawl.error_message}",
                    ),
                    unsafe_allow_html=True,
                )
                for note in company_crawl.fallback_notes:
                    st.markdown(_render_fallback_note(note), unsafe_allow_html=True)
                status_box.update(
                    label="Gagal mengambil data website perusahaan", state="error"
                )
            else:
                anim_slot.markdown(
                    _render_crawl_success(
                        norm_company,
                        company_crawl.engine_used,
                        len(company_crawl.pages),
                    ),
                    unsafe_allow_html=True,
                )
                for note in company_crawl.fallback_notes:
                    st.markdown(_render_fallback_note(note), unsafe_allow_html=True)

                st.caption(f"⏱️ Waktu crawl: {crawl_elapsed:.1f}s")

                # AI extraction
                extract_slot = st.empty()
                extract_slot.markdown(
                    _render_extraction_active(norm_company), unsafe_allow_html=True
                )
                try:
                    company_profile = extract_structured_profile(
                        gemini_client,
                        settings,
                        company_crawl,
                        output_language=output_lang,
                    )
                    name = company_profile.get("company_name", norm_company)
                    extract_slot.markdown(
                        _render_extraction_done(
                            norm_company,
                            name,
                            ai_provider=company_profile.get("ai_provider", "gemini"),
                            fallback_note=company_profile.get("ai_fallback_note"),
                        ),
                        unsafe_allow_html=True,
                    )
                except (RuntimeError, ValueError, OSError) as exc:
                    extract_slot.markdown(
                        _render_crawl_step(
                            f'❌ Gagal mengekstrak profil AI dari <code>{norm_company}</code>: {exc}',
                            step_class="error",
                        ),
                        unsafe_allow_html=True,
                    )
                    status_box.update(label="Gagal saat ekstraksi AI", state="error")

            # 2. Crawl & extract competitors
            if company_profile:
                st.divider()
                for comp_idx, comp_url in enumerate(norm_competitors, start=1):
                    comp_anim = st.empty()
                    comp_anim.markdown(
                        _render_crawl_active(
                            comp_url, f"Kompetitor #{comp_idx}", 1
                        ),
                        unsafe_allow_html=True,
                    )

                    t0 = time.perf_counter()
                    comp_crawl = crawl_website(comp_url, settings)
                    comp_elapsed = time.perf_counter() - t0

                    if not comp_crawl.success:
                        comp_anim.markdown(
                            _render_crawl_warning(
                                comp_url,
                                f"Gagal crawl: {comp_crawl.error_message}",
                            ),
                            unsafe_allow_html=True,
                        )
                        for note in comp_crawl.fallback_notes:
                            st.markdown(
                                _render_fallback_note(note), unsafe_allow_html=True
                            )
                        continue

                    comp_anim.markdown(
                        _render_crawl_success(
                            comp_url,
                            comp_crawl.engine_used,
                            len(comp_crawl.pages),
                        ),
                        unsafe_allow_html=True,
                    )
                    for note in comp_crawl.fallback_notes:
                        st.markdown(
                            _render_fallback_note(note), unsafe_allow_html=True
                        )
                    st.caption(f"⏱️ Waktu crawl: {comp_elapsed:.1f}s")

                    # AI extraction for competitor
                    comp_extract = st.empty()
                    comp_extract.markdown(
                        _render_extraction_active(comp_url), unsafe_allow_html=True
                    )
                    try:
                        comp_profile = extract_structured_profile(
                            gemini_client,
                            settings,
                            comp_crawl,
                            output_language=output_lang,
                        )
                        comp_name = comp_profile.get("company_name", comp_url)
                        comp_extract.markdown(
                            _render_extraction_done(
                                comp_url,
                                comp_name,
                                ai_provider=comp_profile.get("ai_provider", "gemini"),
                                fallback_note=comp_profile.get("ai_fallback_note"),
                            ),
                            unsafe_allow_html=True,
                        )
                        competitor_profiles.append(comp_profile)
                    except (RuntimeError, ValueError, OSError) as exc:
                        comp_extract.markdown(
                            _render_crawl_step(
                                f'⚠️ Gagal mengekstrak profil AI dari <code>{comp_url}</code>: {exc}',
                                step_class="warn",
                            ),
                            unsafe_allow_html=True,
                        )

            # === PHASE 2 & 3: Comparison & Strategic Analysis ===
            if company_profile and competitor_profiles:
                st.divider()
                phase_labels = [
                    ("Crawl & Ekstrak", "done"),
                    ("Perbandingan", "active"),
                    ("Analisis Strategis", "pending"),
                ]
                timeline_placeholder.markdown(
                    _render_phase_timeline(phase_labels), unsafe_allow_html=True
                )
                st.markdown("##### 📊 Fase 2: Menyusun Perbandingan")

                analysis_anim = st.empty()
                analysis_anim.markdown(
                    _render_crawl_step(
                        '<span class="crawl-spinner">📊</span> Membangun tabel perbandingan kompetitor...',
                        step_class="active",
                    ),
                    unsafe_allow_html=True,
                )

                try:
                    comparison_df = build_comparison_dataframe(
                        company_profile,
                        competitor_profiles,
                        output_language=output_lang,
                    )
                    analysis_anim.markdown(
                        _render_crawl_step(
                            "✅ Tabel perbandingan berhasil disusun",
                            step_class="success",
                        ),
                        unsafe_allow_html=True,
                    )

                    # Phase 3: Strategic analysis
                    phase_labels = [
                        ("Crawl & Ekstrak", "done"),
                        ("Perbandingan", "done"),
                        ("Analisis Strategis", "active"),
                    ]
                    timeline_placeholder.markdown(
                        _render_phase_timeline(phase_labels),
                        unsafe_allow_html=True,
                    )
                    st.markdown("##### 🎯 Fase 3: Analisis Strategis AI")

                    strat_anim = st.empty()
                    strat_anim.markdown(
                        _render_analysis_active(), unsafe_allow_html=True
                    )

                    strategic_analysis = generate_strategic_analysis(
                        gemini_client,
                        settings,
                        company_profile,
                        competitor_profiles,
                        output_language=output_lang,
                    )

                    strat_anim.markdown(
                        _render_crawl_step(
                            "✅ Analisis strategis selesai dihasilkan",
                            step_class="success",
                        ),
                        unsafe_allow_html=True,
                    )

                    # Final timeline — all done
                    phase_labels = [
                        ("Crawl & Ekstrak", "done"),
                        ("Perbandingan", "done"),
                        ("Analisis Strategis", "done"),
                    ]
                    timeline_placeholder.markdown(
                        _render_phase_timeline(phase_labels),
                        unsafe_allow_html=True,
                    )

                    st.session_state["last_analysis"] = {
                        "company_profile": company_profile,
                        "competitor_profiles": competitor_profiles,
                        "comparison_df": comparison_df,
                        "strategic_analysis": strategic_analysis,
                    }
                    status_box.update(
                        label=f"✨ Analisis selesai ({1 + len(competitor_profiles)} website berhasil diproses)",
                        state="complete",
                        expanded=False,
                    )
                except (RuntimeError, ValueError, OSError) as exc:
                    phase_labels = [
                        ("Crawl & Ekstrak", "done"),
                        ("Perbandingan", "error"),
                        ("Analisis Strategis", "error"),
                    ]
                    timeline_placeholder.markdown(
                        _render_phase_timeline(phase_labels),
                        unsafe_allow_html=True,
                    )
                    st.error(f"Gagal menghasilkan analisis strategis: {exc}")
                    status_box.update(
                        label="Gagal saat analisis strategis", state="error"
                    )
            elif company_profile:
                st.error(
                    "Tidak ada data kompetitor yang berhasil dikumpulkan untuk dibandingkan."
                )
                status_box.update(
                    label="Data kompetitor tidak mencukupi", state="error"
                )


# ---------------------------------------------------------------------------
# Persisted results view
# ---------------------------------------------------------------------------
if "last_analysis" in st.session_state:
    result = st.session_state["last_analysis"]
    company_prof = result["company_profile"]
    competitor_profs = result["competitor_profiles"]
    comparison_df = result["comparison_df"]
    strategic_analysis = result["strategic_analysis"]
    all_profs = [company_prof] + competitor_profs

    total_pages = sum(p.get("pages_count", 0) for p in all_profs)
    sg1_count = sum(1 for p in all_profs if p.get("engine_used") in ("scrapegraph", "scrapegraph_1"))
    sg2_count = sum(1 for p in all_profs if p.get("engine_used") == "scrapegraph_2")
    sg3_count = sum(1 for p in all_profs if p.get("engine_used") == "scrapegraph_3")

    st.divider()
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Website Dianalisis", len(all_profs))
    kpi2.metric("Total Halaman Ter-crawl", total_pages)
    kpi3.metric("🥇 ScrapeGraph #1", sg1_count)
    kpi4.metric("🥈 Akun #2 / 🥉 Akun #3", f"{sg2_count} / {sg3_count}")

    dl_col1, dl_col2, _dl_spacer = st.columns([1, 1, 2])
    report_md = _format_markdown_report(
        company_prof, competitor_profs, comparison_df, strategic_analysis
    )
    dl_col1.download_button(
        label="⬇️ Download Laporan (.md)",
        data=report_md,
        file_name="competitor_intelligence_report.md",
        mime="text/markdown",
        use_container_width=True,
    )
    dl_col2.download_button(
        label="⬇️ Download Tabel (.csv)",
        data=comparison_df.to_csv(index=False).encode("utf-8"),
        file_name="competitor_comparison.csv",
        mime="text/csv",
        use_container_width=True,
    )

    tab_comp, tab_strategy, tab_profiles = st.tabs(
        ["📊 Perbandingan Kompetitor", "🎯 Strategic Analysis", "🔎 Detail Profil & Subpage"]
    )

    with tab_comp:
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    with tab_strategy, st.container(border=True):
        st.markdown(strategic_analysis)

    with tab_profiles:
        for idx, prof in enumerate(all_profs):
            role = "Perusahaan Anda" if idx == 0 else f"Kompetitor #{idx}"
            badge = _ENGINE_BADGE.get(prof.get("engine_used", ""), "Unknown")
            with st.container(border=True):
                head_l, head_r = st.columns([3, 1])
                head_l.markdown(f"### {role}: {prof.get('company_name')}")
                head_r.caption(f"{badge} • {prof.get('pages_count', 0)} halaman")

                if prof.get("ai_fallback_note"):
                    st.info(f"AI fallback log: {prof['ai_fallback_note']}")
                if prof.get("fallback_notes"):
                    for note in prof["fallback_notes"]:
                        st.info(f"Fallback log: {note}")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f"**URL:** {prof.get('url')}")
                    st.markdown(f"**Value Proposition:** {prof.get('value_proposition')}")
                    st.markdown(f"**Pricing:** {prof.get('pricing')}")
                with c2:
                    st.markdown(f"**Key Features:** {', '.join(prof.get('key_features', []))}")
                    st.markdown(f"**Tech Stack:** {', '.join(prof.get('tech_stack', []))}")
                    st.markdown(f"**Marketing Focus:** {prof.get('marketing_focus')}")
                    st.markdown(f"**Customer Feedback:** {prof.get('customer_feedback')}")

                if prof.get("pages_crawled"):
                    with st.expander(f"Lihat {len(prof['pages_crawled'])} halaman yang ter-crawl"):
                        for page_url in prof["pages_crawled"]:
                            st.markdown(f"- `{page_url}`")
