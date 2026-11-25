# AI Competitor Intelligence Agent

Aplikasi web interaktif berbasis **Streamlit** untuk riset kompetitor otomatis dengan multi-page crawling **ScrapeGraphAI** tanpa batas max page artifisial (beserta fallback hingga 3 akun API key ScrapeGraphAI) dan analisis strategis AI.

## Fitur Utama

- **`.env`-Only Credential Management:** Seluruh API key disimpan aman di file `.env`. Tidak ada form input API key di UI — langsung masukkan URL perusahaan dan kompetitor.
- **Crawler ScrapeGraphAI v2 dengan 3-Akun Fallback (`ScrapeGraphAI` + `MarkdownFormatConfig`):**
  1. **🥇 Akun #1 (`SGAI_API_KEY`):** Digunakan sebagai primary crawler via `sgai.crawl.start(...)` tanpa batas `max_pages`.
  2. **🥈 Akun #2 (`SGAI_API_KEY_2`):** Fallback otomatis hanya jika Akun #1 gagal, terkena limit/kuota, atau tidak mengembalikan konten.
  3. **🥉 Akun #3 (`SGAI_API_KEY_3`):** Fallback otomatis terakhir jika Akun #1 dan #2 tidak tersedia atau terkena limit.
- **Gemini AI Free Tier + OpenRouter Fallback:** Menggunakan SDK resmi `google-genai` (`gemini-3.8-flash` → `gemini-3.5-flash-lite`), lalu otomatis fallback ke **OpenRouter** (`https://openrouter.ai/api/v1/chat/completions`, model `openrouter/free` dengan `reasoning: {"enabled": True}`) jika Gemini terkena limit atau error.
- **UI Modern Berbasis Tabs, Animasi Proses & Ekspor Laporan:** Timeline proses animatif, tabel perbandingan terstruktur, rincian profil & subpage yang ter-crawl, rekomendasi strategis AI, serta tombol unduh laporan `.md` dan `.csv`.

## Arsitektur Modul

- [config.py](config.py) — Loader konfigurasi `.env` (3 key ScrapeGraphAI, Gemini, OpenRouter) dan normalisasi URL (`www.collabox.id` -> `https://www.collabox.id`).
- [crawler.py](crawler.py) — Crawler ScrapeGraphAI v2 + ekstraksi `/pages` dan rotasi fallback hingga 3 akun API key ScrapeGraphAI serta agregator Markdown.
- [analyzer.py](analyzer.py) — Ekstraksi JSON terstruktur (`CompetitorDataSchema`), pembangun tabel Pandas, dan generator analisis strategis via `google-genai` + OpenRouter fallback.
- [app.py](app.py) — Antarmuka Streamlit dengan animasi proses crawl dan analisis AI.

## Persyaratan & Instalasi

- **Python >= 3.12** (dibutuhkan oleh `scrapegraph-py >= 2.3.1`)

1. **Install dependensi:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Siapkan file `.env`:**
   Salin/edit file `.env`, lalu isi API key Anda:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   OPENROUTER_API_KEY=your_openrouter_api_key_here
   SGAI_API_KEY=your_scrapegraph_api_key_1_here
   SGAI_API_KEY_2=your_scrapegraph_api_key_2_here
   SGAI_API_KEY_3=your_scrapegraph_api_key_3_here
   ```

3. **Jalankan aplikasi:**
   ```bash
   streamlit run app.py
   ```

## Menjalankan Pengujian

```bash
pytest
```
