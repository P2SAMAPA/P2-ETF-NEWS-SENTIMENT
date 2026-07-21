# 📰 P2-ETF-NEWS-SENTIMENT

**News Sentiment Engine — GDELT + FinBERT**

Part of the **P2Quant Engine Suite** · [P2SAMAPA](https://github.com/P2SAMAPA)

---

## What This Engine Does

This is the first engine in the suite built on **text**, not price/return
data. It ingests free news data from GDELT, scores it with FinBERT
(falling back gracefully to GDELT's own tone score when article text
can't be fetched), and fits a simple closed-form OLS regression of forward
returns on sentiment features — deliberately not a neural network, since
a sentiment-return relationship, if one exists, is low-dimensional.

**Read the honest caveats section before trusting this the way you would
other engines in this suite.**

---

## Data Source: GDELT 2.0 GKG

Free, U.S.-government-funded, updated every 15 minutes, archives back to
2015 — no API key, no account, plain HTTP flat-file downloads:# P2-ETF-NEWS-SENTIMENT
http://data.gdeltproject.org/gdeltv2/<timestamp>.gkg.csv.zip

Every GKG record already includes a per-article tone score (`V2Tone`)
computed by GDELT's own NLP pipeline, without needing the article's full
text. This is the always-available fallback used when full-text scoring
fails.

**Why not earnings call transcripts?** Your ETFs are broad baskets (SPY,
sector ETFs, bond ETFs) — individual company earnings calls don't map
cleanly onto them, and full transcripts are paywalled everywhere free
anyway. Macro-level and sector-keyword-filtered news maps far better onto
what these tickers actually are.

---

## Bucketing: Keyword Matching on Metadata, Not Full-Text Search

Which ticker(s) an article applies to is decided from GDELT's own
metadata — source name, extracted themes, URL — via keyword substring
matching. This does **not** require fetching the article webpage.
`MACRO_KEYWORDS` (Fed, rates, inflation, jobs, GDP, etc.) apply to every
ticker; `SECTOR_KEYWORDS` apply only to their specific ETF (e.g. "oil
price", "opec" → XLE; "semiconductor" → XLK/XSD).

---

## Sentiment Scoring: FinBERT with Graceful GDELT Fallback

**FinBERT** (`ProsusAI/finbert`) is a BERT-scale model fine-tuned for
financial sentiment classification — CPU-feasible, no GPU required, no
per-call API cost (unlike calling an LLM API for sentiment, which would
work but breaks the zero-marginal-cost pattern every other engine in this
suite follows).

Article-webpage fetching is **inherently fragile** — paywalls,
bot-blocking, dead links, JS-rendered pages with no readable text. This is
a real engineering risk unique to this engine, unlike every other engine
here, which depends only on the already-proven HF master price dataset.
Any article that can't be fetched or scored keeps its GDELT-native
`V2Tone` score instead (rescaled onto a comparable range) — this is
graceful degradation by design, not a failure state, and a
`n_finbert`/`n_fallback` split is tracked and surfaced in the dashboard so
you can see how much of a given day's sentiment came from which path.

Only a capped number of articles per ticker per day
(`MAX_ARTICLES_PER_TICKER_PER_DAY`) get an attempted full-text fetch, to
bound runtime regardless of how many articles matched.

---

## Model: Closed-Form OLS Regression

Deliberately **not** a neural network:
forward_return_t = a + bsentiment_t + csentiment_momentum_t + d*news_volume_t + eps_t

fit via exact linear algebra (`np.linalg.lstsq`), matching the same
"exact math, no approximation" philosophy as EDMD and GP-Vol elsewhere in
this suite. A sentiment-return relationship, if it exists at all, is
low-dimensional and doesn't justify more model complexity than the data
can support.

### Score Construction
score = 0.50sentiment_signal + 0.25sentiment_persistencesign(sentiment_signal) + 0.25fit_quality

| Component | Meaning |
|-----------|---------|
| sentiment_signal | OLS-predicted forward return from today's sentiment features |
| sentiment_persistence | Has sentiment been consistently one-directional recently, or is today a one-off blip? |
| fit_quality | R² of the OLS regression on its own training data |

---

## ⚠️ Honest Caveats — Read Before Trusting This Engine

1. **The empirical literature on text sentiment predicting MULTI-DAY-
   FORWARD returns is genuinely mixed and often weak.** Most robust
   findings concern same-day or intraday reactions to news, not multi-day-
   ahead prediction. Treat `fit_quality` here with more skepticism than
   the equivalent diagnostic elsewhere in this suite — this is exactly
   the kind of claim that should be validated with an out-of-sample
   ablation (the same discipline the Temporal Neighbourhood Network
   engine's `ablation_graph_value.py` applied) rather than trusted from
   in-sample R² alone. No such ablation has been built for this engine
   yet.

2. **Article-webpage scraping is fragile, by nature, not by bug.** Expect
   a meaningful fraction of articles to fall back to GDELT's native tone
   score rather than getting a full FinBERT pass. Check the dashboard's
   "Sentiment cache status" panel to see the actual FinBERT-vs-fallback
   split before trusting a given day's sentiment reading.

3. **The live GDELT fetch and FinBERT model download could not be tested
   from the sandbox this was built in** (network egress restrictions
   blocked both `data.gdeltproject.org` and `huggingface.co`). The
   aggregation, regression, and scoring logic were validated against
   synthetic data with a known ground-truth relationship — the OLS
   correctly recovered a positive coefficient (+0.0118) when a genuine
   positive sentiment-return relationship was built into synthetic data
   (`true_beta=0.003`), and showed a much lower fit on pure noise (0.055
   vs. 0.196 R²) — but the actual live network paths need verification
   once deployed somewhere with normal internet access.

4. **This is a heavier engineering lift than any other engine in the
   suite.** New data ingestion layer, first heavy ML dependency (torch +
   transformers, versus pure numpy everywhere else), and a caching
   strategy needed just to get useful historical depth before the first
   meaningful `trainer.py` run.

---

## Architecture: Ingestion and Scoring Are Separate
update_sentiment_cache.py   # GDELT + FinBERT ingestion, resumable/incremental
↓  (writes to)
sentiment_cache.parquet     # accumulates on HF across runs
↓  (read by)
trainer.py                  # scoring only — never fetches GDELT itself

**Incremental, resumable by design:** the cache is saved and re-uploaded
every 10 days processed, not just once at the end. An initial backfill
(`config.BACKFILL_DAYS`, default 400 days) can be slow — if a run times
out partway through, the next scheduled run just continues from the
cache's last date. No lost progress, no special "first run" handling
needed.

---

## Universes & Windows

| Universe | Tickers |
|---|---|
| FI_COMMODITIES | TLT, VCIT, LQD, HYG, VNQ, GLD, SLV |
| EQUITY_SECTORS | SPY, QQQ, XLK, XLF, XLE, XLV, XLI, XLY, XLP, XLU, GDX, XME, IWF, XSD, XBI, IWM, IWD, IWO, XLB, XLRE, SOXX, SMH |
| COMBINED | All of the above |

**Windows:** `63d · 126d · 252d · 504d`

---

## Repository Structure
P2-ETF-NEWS-SENTIMENT/
├── config.py                    # Universes, keyword mapping, GDELT/FinBERT params, weights
├── data_manager.py               # HuggingFace price loader
├── gdelt_client.py                # GDELT GKG fetch + metadata-based ticker bucketing
├── sentiment_scorer.py            # FinBERT scoring + GDELT-tone fallback
├── news_sentiment_engine.py        # Core: OLS regression, signal + diagnostics
├── update_sentiment_cache.py       # Ingestion (separate from scoring, resumable)
├── trainer.py                      # Scoring orchestrator (reads cache, never fetches GDELT)
├── push_results.py                 # HfApi.upload_file wrapper
├── streamlit_app.py                 # Two-tab Streamlit dashboard + cache status panel
├── us_calendar.py                  # US trading calendar helper
├── requirements.txt                 # First engine with torch/transformers/bs4
└── .github/
└── workflows/
└── daily.yml                # Two steps: ingest, then score

---

## Setup

```bash
git clone https://github.com/P2SAMAPA/P2-ETF-NEWS-SENTIMENT
cd P2-ETF-NEWS-SENTIMENT
pip install -r requirements.txt

export HF_TOKEN=hf_...
python update_sentiment_cache.py   # slow on first run — backfills BACKFILL_DAYS
python trainer.py
streamlit run streamlit_app.py
```

**Required GitHub secret:** `HF_TOKEN`

**Required HuggingFace dataset repo:** `P2SAMAPA/p2-etf-news-sentiment-results`

---

## Not Yet Built (Phase 4+)

Fed-statement-specific ingestion (FOMC statements/minutes/speeches, given
special weight for rate-sensitive tickers like TLT/VCIT/LQD/HYG) was
scoped but deliberately deferred — see the original design conversation.
An out-of-sample ablation script (mirroring TNN's `ablation_graph_value.py`)
would also be a natural next addition, given caveat #1 above.

---

## References

- Leetaru, K. & Schrodt, P. (2013). GDELT: Global Data on Events,
  Location, and Tone, 1979-2012.
- Araci, D. (2019). FinBERT: Financial Sentiment Analysis with
  Pre-trained Language Models.
