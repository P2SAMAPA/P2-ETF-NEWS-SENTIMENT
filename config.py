import os

HF_TOKEN    = os.environ.get("HF_TOKEN", "")
DATA_REPO   = "P2SAMAPA/fi-etf-macro-signal-master-data"
OUTPUT_REPO = "P2SAMAPA/p2-etf-news-sentiment-results"

# The sentiment cache is stored as its own file in OUTPUT_REPO (not the
# dated results files) — it accumulates incrementally across runs so we
# never have to re-fetch/re-score the entire GDELT history every day.
SENTIMENT_CACHE_FILENAME = "sentiment_cache.parquet"

UNIVERSES = {
    "FI_COMMODITIES": ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"],
    "EQUITY_SECTORS": [
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "URA",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
    "COMBINED": [
        "TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV",
        "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV", "XLI", "XLY",
        "XLP", "XLU", "GDX", "XME", "IWF", "XSD", "XBI", "SMH", "SOXX", "URA",
        "IWM", "IWD", "IWO", "XLB", "XLRE",
    ],
}

# ── Data source: GDELT 2.0 GKG (free, 15-min updates, archives to 2015) ───────
# NOT Google Cloud/BigQuery — plain HTTP flat-file downloads, no API key,
# no account needed:
#   http://data.gdeltproject.org/gdeltv2/<timestamp>.gkg.csv.zip
#   http://data.gdeltproject.org/gdeltv2/masterfilelist.txt
#
# GDELT's GKG already includes a per-article tone score (V2Tone) computed by
# GDELT's own NLP pipeline, WITHOUT needing the article's full text — this is
# always available and used as a fallback. Where the article's own webpage
# can be fetched successfully, FinBERT is run on the actual text for a more
# targeted score. Article-page scraping is inherently fragile (paywalls,
# bot-blocking, dead links, JS-rendered pages) — this is a REAL engineering
# risk unique to this engine, unlike every other engine in this suite, which
# depends only on the already-proven HF master price dataset. The fallback
# to GDELT's own tone score exists specifically so a bad scraping day
# degrades gracefully rather than producing no data at all.

GDELT_BASE_URL = "http://data.gdeltproject.org/gdeltv2"
GDELT_SAMPLES_PER_DAY = 8      # GDELT publishes 96 15-min files/day; sample a
                                # subset to bound bandwidth/runtime — a daily
                                # aggregate doesn't need exhaustive coverage
MAX_ARTICLES_PER_TICKER_PER_DAY = 8   # cap on how many articles get a full
                                        # text-fetch + FinBERT pass per ticker/day
REQUEST_TIMEOUT_SECONDS = 10

BACKFILL_DAYS = 400   # initial history to build on first run (slow, one-time);
                       # every subsequent run only fetches new days since the
                       # cache's last date

# ── Sector/macro keyword mapping ──────────────────────────────────────────────
# MACRO keywords apply to every ticker in every universe; sector keywords are
# additive, applied only to the relevant ticker(s). This is GDELT GKG THEME/
# keyword matching against article titles and GKG-extracted themes, not full-
# text search.

MACRO_KEYWORDS = [
    "federal reserve", "fomc", "interest rate", "inflation", "cpi", "jobs report",
    "unemployment", "gdp", "recession", "treasury yield", "central bank",
]

SECTOR_KEYWORDS = {
    "TLT":  ["treasury bond", "long-term interest rate", "yield curve"],
    "VCIT": ["corporate bond", "investment grade credit"],
    "LQD":  ["corporate bond", "investment grade credit"],
    "HYG":  ["high yield bond", "junk bond", "credit spread"],
    "VNQ":  ["real estate", "reit", "commercial property", "housing market"],
    "GLD":  ["gold price", "gold market", "safe haven"],
    "SLV":  ["silver price", "silver market"],

    "SPY":  ["stock market", "wall street", "s&p 500"],
    "QQQ":  ["nasdaq", "big tech"],
    "XLK":  ["technology sector", "software", "semiconductor"],
    "XLF":  ["bank", "financial sector", "insurance company"],
    "XLE":  ["oil price", "energy sector", "opec", "natural gas"],
    "XLV":  ["healthcare", "pharmaceutical", "biotech industry"],
    "XLI":  ["industrial sector", "manufacturing", "aerospace"],
    "XLY":  ["consumer discretionary", "retail sales", "e-commerce"],
    "XLP":  ["consumer staples", "packaged goods"],
    "XLU":  ["utility sector", "electric utility", "power grid"],
    "GDX":  ["gold mining", "mining company"],
    "XME":  ["metals and mining", "steel industry", "commodity mining"],
    "IWF":  ["growth stocks", "large cap growth"],
    "XSD":  ["semiconductor industry", "chipmaker"],
    "XBI":  ["biotechnology", "drug development", "fda approval"],
    "IWM":  ["small cap stocks", "russell 2000"],
    "IWD":  ["value stocks", "large cap value"],
    "IWO":  ["small cap growth"],
    "XLB":  ["materials sector", "chemical industry"],
    "XLRE": ["real estate sector", "reit"],
}

# ── Sentiment scoring: FinBERT, with GDELT tone as the graceful fallback ─────
# FinBERT (ProsusAI/finbert) is a BERT-scale model fine-tuned for financial
# sentiment classification — CPU-feasible, no GPU required, no per-call API
# cost (unlike calling an LLM API), matching this suite's free/self-hosted
# philosophy. This is nonetheless the first engine in the suite with a heavy
# ML dependency (torch + transformers) rather than pure numpy.
FINBERT_MODEL_NAME = "ProsusAI/finbert"
FINBERT_MAX_LENGTH  = 256
GDELT_TONE_SCALE     = 10.0   # V2Tone typically ranges roughly -10..+10;
                               # divide by this and clip to make it
                               # comparable to FinBERT's native [-1,1] output

# ── Regression + windows ──────────────────────────────────────────────────────
WINDOWS = [63, 126, 252, 504]
PRED_HORIZON     = 21     # H: forward return horizon (regression target)
SENTIMENT_MOM_LOOKBACK = 5   # days used to measure sentiment persistence/momentum

# ── Score construction ────────────────────────────────────────────────────────
# sentiment_signal      : OLS-predicted forward return from today's sentiment
#                         features — the primary, direct signal
# sentiment_persistence : has sentiment been consistently one-directional
#                         recently, or is today a one-off blip? Confirmation,
#                         scaled by the sign of sentiment_signal.
# fit_quality           : R^2 of the OLS regression on its own training data
#
# IMPORTANT HONEST CAVEAT: the empirical literature on text sentiment
# predicting MULTI-DAY-FORWARD returns (as opposed to same-day or intraday
# reactions) is genuinely mixed and often weak. Treat fit_quality here with
# real skepticism relative to the other engines in this suite — this is
# exactly the kind of claim the TNN graph-value ablation was built to test
# empirically rather than assume, and the same discipline applies here.

WEIGHT_SENTIMENT    = 0.50
WEIGHT_PERSISTENCE   = 0.25
WEIGHT_FIT            = 0.25

TOP_N = 3
