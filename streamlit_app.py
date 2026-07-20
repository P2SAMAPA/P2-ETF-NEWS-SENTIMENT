import streamlit as st
import pandas as pd
import json
from huggingface_hub import HfFileSystem
import config
from us_calendar import next_trading_day

st.set_page_config(page_title="News Sentiment Engine", layout="wide")

st.markdown("""
<style>
.main-header { font-size:2.4rem; font-weight:700; color:#2c1810; margin-bottom:0.3rem; }
.sub-header  { font-size:1.1rem; color:#555; margin-bottom:1.5rem; }
.uni-title   { font-size:1.4rem; font-weight:600; margin-top:1rem; margin-bottom:0.8rem;
               padding-left:0.5rem; border-left:5px solid #a0522d; }
.etf-card    { background:linear-gradient(135deg,#2c1810 0%,#a0522d 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.win-card    { background:linear-gradient(135deg,#2c1810 0%,#5c3a26 100%); color:white;
               border-radius:14px; padding:1rem; margin:0.4rem; text-align:center;
               box-shadow:0 4px 6px rgba(0,0,0,0.2); }
.etf-ticker  { font-size:1.3rem; font-weight:bold; }
.etf-score   { font-size:0.88rem; margin-top:0.25rem; opacity:0.9; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">📰 News Sentiment Engine</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">GDELT (free, 15-min updates) + FinBERT financial sentiment · '
    'OLS regression on sentiment features — exact linear algebra, no neural network · '
    'Multi-window cross-sectional z-score</div>',
    unsafe_allow_html=True)

st.warning(
    "⚠️ **Honest caveat**: the empirical literature on text sentiment predicting "
    "MULTI-DAY-FORWARD returns is genuinely mixed and often weak. Treat `fit_quality` "
    "here with more skepticism than the equivalent diagnostic in other engines in this "
    "suite — this is exactly the kind of claim worth validating with an out-of-sample "
    "ablation (as was built for the Temporal Neighbourhood Network engine) rather than "
    "trusting in-sample R² alone."
)

st.sidebar.markdown("## News Sentiment Engine")
st.sidebar.markdown(f"**Next Trading Day:** `{next_trading_day()}`")
st.sidebar.markdown(f"**Windows:** {config.WINDOWS}")
st.sidebar.markdown(f"**Sentiment model:** {config.FINBERT_MODEL_NAME} (+ GDELT tone fallback)")
st.sidebar.markdown(f"**Forecast horizon:** {config.PRED_HORIZON}d")
st.sidebar.markdown(f"**Sentiment momentum lookback:** {config.SENTIMENT_MOM_LOOKBACK}d")
st.sidebar.markdown(
    f"**Weights:** Sentiment {config.WEIGHT_SENTIMENT:.0%} | "
    f"Persistence {config.WEIGHT_PERSISTENCE:.0%} | "
    f"Fit {config.WEIGHT_FIT:.0%}")

HF_TOKEN    = config.HF_TOKEN
OUTPUT_REPO = config.OUTPUT_REPO


@st.cache_data(ttl=3600)
def list_repo_files():
    fs = HfFileSystem(token=HF_TOKEN or None)
    try:
        files = [f["name"] for f in fs.ls(f"datasets/{OUTPUT_REPO}",
                                           detail=True, recursive=True)
                 if f["type"] == "file"]
        return files, None
    except Exception as e:
        return [], str(e)


def find_latest(files, prefix):
    matches = sorted([f for f in files if f.endswith(".json") and prefix in f],
                     reverse=True)
    return matches[0] if matches else None


@st.cache_data(ttl=3600)
def load_json(path):
    fs = HfFileSystem(token=HF_TOKEN or None)
    try:
        with fs.open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        return {"error": str(e)}


@st.cache_data(ttl=3600)
def load_cache_stats():
    """Quick look at the sentiment cache itself — how much data, how recent,
    how much came from FinBERT vs. the GDELT-tone fallback."""
    fs = HfFileSystem(token=HF_TOKEN or None)
    try:
        with fs.open(f"datasets/{OUTPUT_REPO}/{config.SENTIMENT_CACHE_FILENAME}", "rb") as f:
            cache_df = pd.read_parquet(f)
        return cache_df, None
    except Exception as e:
        return None, str(e)


files, list_error = list_repo_files()

with st.expander("🔧 Debug: what the dashboard sees on HuggingFace", expanded=bool(list_error)):
    st.markdown(f"**Repo:** `{OUTPUT_REPO}`  ·  **Token set:** {'yes' if bool(HF_TOKEN) else 'no'}")
    if list_error:
        st.error(f"Could not list repo files: {list_error}")
    else:
        st.write(f"{len(files)} file(s) found:")
        st.code("\n".join(sorted(files)) if files else "(empty)")

with st.expander("📊 Sentiment cache status", expanded=False):
    cache_df, cache_err = load_cache_stats()
    if cache_err:
        st.info(f"Could not load cache stats: {cache_err}")
    elif cache_df is not None and not cache_df.empty:
        cache_df["date"] = pd.to_datetime(cache_df["date"])
        n_finbert = int(cache_df["n_finbert"].sum()) if "n_finbert" in cache_df else 0
        n_fallback = int(cache_df["n_fallback"].sum()) if "n_fallback" in cache_df else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Cache rows", len(cache_df))
        c2.metric("Date range", f"{cache_df['date'].min().date()} → {cache_df['date'].max().date()}")
        total = n_finbert + n_fallback
        pct_finbert = (n_finbert / total * 100) if total > 0 else 0
        c3.metric("FinBERT-scored articles", f"{n_finbert} ({pct_finbert:.0f}%)",
                   help="Remainder used the GDELT-native tone fallback (article fetch failed or capped)")

tab1_path = find_latest(files, "news_sentiment_engine_2")
tab2_path = find_latest(files, "news_sentiment_engine_windows_")

if not tab1_path:
    if list_error:
        st.error("Could not reach HuggingFace to look for results (see 🔧 Debug above).")
    else:
        st.error(
            "Connected to HuggingFace successfully, but no file matching "
            "`news_sentiment_engine_2*.json` was found. Run "
            "`update_sentiment_cache.py` then `trainer.py` — the cache "
            "needs at least some matched news before scores can be produced."
        )
    st.stop()

data1 = load_json(tab1_path)
if "error" in data1:
    st.error(f"Error loading data: {data1['error']}")
    st.stop()

data2      = load_json(tab2_path) if tab2_path else None
universes1 = data1["universes"]
universes2 = data2["universes"] if data2 and "error" not in data2 else None

st.sidebar.markdown(f"**Run date:** `{data1.get('run_date','?')}`")

tab1, tab2 = st.tabs(["🏆 Best Window per ETF", "🔍 Explore by Window"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.header("🏆 Top ETFs — Sentiment Signal")

    with st.expander("News Sentiment Methodology", expanded=True):
        st.markdown("""
**Data source: GDELT 2.0 GKG** — free, U.S.-government-funded, updated
every 15 minutes, archives back to 2015. No API key, plain HTTP flat-file
downloads. Every article already has a GDELT-computed tone score
(V2Tone), used as an always-available fallback.

**Sentiment scoring: FinBERT**, a BERT-scale model fine-tuned for
financial sentiment — CPU-feasible, no GPU, no per-call API cost, run
directly on a capped number of fetched article pages per ticker per day.
Article-webpage fetching is inherently fragile (paywalls, bot-blocking,
dead links); any article FinBERT can't score keeps its GDELT-native tone
score instead — this is graceful degradation, not a failure state.

**Bucketing** (which ticker an article applies to) uses GDELT's own
metadata — source name, extracted themes, URL — via keyword matching, not
full-text search. MACRO keywords (Fed, rates, inflation, etc.) apply to
every ticker; sector keywords apply only to their specific ETF.

**Model: closed-form OLS regression**, deliberately not a neural network —
a sentiment-return relationship, if one exists, is low-dimensional and
doesn't justify more model complexity than the data supports:
