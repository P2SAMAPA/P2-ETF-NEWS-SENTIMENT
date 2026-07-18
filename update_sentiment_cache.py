"""
update_sentiment_cache.py — GDELT + FinBERT ingestion, separate from scoring
================================================================================

This is a standalone ingestion step, deliberately separated from trainer.py's
scoring step — mirroring standard data-engineering practice of keeping
ingestion and modeling pipelines independent. Run this first; trainer.py
then reads whatever is in the cache and never fetches GDELT itself.

INCREMENTAL, RESUMABLE DESIGN: the cache is saved and re-uploaded to HF
after every SAVE_EVERY_N_DAYS days processed, not just once at the end.
This matters because an initial backfill (config.BACKFILL_DAYS, default
400 days) can be slow — hundreds of days x GDELT_SAMPLES_PER_DAY file
downloads, plus FinBERT scoring — and could plausibly exceed a single
GitHub Actions run's time budget. If a run times out partway through, the
next run picks up exactly where the cache's last date left off, rather
than losing all progress and starting over.

IMPORTANT: the live GDELT fetch and FinBERT model download could not be
tested from the sandbox this was built in (network egress restrictions —
see gdelt_client.py and sentiment_scorer.py docstrings). Verify this
actually runs successfully once deployed somewhere with normal internet
access before relying on its output.
"""

import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, HfApi

import config
import gdelt_client
import sentiment_scorer

SAVE_EVERY_N_DAYS = 10


def _all_tickers():
    seen = []
    for tickers in config.UNIVERSES.values():
        for t in tickers:
            if t not in seen:
                seen.append(t)
    return seen


def load_cache() -> pd.DataFrame:
    cols = ["date", "ticker", "avg_sentiment", "article_count", "n_finbert", "n_fallback"]
    if not config.HF_TOKEN:
        return pd.DataFrame(columns=cols)
    try:
        path = hf_hub_download(
            repo_id=config.OUTPUT_REPO,
            filename=config.SENTIMENT_CACHE_FILENAME,
            repo_type="dataset",
            token=config.HF_TOKEN,
        )
        df = pd.read_parquet(path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    except Exception as e:
        print(f"No existing cache found ({e}) — starting a fresh backfill.")
        return pd.DataFrame(columns=cols)


def save_cache(df: pd.DataFrame):
    Path("cache").mkdir(exist_ok=True)
    local_path = Path(f"cache/{config.SENTIMENT_CACHE_FILENAME}")
    df.to_parquet(local_path, index=False)

    if not config.HF_TOKEN:
        print("HF_TOKEN not set — cache saved locally only, not uploaded.")
        return

    api = HfApi(token=config.HF_TOKEN)
    try:
        api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=config.SENTIMENT_CACHE_FILENAME,
            repo_id=config.OUTPUT_REPO,
            repo_type="dataset",
            commit_message=f"Update sentiment cache through {df['date'].max()}",
        )
        print(f"✅ Cache uploaded ({len(df)} rows, through {df['date'].max()})")
    except Exception as e:
        print(f"❌ Cache upload failed: {e}")


def fetch_and_score_day(day: date, tickers: list) -> list:
    """Returns a list of row-dicts for one day, one row per ticker with any
    matched articles that day. Tickers with zero matches produce no row
    (build_sentiment_features forward-fills across gaps like this)."""
    buckets = gdelt_client.fetch_gdelt_day_bucketed(day, tickers)

    rows = []
    for ticker, articles in buckets.items():
        if not articles:
            continue
        scores, n_finbert, n_fallback = sentiment_scorer.score_articles(articles)
        rows.append({
            "date": pd.Timestamp(day),
            "ticker": ticker,
            "avg_sentiment": float(sum(scores) / len(scores)),
            "article_count": len(scores),
            "n_finbert": n_finbert,
            "n_fallback": n_fallback,
        })
    return rows


def main():
    tickers = _all_tickers()
    cache = load_cache()

    if cache.empty:
        start_day = date.today() - timedelta(days=config.BACKFILL_DAYS)
        print(f"No cache found — backfilling from {start_day} ({config.BACKFILL_DAYS} days). "
              f"This will be slow; progress is saved incrementally.")
    else:
        last_date = cache["date"].max().date()
        start_day = last_date + timedelta(days=1)
        print(f"Cache found through {last_date} — fetching new days from {start_day}.")

    end_day = date.today() - timedelta(days=1)   # never fetch "today" (incomplete)
    if start_day > end_day:
        print("Cache already up to date. Nothing to do.")
        return

    all_new_rows = []
    days_since_save = 0
    current = start_day

    while current <= end_day:
        t0 = time.time()
        print(f"\n--- {current} ---")
        try:
            rows = fetch_and_score_day(current, tickers)
            all_new_rows.extend(rows)
            print(f"  {len(rows)} ticker-days with matched news "
                  f"({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"  Failed to process {current}: {e} — skipping this day")

        days_since_save += 1
        if days_since_save >= SAVE_EVERY_N_DAYS:
            if all_new_rows:
                cache = pd.concat([cache, pd.DataFrame(all_new_rows)], ignore_index=True)
                save_cache(cache)
                all_new_rows = []
            days_since_save = 0

        current += timedelta(days=1)

    if all_new_rows:
        cache = pd.concat([cache, pd.DataFrame(all_new_rows)], ignore_index=True)
        save_cache(cache)

    print(f"\n=== Sentiment cache update complete: {len(cache)} total rows ===")


if __name__ == "__main__":
    main()
