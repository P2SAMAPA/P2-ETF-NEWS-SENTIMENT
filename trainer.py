import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

import config
import data_manager
import push_results
from news_sentiment_engine import compute_news_sentiment_scores
from huggingface_hub import hf_hub_download


def convert_to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_serializable(v) for v in obj]
    return obj


def load_sentiment_cache() -> pd.DataFrame:
    """Reads the cache built by update_sentiment_cache.py — this script never
    fetches GDELT itself, keeping ingestion and scoring cleanly separate."""
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
        print(f"Could not load sentiment cache: {e}")
        print("Run update_sentiment_cache.py first.")
        return pd.DataFrame(columns=["date", "ticker", "avg_sentiment", "article_count"])


def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set"); return

    df               = data_manager.load_master_data()
    sentiment_cache  = load_sentiment_cache()
    today            = datetime.now().strftime("%Y-%m-%d")

    if sentiment_cache.empty:
        print("Sentiment cache is empty — nothing to score. "
              "Run update_sentiment_cache.py first (it can take a while on the initial backfill).")
        return

    print(f"Sentiment cache: {len(sentiment_cache)} rows, "
          f"through {sentiment_cache['date'].max().date()}")

    all_results = {}
    all_windows = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Universe: {universe_name} (News Sentiment Engine) ===")

        prices            = data_manager.prepare_prices(df, tickers)
        available_tickers = [t for t in tickers if t in prices.columns]

        if not available_tickers or prices.empty:
            print("  No price data")
            all_results[universe_name] = {"top_etfs": [], "full_scores": {}}
            all_windows[universe_name] = {"windows": {}}
            continue

        best_per_etf   = {}
        window_results = {}

        for win in config.WINDOWS:
            min_bars = win + config.PRED_HORIZON + config.SENTIMENT_MOM_LOOKBACK + 30
            if len(prices) < min_bars:
                print(f"  Skipping window {win}d")
                continue

            print(f"\n  Window: {win}d")

            try:
                scores_df = compute_news_sentiment_scores(
                    prices          = prices,
                    sentiment_long  = sentiment_cache,
                    tickers         = available_tickers,
                    window          = win,
                )
            except Exception as e:
                print(f"  Failed: {e}")
                import traceback; traceback.print_exc()
                continue

            if scores_df.empty:
                print("  No scores (likely no matched news for these tickers yet)")
                continue

            score_records = {}
            for t, row in scores_df.iterrows():
                if np.isnan(row["score"]):
                    continue
                score_records[t] = {
                    "score":                   float(row["score"]),
                    "sentiment_signal":        float(row["sentiment_signal"]),
                    "sentiment_persistence":   float(row["sentiment_persistence"]),
                    "fit_quality":             float(row["fit_quality"]),
                    "avg_sentiment_today":     float(row["avg_sentiment_today"]),
                    "news_volume_today":       float(row["news_volume_today"]),
                    "days_since_last_news":    float(row["days_since_last_news"]),
                }

            sorted_scores = sorted(score_records.items(), key=lambda x: x[1]["score"], reverse=True)
            print(f"  Top 3: {[t for t, _ in sorted_scores[:3]]}")

            window_results[win] = score_records

            for etf, rec in score_records.items():
                if etf not in best_per_etf or abs(rec["score"]) > abs(best_per_etf[etf]["score"]):
                    best_per_etf[etf] = {**rec, "window": win}

        if not best_per_etf:
            all_results[universe_name] = {"top_etfs": [], "full_scores": {}, "run_date": today}
            all_windows[universe_name] = {"windows": {}, "run_date": today}
            continue

        sorted_etfs = sorted(best_per_etf.items(), key=lambda x: x[1]["score"], reverse=True)
        top_etfs    = [
            {
                "ticker": t,
                "sentiment_score": rec["score"],
                "best_window": int(rec["window"]),
                "sentiment_signal": rec["sentiment_signal"],
                "sentiment_persistence": rec["sentiment_persistence"],
                "fit_quality": rec["fit_quality"],
                "avg_sentiment_today": rec["avg_sentiment_today"],
                "news_volume_today": rec["news_volume_today"],
                "days_since_last_news": rec["days_since_last_news"],
            }
            for t, rec in sorted_etfs[:config.TOP_N]
        ]
        full_scores = {
            t: {
                "score": rec["score"], "best_window": int(rec["window"]),
                "sentiment_signal": rec["sentiment_signal"],
                "sentiment_persistence": rec["sentiment_persistence"],
                "fit_quality": rec["fit_quality"],
                "avg_sentiment_today": rec["avg_sentiment_today"],
                "news_volume_today": rec["news_volume_today"],
                "days_since_last_news": rec["days_since_last_news"],
            }
            for t, rec in sorted_etfs
        }
        all_results[universe_name] = {
            "top_etfs": top_etfs, "full_scores": full_scores, "run_date": today
        }
        print(f"\n  Final top {config.TOP_N}: {[e['ticker'] for e in top_etfs]}")

        windows_tab2 = {}
        for win, score_records in window_results.items():
            sw = sorted(score_records.items(), key=lambda x: x[1]["score"], reverse=True)
            windows_tab2[str(win)] = {
                "top_etfs": [
                    {
                        "ticker": t, "sentiment_score": rec["score"],
                        "sentiment_signal": rec["sentiment_signal"],
                        "sentiment_persistence": rec["sentiment_persistence"],
                        "fit_quality": rec["fit_quality"],
                    }
                    for t, rec in sw[:config.TOP_N]
                ],
                "full_ranking": [
                    [t, rec["score"], rec["sentiment_signal"],
                     rec["sentiment_persistence"], rec["fit_quality"]]
                    for t, rec in sw
                ],
            }
        all_windows[universe_name] = {"windows": windows_tab2, "run_date": today}

    Path("results").mkdir(exist_ok=True)

    tab1_path = Path(f"results/news_sentiment_engine_{today}.json")
    with open(tab1_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_results}), f, indent=2)

    tab2_path = Path(f"results/news_sentiment_engine_windows_{today}.json")
    with open(tab2_path, "w") as f:
        json.dump(convert_to_serializable({"run_date": today, "universes": all_windows}), f, indent=2)

    push_results.push_daily_result(tab1_path)
    push_results.push_daily_result(tab2_path)

    print(f"\n=== News Sentiment Engine complete: {tab1_path.name}, {tab2_path.name} ===")


if __name__ == "__main__":
    main()
