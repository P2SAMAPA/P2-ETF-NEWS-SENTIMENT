"""
ablation_news_sentiment.py — Out-of-Sample Ablation for the News Sentiment Engine
==================================================================================
Mirrors the discipline of the Temporal Neighbourhood Network engine's
`ablation_graph_value.py`: don't trust in-sample R² (`fit_quality`) on its
own — measure whether the fitted OLS regression actually generalizes to
data it never saw.

**Why this exists.** The News Sentiment Engine's own README flags this as
its single biggest open caveat: the empirical literature on text sentiment
predicting MULTI-DAY-FORWARD returns is genuinely mixed and often weak, and
no out-of-sample validation had been built for this engine. This script is
that validation.

**Method — chronological train/test split, per ticker per window.**
For each (universe, ticker, window) combination:
  1. Build the same feature matrix the live engine builds (sentiment today,
     sentiment momentum, normalized news volume) over the trailing
     `window` days.
  2. Split it chronologically: the first ~70% of observations are TRAIN,
     the last ~30% are TEST. No shuffling — this is a time series, and
     shuffling would leak future information into the fit.
  3. Fit OLS on TRAIN only (volume normalization stats are also computed
     from TRAIN only, to avoid leaking TEST statistics into the features).
  4. Score the frozen TRAIN-fitted model against the untouched TEST
     targets. Report:
       - oos_r2            : out-of-sample R² vs. the test set's own mean
                              (negative means the model is worse than just
                              guessing the average return — a real
                              possibility, and an important one to see)
       - oos_correlation    : Pearson correlation between predicted and
                              realized forward returns on TEST
       - oos_hit_rate       : fraction of TEST points where the predicted
                              return's sign matched the realized return's
                              sign (0.5 = coin flip)
       - in_sample_fit_quality : the TRAIN-only R², for direct comparison
                              against the live engine's in-sample number

**How to read the results.** A window/ticker combination is only worth
trusting if oos_r2 is meaningfully positive and oos_hit_rate is
meaningfully above 0.5 — NOT just because in_sample_fit_quality looks
good. A large gap between in-sample fit and out-of-sample performance is
the signature of overfitting, not signal.

**Honest caveat, again.** Even a well-behaved out-of-sample split on ~1-2
years of daily data is still a small-sample estimate of a noisy
relationship. Treat this as a considerably better sanity check than raw
in-sample R², not as proof the sentiment-return relationship is real or
tradeable.

**A second honest caveat, specific to this ablation.** The forward-return
target (H=`config.PRED_HORIZON` days, sampled once per trading day) is an
overlapping window — each day's target shares H-1 of its H days with the
next day's target. Consecutive TEST-set targets are therefore strongly
autocorrelated, not independent draws. In testing against synthetic data,
this was observed to inflate `oos_hit_rate` even when the underlying
sentiment signal was pure noise, simply because a multi-week directional
run in the test period makes "always predict the same sign" look
accurate. Weight `oos_r2` and `oos_correlation` more heavily than
`oos_hit_rate` for this reason — they're less distorted by the overlap,
though not immune to it either. A larger TEST set (more calendar days)
somewhat dilutes this effect but does not eliminate it.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime

import config
import data_manager
import push_results
from news_sentiment_engine import build_sentiment_features, fit_ols
from trainer import load_sentiment_cache, convert_to_serializable

TRAIN_FRAC = 0.70
MIN_TOTAL_OBSERVATIONS = 40   # need enough on both sides of the split
MIN_TEST_OBSERVATIONS = 10


def ablate_ticker(
    prices: pd.DataFrame,
    sent_wide: pd.DataFrame,
    count_wide: pd.DataFrame,
    ticker: str,
    window: int,
    H: int,
    mom_lb: int,
    train_frac: float = TRAIN_FRAC,
):
    """
    Chronological train/test ablation for a single ticker at a single
    window. Returns a dict of diagnostics, or None if there isn't enough
    data to run a meaningful split.
    """
    if ticker not in prices.columns or ticker not in sent_wide.columns:
        return None

    ps = prices[ticker].dropna()
    common_idx = ps.index.intersection(sent_wide.index)
    if len(common_idx) < window + H + mom_lb + 10:
        return None

    common_idx = common_idx[-(window + H):]
    ps_a = ps.loc[common_idx]
    sent_a = sent_wide.loc[common_idx, ticker].fillna(0.0).values
    count_a = count_wide.loc[common_idx, ticker].fillna(0.0).values

    log_ret = np.log(ps_a / ps_a.shift(1)).values
    T = len(log_ret)

    rows_sent, rows_mom, rows_vol, targets = [], [], [], []
    for t in range(mom_lb, T - H):
        if np.isnan(log_ret[t]):
            continue
        s_today = sent_a[t]
        s_prior = sent_a[t - mom_lb:t].mean()
        momentum = s_today - s_prior
        vol_today = count_a[t]
        fwd = np.nanmean(log_ret[t + 1: t + 1 + H])
        if np.isnan(fwd):
            continue
        rows_sent.append(s_today)
        rows_mom.append(momentum)
        rows_vol.append(vol_today)
        targets.append(fwd)

    n = len(targets)
    if n < MIN_TOTAL_OBSERVATIONS:
        return None

    rows_sent = np.array(rows_sent)
    rows_mom = np.array(rows_mom)
    rows_vol = np.array(rows_vol)
    targets = np.array(targets)

    split = int(n * train_frac)
    n_test = n - split
    if split < MIN_TOTAL_OBSERVATIONS - MIN_TEST_OBSERVATIONS or n_test < MIN_TEST_OBSERVATIONS:
        return None

    train_sl = slice(0, split)
    test_sl = slice(split, n)

    y_train, y_test = targets[train_sl], targets[test_sl]

    # Normalize news volume using TRAIN stats only — using the full
    # sample's stats (including TEST) would leak test-set information
    # into the features the model is scored on.
    vol_mu = rows_vol[train_sl].mean()
    vol_sd = rows_vol[train_sl].std() + 1e-8
    vol_norm = (rows_vol - vol_mu) / vol_sd

    X = np.column_stack([np.ones(n), rows_sent, rows_mom, vol_norm])
    X_train, X_test = X[train_sl], X[test_sl]

    try:
        beta, in_sample_fit = fit_ols(X_train, y_train)
    except Exception:
        return None

    y_pred_test = X_test @ beta

    ss_res = np.sum((y_test - y_pred_test) ** 2)
    ss_tot = np.sum((y_test - y_test.mean()) ** 2)
    oos_r2 = float(1.0 - ss_res / (ss_tot + 1e-10))

    if np.std(y_pred_test) > 1e-12 and np.std(y_test) > 1e-12:
        oos_corr = float(np.corrcoef(y_pred_test, y_test)[0, 1])
    else:
        oos_corr = 0.0

    oos_hit_rate = float(np.mean(np.sign(y_pred_test) == np.sign(y_test)))

    return {
        "n_train": int(split),
        "n_test": int(n_test),
        "in_sample_fit_quality": float(in_sample_fit),
        "oos_r2": oos_r2,
        "oos_correlation": oos_corr,
        "oos_hit_rate": oos_hit_rate,
    }


def main():
    if not config.HF_TOKEN:
        print("HF_TOKEN not set"); return

    df = data_manager.load_master_data()
    sentiment_cache = load_sentiment_cache()

    today = datetime.now().strftime("%Y-%m-%d")

    if sentiment_cache.empty:
        print("Sentiment cache is empty — nothing to ablate. "
              "Run update_sentiment_cache.py and trainer.py first.")
        return

    print(f"Sentiment cache: {len(sentiment_cache)} rows, "
          f"through {sentiment_cache['date'].max().date()}")

    H, mom_lb = config.PRED_HORIZON, config.SENTIMENT_MOM_LOOKBACK

    all_ablation = {}

    for universe_name, tickers in config.UNIVERSES.items():
        print(f"\n=== Ablation — Universe: {universe_name} ===")

        prices = data_manager.prepare_prices(df, tickers)
        available_tickers = [t for t in tickers if t in prices.columns]

        if not available_tickers or prices.empty:
            print("  No price data")
            all_ablation[universe_name] = {"windows": {}}
            continue

        sent_wide, count_wide = build_sentiment_features(sentiment_cache, available_tickers)
        if sent_wide.empty:
            print("  No matched news for this universe yet")
            all_ablation[universe_name] = {"windows": {}}
            continue

        windows_out = {}

        for win in config.WINDOWS:
            print(f"\n  Window: {win}d")
            ticker_results = {}

            for ticker in available_tickers:
                result = ablate_ticker(
                    prices=prices,
                    sent_wide=sent_wide,
                    count_wide=count_wide,
                    ticker=ticker,
                    window=win,
                    H=H,
                    mom_lb=mom_lb,
                )
                if result is None:
                    continue
                ticker_results[ticker] = result
                print(f"    {ticker}: in-sample fit={result['in_sample_fit_quality']:.3f}  "
                      f"OOS R²={result['oos_r2']:.3f}  "
                      f"OOS corr={result['oos_correlation']:.3f}  "
                      f"OOS hit-rate={result['oos_hit_rate']:.2f}  "
                      f"(train={result['n_train']}, test={result['n_test']})")

            if not ticker_results:
                print("    No tickers had enough data for this window")
                continue

            windows_out[str(win)] = ticker_results

        # Per-ticker best window by TRUE out-of-sample R² (not in-sample fit) —
        # this is the number to look at instead of the live engine's
        # in-sample "best_window" when deciding which window to actually trust.
        best_oos_window = {}
        for win_str, ticker_results in windows_out.items():
            for ticker, rec in ticker_results.items():
                if ticker not in best_oos_window or rec["oos_r2"] > best_oos_window[ticker]["oos_r2"]:
                    best_oos_window[ticker] = {**rec, "window": int(win_str)}

        all_ablation[universe_name] = {
            "windows": windows_out,
            "best_oos_window": {
                t: {
                    "window": rec["window"],
                    "oos_r2": rec["oos_r2"],
                    "oos_correlation": rec["oos_correlation"],
                    "oos_hit_rate": rec["oos_hit_rate"],
                    "in_sample_fit_quality": rec["in_sample_fit_quality"],
                }
                for t, rec in best_oos_window.items()
            },
        }

    Path("results").mkdir(exist_ok=True)
    out_path = Path(f"results/news_sentiment_ablation_{today}.json")
    with open(out_path, "w") as f:
        json.dump(
            convert_to_serializable({"run_date": today, "universes": all_ablation}),
            f, indent=2,
        )

    push_results.push_daily_result(out_path)

    print(f"\n=== Ablation complete: {out_path.name} ===")


if __name__ == "__main__":
    main()
