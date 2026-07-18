"""
news_sentiment_engine.py — News Sentiment Engine
========================================================

Theory
------
Given a daily sentiment time series per ticker (built from GDELT + FinBERT
via gdelt_client.py / sentiment_scorer.py), this engine fits a simple
closed-form OLS regression — deliberately NOT a neural network — since a
sentiment-return relationship, if it exists at all, is low-dimensional and
doesn't justify more model complexity than the data can support:

    forward_return_t = a + b*sentiment_t + c*sentiment_momentum_t + d*news_volume_t + eps_t

fit via ordinary least squares (exact linear algebra via np.linalg.lstsq),
matching the same "exact math, no approximation" philosophy as EDMD and
GP-Vol elsewhere in this suite.

**Honest caveat.** The empirical literature on text sentiment predicting
MULTI-DAY-FORWARD returns (as opposed to same-day or intraday reactions)
is genuinely mixed and often weak. `fit_quality` here deserves more
skepticism than the equivalent diagnostic in most other engines in this
suite — this is exactly the kind of claim that should be validated
empirically (e.g. via an out-of-sample ablation, as was built for TNN)
rather than assumed from in-sample R^2 alone.

**Score construction**

    score = 0.50*sentiment_signal + 0.25*sentiment_persistence*sign(sentiment_signal) + 0.25*fit_quality

| Component              | Meaning                                                          |
|--------------------------|----------------------------------------------------------------------|
| sentiment_signal         | OLS-predicted forward return from today's sentiment features       |
| sentiment_persistence    | Has sentiment been consistently one-directional recently, or is today a one-off blip? |
| fit_quality              | R^2 of the OLS regression on its own training data                 |
"""

import numpy as np
import pandas as pd
from typing import List

import config


def build_sentiment_features(sentiment_long: pd.DataFrame, tickers: List[str]):
    """
    sentiment_long: columns [date, ticker, avg_sentiment, article_count].
    Returns (sent_wide, count_wide): date-indexed DataFrames, one column
    per ticker, forward-filled over gaps (days with zero matched articles).
    """
    if sentiment_long.empty:
        return pd.DataFrame(), pd.DataFrame()

    sent_wide = sentiment_long.pivot_table(index="date", columns="ticker", values="avg_sentiment")
    count_wide = sentiment_long.pivot_table(index="date", columns="ticker", values="article_count")

    sent_wide = sent_wide.reindex(columns=tickers)
    count_wide = count_wide.reindex(columns=tickers).fillna(0.0)
    sent_wide = sent_wide.ffill()   # carry sentiment forward over no-news days

    return sent_wide, count_wide


def fit_ols(X: np.ndarray, y: np.ndarray):
    """X: (n, p) design matrix (already includes an intercept column). y: (n,).
    Returns (beta, fit_quality)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = np.sum((y - y_hat) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    fit_quality = float(1.0 - np.clip(ss_res / (ss_tot + 1e-10), 0.0, 1.0))
    return beta, fit_quality


def compute_news_sentiment_scores(
    prices:          pd.DataFrame,
    sentiment_long:  pd.DataFrame,
    tickers:         List[str],
    window:          int,
) -> pd.DataFrame:
    """
    Fit an OLS sentiment-return regression per ETF and extract a
    sentiment-driven signal. Returns a DataFrame of score + diagnostics
    (cross-sectional z-scored on the composite).
    """
    cols = ["score", "sentiment_signal", "sentiment_persistence", "fit_quality",
            "avg_sentiment_today", "news_volume_today"]
    avail = [t for t in tickers if t in prices.columns]
    if not avail or sentiment_long.empty:
        return pd.DataFrame(columns=cols)

    H, mom_lb = config.PRED_HORIZON, config.SENTIMENT_MOM_LOOKBACK
    sent_wide, count_wide = build_sentiment_features(sentiment_long, avail)
    if sent_wide.empty:
        return pd.DataFrame(columns=cols)

    raw_scores = {}

    for ticker in avail:
        if ticker not in sent_wide.columns:
            continue

        ps = prices[ticker].dropna()
        common_idx = ps.index.intersection(sent_wide.index)
        if len(common_idx) < window + H + mom_lb + 10:
            continue

        common_idx = common_idx[-(window + H):]
        ps_a = ps.loc[common_idx]
        sent_a = sent_wide.loc[common_idx, ticker].fillna(0.0).values
        count_a = count_wide.loc[common_idx, ticker].fillna(0.0).values

        log_ret = np.log(ps_a / ps_a.shift(1)).values
        T = len(log_ret)

        n = T - H - mom_lb
        if n < 20:
            continue

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

        if len(targets) < 20:
            continue

        rows_sent = np.array(rows_sent)
        rows_mom  = np.array(rows_mom)
        rows_vol  = np.array(rows_vol)
        targets   = np.array(targets)

        vol_mu, vol_sd = rows_vol.mean(), rows_vol.std() + 1e-8
        vol_norm = (rows_vol - vol_mu) / vol_sd

        X = np.column_stack([np.ones(len(targets)), rows_sent, rows_mom, vol_norm])
        try:
            beta, fit_quality = fit_ols(X, targets)
        except Exception as e:
            print(f"    Failed {ticker}: {e}")
            continue

        s_today_now = sent_a[-1]
        s_prior_now = sent_a[-mom_lb - 1:-1].mean() if len(sent_a) > mom_lb else s_today_now
        momentum_now = s_today_now - s_prior_now
        vol_now_norm = (count_a[-1] - vol_mu) / vol_sd

        x_today = np.array([1.0, s_today_now, momentum_now, vol_now_norm])
        sentiment_signal = float(x_today @ beta)

        recent_sent = sent_a[-mom_lb:]
        sign_today = np.sign(s_today_now) if s_today_now != 0 else 0.0
        if sign_today == 0.0:
            sentiment_persistence = 0.0
        else:
            sentiment_persistence = float(np.mean(np.sign(recent_sent) == sign_today))

        sign = np.sign(sentiment_signal) if sentiment_signal != 0 else 1.0
        composite = (
            config.WEIGHT_SENTIMENT   * sentiment_signal
            + config.WEIGHT_PERSISTENCE * sentiment_persistence * sign
            + config.WEIGHT_FIT          * fit_quality
        )

        raw_scores[ticker] = {
            "composite": composite,
            "sentiment_signal": sentiment_signal,
            "sentiment_persistence": sentiment_persistence,
            "fit_quality": fit_quality,
            "avg_sentiment_today": float(s_today_now),
            "news_volume_today": float(count_a[-1]),
        }
        print(f"    {ticker}: sentiment_signal={sentiment_signal:.5f}  "
              f"persistence={sentiment_persistence:.2f}  fit={fit_quality:.3f}")

    if not raw_scores:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(raw_scores).T
    mu_s, std_s = df["composite"].mean(), df["composite"].std()
    if std_s < 1e-10:
        df["score"] = 0.0
    else:
        df["score"] = (df["composite"] - mu_s) / std_s
    return df[cols]
