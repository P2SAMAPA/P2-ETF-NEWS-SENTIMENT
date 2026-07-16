"""
gdelt_client.py — GDELT 2.0 GKG ingestion
================================================

GDELT (Global Database of Events, Language, and Tone) is a free, U.S.-
government-funded global news database updated every 15 minutes, with
archives back to 2015 — no API key, no account, plain HTTP flat-file
downloads:

    http://data.gdeltproject.org/gdeltv2/<timestamp>.gkg.csv.zip

Each GKG file already includes a per-article tone score (V2Tone) computed
by GDELT's own NLP pipeline, WITHOUT needing the article's full text — this
is used as the always-available fallback sentiment source.

BUCKETING (which ticker(s) does an article apply to) uses GDELT's own
metadata — source name, extracted themes, and URL — via keyword substring
matching, and does NOT require fetching the article webpage. Fetching the
actual article text (for FinBERT scoring — see sentiment_scorer.py) is a
separate, later step, attempted only for a capped number of top articles
per ticker per day, with automatic fallback to V2Tone if the fetch fails.

IMPORTANT: this module's live HTTP behavior could not be tested from the
sandbox this was built in (data.gdeltproject.org was not reachable from
that environment's network egress allowlist). The parsing logic is
defensive and unit-testable against synthetic GKG-shaped data, but the
actual live fetch should be verified once deployed somewhere with normal
internet access.
"""

import io
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests

import config

# GKG 2.1 field indices (tab-separated). Only the fields actually used are
# named here; the file has more columns than this.
GKG_COL_DATE       = 1
GKG_COL_SOURCE     = 3
GKG_COL_URL        = 4
GKG_COL_THEMES     = 8
GKG_COL_V2TONE     = 15


def _gdelt_timestamps_for_day(date: "datetime.date", n_samples: int) -> List[str]:
    """GDELT publishes a file every 15 minutes (96/day). Sample n_samples of
    them, evenly spaced, to bound bandwidth/runtime for a daily aggregate."""
    n_samples = max(1, min(n_samples, 96))
    step = 96 // n_samples
    timestamps = []
    for i in range(n_samples):
        slot = i * step
        hour = (slot * 15) // 60
        minute = (slot * 15) % 60
        ts = datetime(date.year, date.month, date.day, hour, minute)
        timestamps.append(ts.strftime("%Y%m%d%H%M%S"))
    return timestamps


def _parse_v2tone(v2tone_field: str) -> float:
    """V2Tone field is comma-separated: tone,pos,neg,polarity,... — first value."""
    try:
        return float(v2tone_field.split(",")[0])
    except (ValueError, AttributeError, IndexError):
        return 0.0


def fetch_and_parse_gkg_file(timestamp: str) -> pd.DataFrame:
    """
    Download and parse one GKG file. Returns a DataFrame with columns
    [date, source, url, themes, tone]. Returns an empty DataFrame (never
    raises) on any network/parsing failure — this is a real fragility point
    (dead links, format changes, transient outages) and callers should never
    have a single bad file take down a whole day's ingestion.
    """
    url = f"{config.GDELT_BASE_URL}/{timestamp}.gkg.csv.zip"
    cols = ["date", "source", "url", "themes", "tone"]
    try:
        resp = requests.get(url, timeout=config.REQUEST_TIMEOUT_SECONDS)
        resp.raise_for_status()
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            inner_name = zf.namelist()[0]
            with zf.open(inner_name) as f:
                raw = f.read().decode("utf-8", errors="replace")
    except Exception as e:
        print(f"    [gdelt] fetch failed for {timestamp}: {e}")
        return pd.DataFrame(columns=cols)

    rows = []
    for line in raw.split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        max_idx = max(GKG_COL_DATE, GKG_COL_SOURCE, GKG_COL_URL, GKG_COL_THEMES, GKG_COL_V2TONE)
        if len(parts) <= max_idx:
            continue
        tone = _parse_v2tone(parts[GKG_COL_V2TONE])
        rows.append({
            "date": parts[GKG_COL_DATE],
            "source": parts[GKG_COL_SOURCE],
            "url": parts[GKG_COL_URL],
            "themes": parts[GKG_COL_THEMES],
            "tone": tone,
        })

    return pd.DataFrame(rows, columns=cols)


def match_tickers(row: pd.Series, keyword_map: Dict[str, List[str]]) -> List[str]:
    """Which tickers does this article's metadata match, via keyword substring
    search over source name + themes + URL (no webpage fetch needed)."""
    blob = f"{row['source']} {row['themes']} {row['url']}".lower()
    matched = []
    for ticker, keywords in keyword_map.items():
        if any(kw.lower() in blob for kw in keywords):
            matched.append(ticker)
    return matched


def fetch_gdelt_day_bucketed(date, tickers: List[str]) -> Dict[str, List[Tuple[float, str]]]:
    """
    Fetch a sample of one day's GDELT files and bucket articles to tickers.
    Returns {ticker: [(tone, url), ...]} — MACRO_KEYWORDS matches apply to
    every ticker; SECTOR_KEYWORDS matches apply only to their specific ticker.
    Never raises — a day with zero successfully-fetched files just returns
    empty buckets, and callers (news_sentiment_engine.py) handle that as
    "no data today" rather than crashing the whole run.
    """
    buckets: Dict[str, List[Tuple[float, str]]] = {t: [] for t in tickers}

    keyword_map = {t: list(config.SECTOR_KEYWORDS.get(t, [])) for t in tickers}

    timestamps = _gdelt_timestamps_for_day(date, config.GDELT_SAMPLES_PER_DAY)
    n_files_ok = 0

    for ts in timestamps:
        df = fetch_and_parse_gkg_file(ts)
        if df.empty:
            continue
        n_files_ok += 1

        macro_blob_mask = df.apply(
            lambda r: any(kw.lower() in f"{r['source']} {r['themes']} {r['url']}".lower()
                          for kw in config.MACRO_KEYWORDS),
            axis=1,
        )
        for _, row in df[macro_blob_mask].iterrows():
            for t in tickers:
                buckets[t].append((row["tone"], row["url"]))

        for _, row in df.iterrows():
            matched = match_tickers(row, keyword_map)
            for t in matched:
                buckets[t].append((row["tone"], row["url"]))

    if n_files_ok == 0:
        print(f"    [gdelt] WARNING: no GKG files successfully fetched for {date}")

    return buckets
