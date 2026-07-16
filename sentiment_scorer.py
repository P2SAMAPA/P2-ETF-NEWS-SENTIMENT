"""
sentiment_scorer.py — FinBERT sentiment scoring with graceful GDELT fallback
==============================================================================

FinBERT (ProsusAI/finbert) is a BERT-scale model fine-tuned for financial
sentiment classification — CPU-feasible, no GPU required, no per-call API
cost, self-hosted like the rest of this suite's models (unlike calling an
LLM API for sentiment, which would work but breaks the zero-marginal-cost
pattern every other engine follows).

Article-webpage fetching is inherently fragile (paywalls, bot-blocking,
dead links, JS-rendered pages that return no readable text). This module
NEVER lets a failed fetch or a failed model call crash the pipeline: any
article that can't be scored by FinBERT keeps its GDELT-native V2Tone
score instead, rescaled onto a comparable range. A "sources" field on the
output tracks which path was actually used, for transparency about how
much of a given day's sentiment came from refined FinBERT scoring vs. the
fallback.

IMPORTANT: FinBERT model loading (from HuggingFace) and live article
fetching could not be tested from the sandbox this was built in (neither
huggingface.co nor arbitrary news domains were reachable from that
environment's network egress allowlist). The scoring math and fallback
logic are unit-tested against synthetic/mocked inputs; the actual model
load and HTTP fetch paths should be verified once deployed somewhere with
normal internet access.
"""

from typing import List, Tuple, Optional

import numpy as np
import requests

import config

_finbert_tokenizer = None
_finbert_model = None


def _load_finbert():
    """Lazy singleton load — only pay the (large) model-load cost once per run."""
    global _finbert_tokenizer, _finbert_model
    if _finbert_model is not None:
        return _finbert_tokenizer, _finbert_model

    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch

    print(f"    [sentiment] Loading {config.FINBERT_MODEL_NAME} ...")
    _finbert_tokenizer = AutoTokenizer.from_pretrained(config.FINBERT_MODEL_NAME)
    _finbert_model = AutoModelForSequenceClassification.from_pretrained(config.FINBERT_MODEL_NAME)
    _finbert_model.eval()
    return _finbert_tokenizer, _finbert_model


def fetch_article_text(url: str) -> Optional[str]:
    """
    Best-effort fetch of an article's visible text. Returns None on any
    failure (bad status, timeout, no parseable text) — callers must treat
    None as "use the fallback score", never as an error to propagate.
    """
    try:
        resp = requests.get(url, timeout=config.REQUEST_TIMEOUT_SECONDS,
                             headers={"User-Agent": "Mozilla/5.0"})
        resp.raise_for_status()
    except Exception:
        return None

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(resp.content, "html.parser")
        for tag in soup(["script", "style", "nav", "header", "footer"]):
            tag.decompose()
        text = " ".join(soup.get_text(separator=" ").split())
        return text[:5000] if len(text) > 200 else None
    except Exception:
        return None


def score_text_finbert(text: str) -> Optional[float]:
    """Returns P(positive) - P(negative) in [-1,1], or None on failure."""
    try:
        import torch
        tokenizer, model = _load_finbert()
        inputs = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=config.FINBERT_MAX_LENGTH)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].numpy()
        # ProsusAI/finbert label order: 0=positive, 1=negative, 2=neutral
        return float(probs[0] - probs[1])
    except Exception as e:
        print(f"    [sentiment] FinBERT scoring failed: {e}")
        return None


def score_articles(articles: List[Tuple[float, str]], max_finbert: int = None):
    """
    articles: [(gdelt_tone, url), ...] for one ticker on one day.
    Returns (scores: List[float] in [-1,1], n_finbert: int, n_fallback: int).
    Only the first `max_finbert` articles (by input order) get an attempted
    FinBERT pass; the rest use the GDELT tone fallback directly (bounding
    per-day runtime regardless of how many articles matched).
    """
    max_finbert = max_finbert or config.MAX_ARTICLES_PER_TICKER_PER_DAY
    scores = []
    n_finbert, n_fallback = 0, 0

    for i, (tone, url) in enumerate(articles):
        fallback_score = float(np.clip(tone / config.GDELT_TONE_SCALE, -1.0, 1.0))

        if i < max_finbert:
            text = fetch_article_text(url)
            if text is not None:
                s = score_text_finbert(text)
                if s is not None:
                    scores.append(s)
                    n_finbert += 1
                    continue
            scores.append(fallback_score)
            n_fallback += 1
        else:
            scores.append(fallback_score)
            n_fallback += 1

    return scores, n_finbert, n_fallback
