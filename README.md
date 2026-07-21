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
