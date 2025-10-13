# data_ingestion/yahoo_sec.py
"""
DataAgent helpers:
- Collects & cleans data from Yahoo Finance (news, basic price context)
- (Optional) SEC filings via public submissions JSON (no API key) – simplified
Outputs:
- A list of LangChain Documents suitable for vector DB ingestion
"""

from __future__ import annotations
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, timezone
import os
import json
import re

import yfinance as yf
import pandas as pd
import requests
from langchain_core.documents import Document

USER_AGENT = os.environ.get("SEC_USER_AGENT", "DealIntelBot/1.0 contact@example.com")

MERGER_KEYWORDS = [
    "merger", "acquisition", "acquire", "acquiring", "acquired",
    "buyout", "buy-out", "takeover", "business combination",
    "SPAC", "special purpose acquisition", "divestiture", "spin-off",
    "strategic transaction", "definitive agreement", "tender offer",
]

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def fetch_yahoo_news(tickers: List[str], limit_per_ticker: int = 15) -> List[Dict[str, Any]]:
    """
    Pull recent news items using yfinance's lightweight Ticker.news field.
    Returns a list of dicts with keys: ticker, title, link, publisher, providerPublishTime.
    """
    out: List[Dict[str, Any]] = []
    for t in tickers:
        try:
            tk = yf.Ticker(t)
            news = tk.news or []
            for item in news[:limit_per_ticker]:
                out.append({
                    "ticker": t,
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "publisher": item.get("publisher", ""),
                    "published": item.get("providerPublishTime", 0),
                })
        except Exception:
            # Ignore failures per ticker
            continue
    return out

def fetch_prices_snapshot(tickers: List[str]) -> Dict[str, Any]:
    """
    Get last close + 5d % change for context, not for signals.
    """
    ctx: Dict[str, Any] = {}
    for t in tickers:
        try:
            df = yf.download(t, period="7d", interval="1d", progress=False, auto_adjust=True)
            if df is None or df.empty:
                continue
            close = df["Close"].dropna()
            last = float(close.iloc[-1])
            ch5 = float((last / close.iloc[0] - 1.0)) if len(close) > 1 else 0.0
            ctx[t] = {"last": last, "chg5d": ch5}
        except Exception:
            continue
    return ctx

def _sec_company_submissions(cik: str) -> Optional[Dict[str, Any]]:
    """
    Download SEC submissions for a CIK using the public JSON endpoint.
    NOTE: CIK should be zero-padded to 10 digits. We accept raw and pad here.
    """
    cik10 = cik.zfill(10)
    url = f"https://data.sec.gov/submissions/CIK{cik10}.json"
    try:
        resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
        if resp.status_code == 200:
            return resp.json()
    except Exception:
        return None
    return None

def fetch_sec_filings(ciks: List[str], form_filters: List[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
    """
    Pull a simplified set of recent filings metadata per CIK.
    Default form filter: 8-K and 425 often capture deal announcements.
    """
    if form_filters is None:
        form_filters = ["8-K", "425", "DEFM14A", "SC TO-T", "F-4", "S-4"]

    items: List[Dict[str, Any]] = []
    for cik in ciks:
        data = _sec_company_submissions(cik)
        if not data:
            continue
        recent = data.get("filings", {}).get("recent", {})
        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        docs = recent.get("primaryDocDescription", [])
        for form, dt, desc in zip(forms, dates, docs):
            if form not in form_filters:
                continue
            items.append({
                "cik": cik,
                "form": form,
                "date": dt,
                "description": desc or "",
                "source": "SEC",
                "link": f"https://www.sec.gov/edgar/browse/?CIK={cik}",
            })
            if len(items) >= limit:
                break
    return items

def _contains_deal_keywords(text: str) -> bool:
    text = text.lower()
    return any(k in text for k in MERGER_KEYWORDS)

def build_documents_from_sources(
    yahoo_news: List[Dict[str, Any]],
    prices_ctx: Dict[str, Any],
    sec_items: List[Dict[str, Any]] = None
) -> List[Document]:
    """
    Convert raw items into LangChain Documents with normalized metadata for retrieval.
    """
    docs: List[Document] = []
    # Yahoo news
    for n in yahoo_news:
        published_ts = int(n.get("published", 0))
        published_iso = datetime.fromtimestamp(published_ts, tz=timezone.utc).isoformat() if published_ts else _now_iso()
        text = n.get("title") or ""
        is_dealish = _contains_deal_keywords(text)
        meta = {
            "source": "yahoo_news",
            "ticker": n.get("ticker"),
            "publisher": n.get("publisher"),
            "link": n.get("link"),
            "published": published_iso,
            "is_dealish": is_dealish,
        }
        docs.append(Document(page_content=text, metadata=meta))

    # SEC filings
    if sec_items:
        for it in sec_items:
            text = f"{it.get('form','')} {it.get('date','')}: {it.get('description','')}"
            is_dealish = _contains_deal_keywords(text) or it.get("form") in {"425", "S-4", "F-4", "DEFM14A", "SC TO-T"}
            meta = {
                "source": "sec",
                "cik": it.get("cik"),
                "form": it.get("form"),
                "date": it.get("date"),
                "link": it.get("link"),
                "is_dealish": bool(is_dealish),
            }
            docs.append(Document(page_content=text, metadata=meta))

    # Lightweight “context docs” from prices
    for t, ctx in prices_ctx.items():
        text = f"{t} 5d change: {ctx['chg5d']:.2%}, last: {ctx['last']:.2f}"
        meta = {"source": "price_snapshot", "ticker": t}
        docs.append(Document(page_content=text, metadata=meta))

    return docs
