"""Real German bank-offer scraper (thebanks.eu), append-only snapshot store.

This is the proprietary data asset: point-in-time snapshots of advertised German
mortgage offers per bank. We NEVER overwrite previous values (section 19); each
run appends rows stamped with a source hash so a historical bank spread
    BankSpread(b,t) = Rate(b,t) - MarketBenchmark(t)
can be reconstructed exactly.

Refactored from the repo's ``mortgage_percentage_scraper.ipynb``.
"""

from __future__ import annotations

import hashlib
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup

URL = "https://thebanks.eu/compare-banking-products/mortgage-loans/Germany"
HEADERS = {"User-Agent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/139.0.0.0 Safari/537.36")}
SNAPSHOT_STORE = Path("data/raw/scraper/bank_offer_snapshot.parquet")


def _clean(t: str | None) -> str | None:
    return re.sub(r"\s+", " ", t).strip() if t else None


def _nums(t: str | None) -> list[float]:
    if not t:
        return []
    return [float(x.replace(",", "")) for x in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", t)]


def _bank_id(name: str | None) -> str | None:
    if not name:
        return None
    s = name.lower()
    for a, b in {"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss"}.items():
        s = s.replace(a, b)
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_") or None


_EXCLUDED_NAMES = {"Customer Group", "Amount", "Effective Rate", "Term",
                   "View Details", "View Details +", "Apply for Loan", "Germany", "DE"}


def _extract_bank_name(container) -> str | None:
    for a in container.find_all("a"):
        if "/banks/" in a.get("href", ""):
            t = _clean(a.get_text(" ", strip=True))
            if t:
                return t
    for tag in container.find_all(["h1", "h2", "h3", "h4", "strong", "b"]):
        t = _clean(tag.get_text(" ", strip=True))
        if not t or t in _EXCLUDED_NAMES or re.fullmatch(r"[\d\s.,%€EUR-]+", t):
            continue
        if len(t) > 2:
            return t
    return None


def _find_offer_container(marker):
    """Walk up from a 'Customer Group' marker to the smallest block that holds
    Customer Group + Effective Rate + Term and a resolvable bank name."""
    parent, candidates = marker, []
    for _ in range(1, 12):
        parent = parent.parent
        if parent is None:
            break
        text = _clean(parent.get_text(" ", strip=True)) or ""
        if not text or len(text) > 5000:
            continue
        if (re.search(r"Customer Group", text, re.I) and
                re.search(r"Effective Rate", text, re.I) and
                re.search(r"Term", text, re.I)):
            candidates.append(parent)
    for cand in candidates:
        name = _extract_bank_name(cand)
        if name:
            return cand, name
    return None, None


def scrape_offers(url: str = URL, timeout: int = 30) -> pd.DataFrame:
    """Scrape current German mortgage offers (one row per offer).

    Uses the proven thebanks.eu structure: 'Customer Group' markers anchor each
    offer; we walk up to the offer container and regex the rate/term/amount.
    """
    resp = requests.get(url, headers=HEADERS, timeout=timeout)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    source_hash = hashlib.sha256(resp.content).hexdigest()[:16]
    now = datetime.now(timezone.utc).isoformat()

    offers = []
    for marker in soup.find_all(string=re.compile(r"^\s*Customer Group\s*$", re.I)):
        container, bank_name = _find_offer_container(marker)
        if container is None:
            continue
        text = _clean(container.get_text(" ", strip=True)) or ""
        rate = re.search(r"Effective Rate[s]?\s*([\d.]+)\s*-\s*([\d.]+)\s*%", text, re.I)
        term = re.search(r"Term\s*(\d+)\s*-\s*(\d+)\s*years", text, re.I)
        amount = re.search(r"([\d,]+(?:\.\d+)?)\s*-\s*([\d,]+(?:\.\d+)?)\s*EUR", text, re.I)
        if not rate:
            continue
        offers.append({
            "bank_id": _bank_id(bank_name),
            "bank_name": bank_name,
            "country": "Germany",
            "effective_rate_min_pct": float(rate.group(1)),
            "effective_rate_max_pct": float(rate.group(2)),
            "term_min_years": int(term.group(1)) if term else None,
            "term_max_years": int(term.group(2)) if term else None,
            "amount_min": float(amount.group(1).replace(",", "")) if amount else None,
            "amount_max": float(amount.group(2).replace(",", "")) if amount else None,
            "source_url": url,
            "source_hash": source_hash,
            "snapshot_timestamp": now,
        })
    df = pd.DataFrame(offers)
    if df.empty:
        return df
    return df.dropna(subset=["bank_id"]).drop_duplicates(
        subset=["bank_id", "effective_rate_min_pct", "effective_rate_max_pct"]
    ).reset_index(drop=True)


def append_snapshot(df: pd.DataFrame, store: Path = SNAPSHOT_STORE) -> Path:
    """Append-only: never overwrite prior snapshots."""
    store.parent.mkdir(parents=True, exist_ok=True)
    if store.exists():
        df = pd.concat([pd.read_parquet(store), df], ignore_index=True)
    df.to_parquet(store, index=False)
    return store


def load_snapshots(store: Path = SNAPSHOT_STORE) -> pd.DataFrame:
    return pd.read_parquet(store) if store.exists() else pd.DataFrame()


def scrape_and_store(retries: int = 2) -> pd.DataFrame:
    """Scrape and append a snapshot (best-effort; returns what was scraped)."""
    last = None
    for _ in range(retries + 1):
        try:
            df = scrape_offers()
            if len(df):
                append_snapshot(df)
                return df
        except Exception as exc:  # noqa: BLE001
            last = exc
            time.sleep(2)
    print(f"[scraper] WARN could not scrape offers: {last}")
    return load_snapshots()
