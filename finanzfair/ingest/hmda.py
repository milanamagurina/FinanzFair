"""Real realized-pricing surface from HMDA (CFPB/FFIEC Data Browser, no key).

HMDA is the most comprehensive public US mortgage-pricing dataset. We use it as
a *research prior / empirical pricing surface*, never as German ground truth
(see the cross-country caveat): millions of real loans with interest rate,
rate spread (APR - APOR), income, loan amount, DTI, loan term, lender (LEI),
loan purpose and property value.

The surface answers the core empirical question that replaces invented labels:

    "For borrowers with comparable observable characteristics, what distribution
     of rates did they actually obtain?"  ->  R_floor(Q10), R_median, R_Q90.
"""

from __future__ import annotations

import gzip
import io
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HMDA_CSV = "https://ffiec.cfpb.gov/v2/data-browser-api/view/csv"

# Columns we keep from the 99-column file; all are real HMDA fields.
KEEP = [
    "activity_year", "state_code", "lei", "loan_purpose", "occupancy_type",
    "loan_type", "interest_rate", "rate_spread", "loan_amount", "income",
    "debt_to_income_ratio", "loan_term", "property_value", "loan_to_value_ratio",
]

LOAN_PURPOSE = {1: "purchase", 2: "home_improvement", 31: "refinance",
                32: "cash_out_refinance", 4: "other", 5: "na"}


def fetch_hmda(years: list[int], states: list[str], actions: tuple[int, ...] = (1,),
               cache_dir: str | Path = "data/raw/hmda", use_cache: bool = True,
               timeout: int = 180) -> pd.DataFrame:
    """Download (or load cached) real HMDA loan-level rows for the given scope.

    ``actions=(1,)`` keeps originated loans only. Returns the raw kept columns;
    call :func:`clean_hmda` to get a modelling-ready frame.
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames = []
    for year in years:
        tag = f"{year}_{'-'.join(states)}_{'-'.join(map(str, actions))}"
        cache_file = cache / f"hmda_{tag}.parquet"
        if use_cache and cache_file.exists():
            frames.append(pd.read_parquet(cache_file))
            continue
        params = {
            "years": year,
            "states": ",".join(states),
            "actions_taken": ",".join(map(str, actions)),
        }
        r = requests.get(HMDA_CSV, params=params, timeout=timeout,
                         headers={"Accept-Encoding": "gzip"})
        r.raise_for_status()
        raw = r.content
        text = gzip.decompress(raw).decode() if raw[:2] == b"\x1f\x8b" else r.text
        df = pd.read_csv(io.StringIO(text), low_memory=False)
        df = df[[c for c in KEEP if c in df.columns]].copy()
        df["_fetched_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        df.to_parquet(cache_file, index=False)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean_hmda(df: pd.DataFrame, min_rate: float = 1.0, max_rate: float = 12.0) -> pd.DataFrame:
    """Coerce HMDA to a clean modelling frame with derived, real features."""
    d = df.copy()
    num = ["interest_rate", "rate_spread", "loan_amount", "income",
           "loan_term", "property_value", "loan_to_value_ratio"]
    for c in num:
        if c in d:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # DTI arrives as strings incl. ranges like "20%-<30%"; map to numeric midpoints.
    d["dti"] = _parse_dti(d.get("debt_to_income_ratio"))
    # Income in HMDA is thousands of USD/year.
    d["income_eur_proxy"] = d["income"] * 1000.0
    # Real LTV: prefer reported ratio, else derive from loan/property value.
    ltv = d.get("loan_to_value_ratio")
    derived = 100.0 * d["loan_amount"] / d["property_value"]
    d["ltv"] = pd.to_numeric(ltv, errors="coerce").fillna(derived) if ltv is not None else derived
    d["loan_purpose_label"] = pd.to_numeric(d.get("loan_purpose"), errors="coerce").map(LOAN_PURPOSE)
    d["term_years"] = d["loan_term"] / 12.0

    d = d[(d["interest_rate"].between(min_rate, max_rate))
          & (d["ltv"].between(1, 100))
          & (d["dti"].between(1, 70))
          & (d["income_eur_proxy"].between(10_000, 2_000_000))
          & (d["loan_amount"] > 10_000)]
    # Winsorise the benchmark-relative spread: APR-APOR is rarely outside this
    # band, and the tails are dominated by points/credits noise we do not model.
    d = d[d["rate_spread"].notna()]
    d["rate_spread"] = d["rate_spread"].clip(-1.0, 3.0)
    return d.reset_index(drop=True)


def _parse_dti(s: pd.Series | None) -> pd.Series:
    if s is None:
        return pd.Series(dtype=float)
    def one(v):
        if pd.isna(v):
            return np.nan
        t = str(v).replace("%", "").replace("<", "").replace(">", "").strip()
        if "-" in t:
            a, b = t.split("-")[:2]
            try:
                return (float(a) + float(b)) / 2.0
            except ValueError:
                return np.nan
        try:
            return float(t)
        except ValueError:
            return np.nan
    return s.map(one)
