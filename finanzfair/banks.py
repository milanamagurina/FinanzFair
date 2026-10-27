"""Bank panel from the REAL scraped German offers.

BankSpread(b, t) = advertised_effective_rate(b) - MarketBenchmark(t), computed
from append-only scraper snapshots. These are real German lenders and real
advertised rates; the spread is what the pricing surface adds on top of the
real ECB benchmark to get a bank-specific initial rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def bank_spreads_from_offers(offers: pd.DataFrame, benchmark_rate: float,
                             use: str = "mid") -> pd.DataFrame:
    """Compute per-bank spread (pp) vs the real market benchmark.

    ``use`` selects which advertised rate to anchor on: 'min' (most competitive),
    'mid' (midpoint of the advertised range), or 'max'.
    """
    d = offers.copy()
    if use == "min":
        rate = d["effective_rate_min_pct"]
    elif use == "max":
        rate = d["effective_rate_max_pct"]
    else:
        rate = (d["effective_rate_min_pct"] + d["effective_rate_max_pct"]) / 2.0
    d = d.assign(bank_rate=rate)
    agg = (d.groupby("bank_id")
             .agg(bank_name=("bank_name", "first"),
                  bank_rate=("bank_rate", "mean"),
                  rate_min=("effective_rate_min_pct", "min"),
                  rate_max=("effective_rate_max_pct", "max"),
                  n_offers=("bank_rate", "size"))
             .reset_index())
    agg["bank_spread_bp"] = agg["bank_rate"] - benchmark_rate
    agg["bank_rate_percentile"] = agg["bank_rate"].rank(pct=True)
    return agg[["bank_id", "bank_name", "bank_spread_bp", "bank_rate",
                "rate_min", "rate_max", "bank_rate_percentile", "n_offers"]]


def latest_offers(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Keep only the most recent snapshot per bank from the append-only store."""
    if snapshots.empty:
        return snapshots
    d = snapshots.sort_values("snapshot_timestamp")
    return d.groupby(["bank_id", "effective_rate_min_pct", "effective_rate_max_pct"],
                     as_index=False).last()


def data_density_penalty(bank_spreads: pd.DataFrame, floor: int = 1) -> pd.Series:
    """Penalty for banks with thin history (feeds the ranking + confidence)."""
    n = bank_spreads.set_index("bank_id")["n_offers"].clip(lower=floor)
    return (0.02 / np.sqrt(n)).rename("penalty")
