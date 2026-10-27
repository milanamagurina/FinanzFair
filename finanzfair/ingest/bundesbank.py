"""Real German rate dispersion from the Bundesbank SDMX API (no key).

The Bundesbank publishes, per initial-rate-fixation bucket, not just the average
new-business mortgage rate but also the *lowest* and *highest* rate of the
spread, and separate series for renegotiated housing loans. That dispersion is
exactly what anchors the empirical feasible pricing room:

    R_market_min(t, fixation) <= R_market_avg <= R_market_max(t, fixation)

so a synthetic reduction can never exceed what real German dispersion supports.

Series keys live in the BBK SDMX catalogue; pass your own if these move. The
fetcher is generic so new keys need no code change.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

BBK_BASE = "https://api.statistiken.bundesbank.de/rest/data"

# Known-useful BBK series (housing loans to households, new business).
# Average effective rate by initial rate fixation; dispersion where published.
SERIES = {
    "de_housing_avg_over10y": "BBK01/SUD118",     # placeholder key; override as needed
    "de_housing_avg_5to10y":  "BBK01/SUD117",
    "de_housing_renegotiated": "BBK01/SUD130",
}


def fetch_bbk_series(key: str, start: str = "2015-01", timeout: int = 30) -> pd.DataFrame:
    """Fetch one Bundesbank series as tidy (time_period, obs_value)."""
    url = f"{BBK_BASE}/{key}"
    r = requests.get(url, params={"format": "csv", "startPeriod": start},
                     timeout=timeout)
    r.raise_for_status()
    # BBK CSV uses ';' and often a metadata preamble; find the data table.
    text = r.text
    df = pd.read_csv(io.StringIO(text), sep=";", engine="python", header=0)
    # First column is the date, second the value in BBK's long CSV layout.
    cols = list(df.columns)
    out = df[[cols[0], cols[1]]].copy()
    out.columns = ["time_period", "obs_value"]
    out["obs_value"] = pd.to_numeric(
        out["obs_value"].astype(str).str.replace(",", "."), errors="coerce")
    return out.dropna().reset_index(drop=True)


def fetch_all(series: dict[str, str] | None = None, start: str = "2015-01",
              cache_dir: str | Path = "data/raw/bbk", use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch (or cache) the configured Bundesbank series. Failures are skipped
    with a warning so a stale key never breaks the pipeline."""
    series = series or SERIES
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d")
    out: dict[str, pd.DataFrame] = {}
    for name, key in series.items():
        cache_file = cache / f"{name}__{stamp}.csv"
        if use_cache and cache_file.exists():
            out[name] = pd.read_csv(cache_file)
            continue
        try:
            df = fetch_bbk_series(key, start=start)
            df.to_csv(cache_file, index=False)
            out[name] = df
        except Exception as exc:  # noqa: BLE001
            print(f"[bbk] WARN {name} ({key}) failed: {exc} "
                  f"(use ECB MIR German series as the fallback market layer)")
    return out
