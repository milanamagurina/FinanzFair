"""Real market-state ingestion from the ECB Data Portal (SDMX REST, no key).

Returns the point-in-time market layer Market_t from *real* published series:
German new-business mortgage rates, the ECB policy (deposit facility) rate,
euro short-term rate (€STR), and the euro-area AAA government yield curve.

Every pull is cached append-only to ``data/raw/ecb/`` with the fetch timestamp
so the market snapshot is reproducible and point-in-time correct.
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import pandas as pd
import requests

ECB_BASE = "https://data-api.ecb.europa.eu/service/data"

# Real ECB series. Each entry: logical name -> (dataset, key, freq_hint).
SERIES = {
    # German MFI interest rates (MIR), loans to households for house purchase.
    "de_mortgage_rate_all_irf": ("MIR", "M.DE.B.A2C.AM.R.A.2250.EUR.N"),      # all fixation
    "de_mortgage_rate_over10y": ("MIR", "M.DE.B.A2C.P.R.A.2250.EUR.N"),        # >10y fixation
    "de_mortgage_rate_5to10y":  ("MIR", "M.DE.B.A2C.O.R.A.2250.EUR.N"),        # 5-10y fixation
    "de_mortgage_rate_1to5y":   ("MIR", "M.DE.B.A2C.I.R.A.2250.EUR.N"),        # 1-5y fixation
    "de_mortgage_rate_float":   ("MIR", "M.DE.B.A2C.F.R.A.2250.EUR.N"),        # <=1y fixation
    # Euro-area policy and money-market.
    "ecb_deposit_facility":     ("FM",  "B.U2.EUR.4F.KR.DFR.LEV"),
    "estr":                     ("EST", "B.EU000A2X2A25.WT"),
    # Euro-area AAA government spot yields (Bund-proxy curve).
    "ea_aaa_2y":                ("YC",  "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_2Y"),
    "ea_aaa_5y":                ("YC",  "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_5Y"),
    "ea_aaa_10y":               ("YC",  "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y"),
}


def _fetch_series(dataset: str, key: str, last_n: int | None = None,
                  start: str | None = None, timeout: int = 30) -> pd.DataFrame:
    """Fetch one SDMX series as tidy (time_period, obs_value). Parses by header."""
    url = f"{ECB_BASE}/{dataset}/{key}"
    params = {"format": "csvdata"}
    if last_n:
        params["lastNObservations"] = last_n
    if start:
        params["startPeriod"] = start
    r = requests.get(url, params=params, headers={"Accept": "text/csv"}, timeout=timeout)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text))
    # SDMX csvdata always carries TIME_PERIOD and OBS_VALUE columns.
    out = df[["TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={"TIME_PERIOD": "time_period", "OBS_VALUE": "obs_value"}
    )
    out["obs_value"] = pd.to_numeric(out["obs_value"], errors="coerce")
    return out.dropna(subset=["obs_value"]).reset_index(drop=True)


def fetch_market_series(names: list[str] | None = None, start: str = "2015-01",
                        cache_dir: str | Path = "data/raw/ecb",
                        use_cache: bool = True) -> dict[str, pd.DataFrame]:
    """Fetch (or load cached) real ECB series. Returns {name: tidy DataFrame}.

    Caching is append-only per fetch date so the snapshot is reproducible.
    """
    names = names or list(SERIES)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d")
    out: dict[str, pd.DataFrame] = {}
    for name in names:
        dataset, key = SERIES[name]
        cache_file = cache / f"{name}__{stamp}.csv"
        if use_cache and cache_file.exists():
            out[name] = pd.read_csv(cache_file)
            continue
        try:
            df = _fetch_series(dataset, key, start=start)
            df.to_csv(cache_file, index=False)
            out[name] = df
        except Exception as exc:  # noqa: BLE001 - surface which series failed
            print(f"[ecb] WARN {name} ({dataset}/{key}) failed: {exc}")
    return out


def build_market_panel(series: dict[str, pd.DataFrame] | None = None,
                       **kwargs) -> pd.DataFrame:
    """Assemble a monthly market panel with changes / vol features (Market_t).

    One row per month; columns are the real series plus derived point-in-time
    features (curve slope, 3m/12m changes, 12m volatility). All features use
    only information available at or before each month (no look-ahead).
    """
    series = series or fetch_market_series(**kwargs)
    frames = []
    for name, df in series.items():
        s = df.copy()
        # Normalise to monthly period (daily series -> month-end last value).
        s["month"] = pd.PeriodIndex(pd.to_datetime(s["time_period"]), freq="M").astype(str)
        s = s.groupby("month", as_index=False)["obs_value"].last().rename(
            columns={"obs_value": name})
        frames.append(s.set_index("month"))
    panel = pd.concat(frames, axis=1).sort_index()
    panel = panel.ffill()  # carry last published value forward (point-in-time)

    bench = "de_mortgage_rate_all_irf"
    if "ea_aaa_10y" in panel and "ea_aaa_2y" in panel:
        panel["curve_2y10y"] = panel["ea_aaa_10y"] - panel["ea_aaa_2y"]
    if bench in panel:
        panel["mortgage_chg_3m"] = panel[bench].diff(3)
        panel["mortgage_chg_12m"] = panel[bench].diff(12)
        panel["mortgage_vol_12m"] = panel[bench].rolling(12, min_periods=3).std()
    panel = panel.reset_index().rename(columns={"index": "month"})
    panel["as_of_month"] = panel["month"]
    return panel


def latest_market_row(panel: pd.DataFrame | None = None) -> pd.Series:
    """Most recent fully-populated market state (for a live analysis request)."""
    panel = panel if panel is not None else build_market_panel()
    return panel.dropna(subset=["de_mortgage_rate_all_irf"]).iloc[-1]
