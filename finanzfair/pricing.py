"""Empirical pricing surface — the REAL half of the final-rate model.

We model the *benchmark-relative* rate (HMDA ``rate_spread`` = APR - APOR), which
is far more transferable across markets than the absolute US rate level (see the
cross-country caveat). The German absolute rate is then reconstructed as

    R_initial(X, bank, t) = ECB_DE_benchmark(fixation, t)   [REAL market level]
                          + bank_spread(bank, t)            [REAL scraped spread]
                          + borrower_spread(X)              [REAL HMDA shape]

and the empirically *aggressive-but-plausible* floor (for the Delta_max cap) is

    R_floor(X, bank, t)  = benchmark + min_bank_spread + spread_Q10(X).

So "how low could a comparable borrower realistically get?" is answered from real
dispersion, never invented. Quantile regressors give Q10/Q50/Q90 of the borrower
spread; predictions are sorted to prevent quantile crossing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from .config import QUANTILES

# Borrower features used by the surface (all real, all point-in-time safe).
FEATURES = ["ltv", "dti", "log_income", "log_loan", "term_years"]
CAT_FEATURES = ["loan_purpose_label"]


def _design(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["ltv"] = df["ltv"]
    X["dti"] = df["dti"]
    X["log_income"] = np.log(df["income_eur_proxy"].clip(lower=1))
    X["log_loan"] = np.log(df["loan_amount"].clip(lower=1))
    X["term_years"] = df["term_years"]
    lp = df.get("loan_purpose_label")
    X["loan_purpose_label"] = (lp.astype("category").cat.codes
                               if lp is not None else 0)
    return X


@dataclass
class EmpiricalPricingSurface:
    """Quantile model of benchmark-relative mortgage spread, fit on real HMDA."""

    quantiles: tuple[float, ...] = QUANTILES
    learning_rate: float = 0.08
    max_iter: int = 350
    max_depth: int | None = 4
    min_leaf: int = 120

    def fit(self, hmda: pd.DataFrame) -> "EmpiricalPricingSurface":
        X = _design(hmda)
        y = pd.to_numeric(hmda["rate_spread"], errors="coerce")
        ok = y.notna() & np.isfinite(y)
        X, y = X[ok], y[ok]
        self._models = {}
        for q in self.quantiles:
            m = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, learning_rate=self.learning_rate,
                max_iter=self.max_iter, max_depth=self.max_depth,
                min_samples_leaf=self.min_leaf, l2_regularization=1.0,
                random_state=7)
            m.fit(X, y)
            self._models[q] = m
        self.population_median_spread_ = float(np.median(y))
        self.spread_dispersion_ = float(y.quantile(0.5) - y.quantile(0.1))
        self.n_train_ = int(len(y))
        return self

    def predict_spread_quantiles(self, df: pd.DataFrame) -> pd.DataFrame:
        X = _design(df)
        preds = {f"spread_q{int(q*100)}": self._models[q].predict(X)
                 for q in self.quantiles}
        out = pd.DataFrame(preds, index=df.index)
        # Enforce monotone quantiles row-wise (fix any crossing).
        cols = sorted(out.columns, key=lambda c: int(c.split("q")[1]))
        out[cols] = np.sort(out[cols].to_numpy(), axis=1)
        return out


def assemble_initial_rates(profiles: pd.DataFrame, surface: EmpiricalPricingSurface,
                           bank_spreads: pd.DataFrame, benchmark_rate: float,
                           fixation_benchmarks: dict[int, float] | None = None,
                           center_band: tuple[float, float] = (-1.0, 1.5),
                           quote_sigma_cap: float = 0.18,
                           room_scale: float = 0.45, room_cap: float = 0.40) -> pd.DataFrame:
    """Cross borrowers x banks -> real initial-rate quantiles per (borrower, bank).

    The level decomposes into real components:
        R_initial_median = benchmark(fixation, t)      [REAL ECB German rate]
                         + bank_spread(bank)           [REAL scraped spread]
                         + borrower_center(X)          [REAL HMDA quality shape]
    where ``borrower_center`` is this profile's median benchmark-relative spread
    minus the population median (clipped to a realistic band). The quote interval
    is a NARROW uncertainty band (``quote_sigma``), kept separate from the
    negotiation room (``neg_room``), so the final-rate interval does not double
    count population dispersion as if it were negotiable.
    """
    sp = surface.predict_spread_quantiles(profiles)
    base = profiles[["borrower_id", "fixed_period_years"]].reset_index(drop=True)
    base = pd.concat([base, sp.reset_index(drop=True)], axis=1)

    lo, hi = center_band
    base["borrower_center"] = (base["spread_q50"] - surface.population_median_spread_).clip(lo, hi)
    # Narrow quote uncertainty (half of the 10-90 spread, capped).
    base["quote_sigma"] = (0.5 * (base["spread_q90"] - base["spread_q10"])).clip(0.05, quote_sigma_cap)
    # Empirical negotiation room = a fraction of the room from median toward
    # aggressive (Q10) pricing for comparable borrowers. The HMDA Q50-Q10 spread
    # also carries points/credits noise, so ``room_scale`` discounts it to the
    # realistic German within-bank negotiation band; ``room_cap`` is a hard cap.
    base["neg_room"] = (room_scale * (base["spread_q50"] - base["spread_q10"])).clip(0.03, room_cap)

    def bench(fp):
        if fixation_benchmarks:
            return fixation_benchmarks.get(int(fp), benchmark_rate)
        return benchmark_rate
    base["benchmark"] = base["fixed_period_years"].map(bench)

    rows = base.merge(bank_spreads, how="cross")
    rows["r_initial_q50"] = rows["benchmark"] + rows["bank_spread_bp"] + rows["borrower_center"]
    rows["r_initial_q10"] = rows["r_initial_q50"] - rows["quote_sigma"]
    rows["r_initial_q90"] = rows["r_initial_q50"] + rows["quote_sigma"]
    return rows


def empirical_floor(initial_rates: pd.DataFrame) -> pd.DataFrame:
    """Per (borrower, bank): aggressive-but-plausible floor and the empirical
    Delta_max cap (section 11).

    Delta_max(i,b) = empirical room-to-aggressive pricing for comparable borrowers
    (``neg_room`` from the pricing surface). R_floor = R_initial_median - Delta_max.
    This caps any synthetic reduction at what real HMDA dispersion supports and
    cannot be exceeded by the negotiation layer.
    """
    df = initial_rates.copy()
    df["delta_max"] = df["neg_room"].clip(lower=0.0)
    df["r_floor"] = df["r_initial_q50"] - df["delta_max"]
    return df
