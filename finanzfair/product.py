"""Production-like prediction: Reduction hurdle model, final-rate distribution,
Top-K ranking. This is the V1 product stack (sections 35, 44, 54).

Two models compose the user-facing output:
  1. Initial Offer  -> P(R_initial)   (quantiles from the empirical surface).
  2. Reduction      -> P(Delta | X, bank, market, competition)  (hurdle model).
Final rate R_final = R_initial - Delta is obtained by Monte Carlo over both,
and banks are ranked by a risk-adjusted score, NOT by nominal discount.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor)

from .config import QUANTILES, rng

# Features the reduction model sees (continuous counteroffer strength, not binary).
REDUCTION_FEATURES = ["counteroffer_gap_bp", "delta_max", "credit_quality",
                      "LTV", "DTI", "log_income", "bank_code"]


def _reduction_design(panel: pd.DataFrame, use_counteroffer: bool = True) -> pd.DataFrame:
    """Design matrix for the reduction model.

    ``use_counteroffer=False`` drops the continuous counteroffer-strength feature
    to build the NAIVE challenger for the selection-bias stress test: without it,
    the model cannot separate the counteroffer effect from borrower quality.
    """
    X = pd.DataFrame(index=panel.index)
    if use_counteroffer:
        X["counteroffer_gap_bp"] = panel["counteroffer_gap_bp"]
    X["delta_max"] = panel["delta_max"]
    X["credit_quality"] = panel["credit_quality"]
    X["LTV"] = panel["LTV"]
    X["DTI"] = panel["DTI"]
    X["log_income"] = np.log(panel["gross_household_income"].clip(lower=1))
    X["bank_code"] = panel["bank_id"].astype("category").cat.codes
    return X


@dataclass
class ReductionHurdleModel:
    """P(Delta>0) classifier + quantile/mean regressors for Delta|Delta>0.

    Trained on the semi-synthetic panel (label_source must be 'synthetic'). The
    composed distribution is (1-p) point-mass at 0 plus p * positive distribution.
    """

    quantiles: tuple[float, ...] = QUANTILES
    use_counteroffer: bool = True

    def fit(self, panel: pd.DataFrame) -> "ReductionHurdleModel":
        assert (panel["label_source"] == "synthetic").all(), \
            "ReductionHurdleModel V0 trains on synthetic labels only (no silent mixing)."
        X = _reduction_design(panel, self.use_counteroffer)
        y = panel["true_delta"].to_numpy(float)
        pos = y > 1e-6

        self.clf_ = HistGradientBoostingClassifier(
            learning_rate=0.08, max_iter=300, max_depth=4,
            min_samples_leaf=80, random_state=7).fit(X, pos.astype(int))

        Xp, yp = X[pos], y[pos]
        self.qreg_ = {}
        for q in self.quantiles:
            self.qreg_[q] = HistGradientBoostingRegressor(
                loss="quantile", quantile=q, learning_rate=0.08, max_iter=300,
                max_depth=4, min_samples_leaf=60, random_state=7).fit(Xp, yp)
        self.mean_ = HistGradientBoostingRegressor(
            learning_rate=0.08, max_iter=300, max_depth=4, min_samples_leaf=60,
            random_state=7).fit(Xp, yp)
        self._bank_cats = panel["bank_id"].astype("category").cat.categories
        return self

    def predict(self, panel: pd.DataFrame) -> pd.DataFrame:
        X = _reduction_design(panel, self.use_counteroffer)
        p = self.clf_.predict_proba(X)[:, 1]
        qp = {q: np.clip(self.qreg_[q].predict(X), 0, None) for q in self.quantiles}
        mean_pos = np.clip(self.mean_.predict(X), 0, None)
        out = pd.DataFrame(index=panel.index)
        out["p_reduction"] = p
        for q in self.quantiles:
            out[f"delta_q{int(q*100)}"] = qp[q]
        # Expected reduction E[Delta] = p * E[Delta | Delta>0].
        out["expected_delta"] = p * mean_pos
        out["_mean_pos"] = mean_pos
        return out

    def predict_mu(self, panel: pd.DataFrame, c_grid_bp: np.ndarray) -> np.ndarray:
        """Counterfactual dose-response mu_hat(c, x) over a counteroffer grid.

        Returns array [n_rows, n_grid] of E[Delta | C=c, x] for MISE evaluation.
        """
        mus = np.empty((len(panel), len(c_grid_bp)))
        for j, c in enumerate(c_grid_bp):
            tmp = panel.copy()
            tmp["counteroffer_gap_bp"] = c
            X = _reduction_design(tmp, self.use_counteroffer)
            mus[:, j] = self.clf_.predict_proba(X)[:, 1] * np.clip(self.mean_.predict(X), 0, None)
        return mus


def _sample_from_quantiles(q10: np.ndarray, q50: np.ndarray, q90: np.ndarray,
                           n: int, g: np.random.Generator) -> np.ndarray:
    """Sample via a monotone piecewise-linear inverse-CDF through the 3 quantiles.

    Returns array [len(q50), n]. Tails are linearly extrapolated and the spread
    is symmetric-safe (handles q10==q90 degenerate rows).
    """
    q10 = np.asarray(q10, float)[:, None]
    q50 = np.asarray(q50, float)[:, None]
    q90 = np.asarray(q90, float)[:, None]
    u = g.random((q50.shape[0], n))
    lo = q50 + (u - 0.5) / 0.4 * (q50 - q10)   # for u in [0.1, 0.5]
    hi = q50 + (u - 0.5) / 0.4 * (q90 - q50)   # for u in [0.5, 0.9]
    return np.where(u < 0.5, lo, hi)


def final_rate_distribution(initial: pd.DataFrame, reduction: pd.DataFrame,
                            n_mc: int = 4000, seed: int = 0) -> pd.DataFrame:
    """Monte Carlo R_final = R_initial - Delta per (borrower, bank).

    ``initial`` has r_initial_q10/q50/q90; ``reduction`` has p_reduction and
    delta_q10/q50/q90. Returns E[R_final], SD, and final-rate quantiles.
    """
    g = rng(seed)
    n = len(initial)
    ri = _sample_from_quantiles(initial["r_initial_q10"].to_numpy(),
                                initial["r_initial_q50"].to_numpy(),
                                initial["r_initial_q90"].to_numpy(), n_mc, g)
    dpos = _sample_from_quantiles(reduction["delta_q10"].to_numpy(),
                                  reduction["delta_q50"].to_numpy(),
                                  reduction["delta_q90"].to_numpy(), n_mc, g)
    dpos = np.clip(dpos, 0, None)
    occur = g.random((n, n_mc)) < reduction["p_reduction"].to_numpy()[:, None]
    delta = np.where(occur, dpos, 0.0)
    rfinal = ri - delta

    out = pd.DataFrame(index=initial.index)
    out["expected_final"] = rfinal.mean(axis=1)
    out["sd_final"] = rfinal.std(axis=1)
    out["final_q10"] = np.quantile(rfinal, 0.10, axis=1)
    out["final_q50"] = np.quantile(rfinal, 0.50, axis=1)
    out["final_q90"] = np.quantile(rfinal, 0.90, axis=1)
    out["expected_reduction"] = (ri - rfinal).mean(axis=1)
    out["p_reduction_ge_10bp"] = (delta >= 0.10).mean(axis=1)
    return out


def rank_banks(panel: pd.DataFrame, lam: float = 0.5, top_k: int = 5,
               penalties: pd.Series | None = None) -> pd.DataFrame:
    """Risk-adjusted Top-K ranking per borrower: Score = E[R_final] + lam*SD + pen.

    Lower score is better (lower expected final rate). ``panel`` must contain
    bank_id, expected_final, sd_final and the display columns.
    """
    df = panel.copy()
    df["score"] = df["expected_final"] + lam * df["sd_final"]
    if penalties is not None:
        df["score"] = df["score"] + df["bank_id"].map(penalties).fillna(0.0)
    df = df.sort_values(["borrower_id", "score"])
    df["rank"] = df.groupby("borrower_id").cumcount() + 1
    return df[df["rank"] <= top_k].reset_index(drop=True)


def format_recommendation(ranked_one_borrower: pd.DataFrame) -> pd.DataFrame:
    """Human-readable Top-K table matching the product spec's example output."""
    d = ranked_one_borrower.sort_values("rank")
    show = pd.DataFrame({
        "Rank": d["rank"],
        "Bank": d["bank_id"],
        "Initial (median)": d["r_initial_q50"].round(2).astype(str) + "%",
        "Expected reduction": (d["expected_reduction"] * 100).round(1).astype(str) + " bp",
        "Likely reduction": ((d["delta_q10"] * 100).round(1).astype(str) + "–"
                             + (d["delta_q90"] * 100).round(1).astype(str) + " bp"),
        "P(Δ>0)": (d["p_reduction"] * 100).round(0).astype(int).astype(str) + "%",
        "Expected final": d["expected_final"].round(2).astype(str) + "%",
        "80% final CI": (d["final_q10"].round(2).astype(str) + "–"
                         + d["final_q90"].round(2).astype(str) + "%"),
    })
    return show.reset_index(drop=True)
