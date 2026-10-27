"""Evaluation: distributional quality, calibration, ranking, and the
counterfactual benchmark that only semi-synthetic ground truth makes possible.

Factual metrics (pinball, coverage, Brier, ECE) measure product quality.
Counterfactual metrics (MISE, policy error) measure whether the model recovered
the *whole* response function mu(c, x) — not just observed outcomes — which is
the point of the selection-bias stress test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Distributional / quantile metrics
# --------------------------------------------------------------------------- #
def pinball_loss(y: np.ndarray, q_pred: np.ndarray, q: float) -> float:
    d = y - q_pred
    return float(np.mean(np.maximum(q * d, (q - 1) * d)))


def interval_coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_width(lo: np.ndarray, hi: np.ndarray) -> float:
    return float(np.mean(hi - lo))


def mae(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(np.mean(np.abs(y - yhat)))


# --------------------------------------------------------------------------- #
# Probability calibration (P(Delta>0))
# --------------------------------------------------------------------------- #
def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-9) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)


def reliability_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> pd.DataFrame:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.any():
            rows.append({"bin": f"{edges[b]:.1f}-{edges[b+1]:.1f}",
                         "mean_pred": p[m].mean(), "observed_freq": y[m].mean(),
                         "n": int(m.sum())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Ranking metrics (banks per borrower)
# --------------------------------------------------------------------------- #
def _dcg(rel: np.ndarray) -> float:
    return float(np.sum(rel / np.log2(np.arange(2, len(rel) + 2))))


def ndcg_at_k(ranked_scores_true: list[np.ndarray], k: int = 5) -> float:
    """ranked_scores_true: per borrower, the TRUE relevance (e.g. -final_rate) in
    the model's predicted order. NDCG@k averaged over borrowers."""
    out = []
    for rel in ranked_scores_true:
        rel = np.asarray(rel, float)[:k]
        ideal = np.sort(rel)[::-1]
        denom = _dcg(ideal)
        out.append(_dcg(rel) / denom if denom > 0 else 0.0)
    return float(np.mean(out)) if out else 0.0


def top_k_regret(chosen_true_final: np.ndarray, best_true_final: np.ndarray) -> float:
    """Average rate the user leaves on the table by following the model's #1 pick."""
    return float(np.mean(chosen_true_final - best_true_final))


def hit_rate_at_k(pred_topk_banks: list[list[str]], true_best_bank: list[str],
                  k: int = 3) -> float:
    hits = [tb in banks[:k] for banks, tb in zip(pred_topk_banks, true_best_bank)]
    return float(np.mean(hits)) if hits else 0.0


# --------------------------------------------------------------------------- #
# Counterfactual metrics (need known ground truth from the synthetic layer)
# --------------------------------------------------------------------------- #
def mise(mu_true: np.ndarray, mu_pred: np.ndarray, c_grid: np.ndarray) -> float:
    """Mean Integrated Squared Error of the dose-response over counteroffer grid.

    mu_true, mu_pred: [n_rows, n_grid]. Integrates squared error over c per row,
    then averages over rows.
    """
    sq = (mu_true - mu_pred) ** 2
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2.0 rename
    integ = trapezoid(sq, c_grid, axis=1)
    return float(np.mean(integ))


def policy_error(best_final_true: np.ndarray, best_final_pred_choice: np.ndarray) -> float:
    """Squared error between the true best achievable final rate and the final
    rate actually obtained by following the model's recommended bank."""
    return float(np.mean((best_final_true - best_final_pred_choice) ** 2))


def summarize_reduction(panel_truth: pd.DataFrame, pred: pd.DataFrame) -> dict:
    """Bundle the key factual reduction metrics into one dict."""
    y = panel_truth["true_delta"].to_numpy(float)
    pos = (y > 1e-6).astype(int)
    return {
        "reduction_MAE": mae(y, pred["expected_delta"].to_numpy()),
        "pinball_q10": pinball_loss(y, pred["delta_q10"].to_numpy(), 0.10),
        "pinball_q50": pinball_loss(y, pred["delta_q50"].to_numpy(), 0.50),
        "pinball_q90": pinball_loss(y, pred["delta_q90"].to_numpy(), 0.90),
        "Brier_p_pos": brier_score(pos, pred["p_reduction"].to_numpy()),
        "LogLoss_p_pos": log_loss(pos, pred["p_reduction"].to_numpy()),
        "ECE_p_pos": expected_calibration_error(pos, pred["p_reduction"].to_numpy()),
    }
