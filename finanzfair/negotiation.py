"""The ONE semi-synthetic layer: how much of the real Delta_max a negotiation
extracts, as a function of counteroffer strength.

Everything feeding this is real: the empirical feasible room Delta_max (from the
pricing surface + real dispersion) and the counteroffer strength
    C = R_initial(bank) - R_bestComparable       (real, cross-bank).
Generated here is ONLY the extracted fraction g(C) in [0, 1] and whether any
repricing occurs (hurdle). Every row produced carries label_source='synthetic'
and a synthetic_version, so it can never be mistaken for an observed outcome.

The true response function mu(c, x) = E[Delta | C=c, x] is exposed so the
counterfactual benchmark (MISE, policy error) has known ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import rng

SYNTHETIC_VERSION = "neg_dgp_v0.1.0"
DGP_FAMILIES = ("richards", "stacked_sigmoid", "threshold", "rigid", "borrower_dependent")


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def richards(c_bp: np.ndarray, c0: float, k: float, nu: float = 1.0) -> np.ndarray:
    """Richards (generalised logistic) extracted-fraction curve in [0, 1].

    c_bp: counteroffer strength in basis points. c0: inflection; k: steepness;
    nu: asymmetry (nu<1 early rise, nu>1 late rise).
    """
    z = -k * (c_bp - c0)
    return (1.0 + np.exp(z)) ** (-1.0 / max(nu, 1e-6))


def _fraction(family: str, c_bp: np.ndarray, p: dict) -> np.ndarray:
    if family == "richards":
        return richards(c_bp, p["c0"], p["k"], p.get("nu", 1.0))
    if family == "stacked_sigmoid":
        return 0.5 * _sigmoid(p["k1"] * (c_bp - p["c01"])) + \
               0.5 * _sigmoid(p["k2"] * (c_bp - p["c02"]))
    if family == "threshold":
        f = np.where(c_bp < p["t_lo"], 0.03,
                     np.where(c_bp > p["t_hi"], 0.95,
                              (c_bp - p["t_lo"]) / max(p["t_hi"] - p["t_lo"], 1e-6)))
        return np.clip(f, 0, 1)
    if family == "rigid":
        return 0.15 * _sigmoid(0.05 * (c_bp - p.get("c0", 25.0)))  # low elasticity
    if family == "borrower_dependent":
        return richards(c_bp, p["c0"], p["k"], p.get("nu", 1.0))
    raise ValueError(f"unknown DGP family: {family}")


@dataclass
class BankNegotiationParams:
    """Per-bank latent negotiation behaviour (hierarchical draw)."""

    bank_id: str
    family: str
    c0: float           # counteroffer strength (bp) at which repricing ramps up
    k: float            # sensitivity to competitive pressure (gamma)
    nu: float
    base_logit: float   # baseline propensity to move at all (alpha_b)
    extras: dict = field(default_factory=dict)


@dataclass
class NegotiationDGP:
    """Semi-synthetic negotiation ground truth anchored to real Delta_max."""

    family: str = "richards"
    noise_sd_bp: float = 2.0
    selection_bias: float = 0.0   # theta: how strongly C depends on borrower quality
    seed: int = 0
    mu_c0: float = 18.0
    sd_c0: float = 5.0
    mu_k: float = 0.09
    sd_k: float = 0.02

    def sample_bank_params(self, bank_ids: list[str]) -> dict[str, BankNegotiationParams]:
        g = rng(self.seed)
        params = {}
        for b in bank_ids:
            c0 = float(np.clip(g.normal(self.mu_c0, self.sd_c0), 4, 40))
            k = float(np.clip(g.normal(self.mu_k, self.sd_k), 0.02, 0.25))
            nu = float(np.clip(g.normal(1.0, 0.3), 0.3, 2.5))
            base = float(g.normal(0.3, 0.4))
            params[b] = BankNegotiationParams(b, self.family, c0, k, nu, base,
                                              extras={"c01": c0 - 6, "c02": c0 + 8,
                                                      "k1": k * 1.4, "k2": k * 0.8,
                                                      "t_lo": 10.0, "t_hi": 25.0})
        return params

    def repricing_prob(self, c_bp: np.ndarray, bp: BankNegotiationParams,
                       credit_quality: np.ndarray) -> np.ndarray:
        """P(Delta > 0): hurdle. Rises with counteroffer strength and (mildly)
        borrower quality (stronger borrowers are taken more seriously)."""
        q = 0.0 if self.family == "rigid" else 0.8
        logit = bp.base_logit + bp.k * (c_bp - bp.c0) + q * (credit_quality - 0.5)
        return _sigmoid(logit)

    def true_mu(self, c_bp: np.ndarray, delta_max: np.ndarray,
                bp: BankNegotiationParams, credit_quality: np.ndarray) -> np.ndarray:
        """Known counterfactual E[Delta | C=c, x] = P(move) * Delta_max * fraction.

        For 'borrower_dependent' both the fraction and Delta_max scale with
        borrower quality, reproducing the selection-bias scenario at the level of
        the response itself.
        """
        p = {"c0": bp.c0, "k": bp.k, "nu": bp.nu, **bp.extras}
        frac = _fraction(self.family, c_bp, p)
        if self.family == "borrower_dependent":
            frac = np.clip(frac * (0.6 + 0.8 * credit_quality), 0, 1)
        pr = self.repricing_prob(c_bp, bp, credit_quality)
        return pr * delta_max * frac

    def true_mu_grid(self, panel: pd.DataFrame, c_grid_bp: np.ndarray,
                     bank_params: dict[str, BankNegotiationParams]) -> np.ndarray:
        """Known counterfactual mu(c, x) over a counteroffer grid: [n_rows, n_grid].

        This is the ground truth the model's predict_mu is scored against (MISE).
        """
        dmax = panel["delta_max"].to_numpy(float)
        cq = panel["credit_quality"].to_numpy(float)
        banks = panel["bank_id"].to_numpy()
        out = np.empty((len(panel), len(c_grid_bp)))
        for j, c in enumerate(c_grid_bp):
            cc = np.full(len(panel), float(c))
            col = np.empty(len(panel))
            for b in np.unique(banks):
                m = banks == b
                col[m] = self.true_mu(cc[m], dmax[m], bank_params[b], cq[m])
            out[:, j] = col
        return out

    def simulate(self, panel: pd.DataFrame,
                 bank_params: dict[str, BankNegotiationParams] | None = None
                 ) -> pd.DataFrame:
        """Generate factual negotiation outcomes for a (borrower, bank) panel.

        ``panel`` must contain: borrower_id, bank_id, credit_quality, delta_max,
        r_initial_q50, and the real counteroffer_gap_bp. Returns the panel with
        synthetic true_delta / final_rate columns + label provenance. Also returns
        the per-row realised repricing prob and true_mu for evaluation.
        """
        g = rng(self.seed + 1)
        bank_ids = sorted(panel["bank_id"].unique())
        bp_map = bank_params or self.sample_bank_params(bank_ids)

        c = panel["counteroffer_gap_bp"].to_numpy(float)
        dmax = panel["delta_max"].to_numpy(float)
        cq = panel["credit_quality"].to_numpy(float)

        frac = np.empty(len(panel)); pr = np.empty(len(panel)); mu = np.empty(len(panel))
        for b in bank_ids:
            m = (panel["bank_id"] == b).to_numpy()
            bp = bp_map[b]
            p = {"c0": bp.c0, "k": bp.k, "nu": bp.nu, **bp.extras}
            f = _fraction(self.family, c[m], p)
            if self.family == "borrower_dependent":
                f = np.clip(f * (0.6 + 0.8 * cq[m]), 0, 1)
            frac[m] = f
            pr[m] = self.repricing_prob(c[m], bp, cq[m])
            mu[m] = pr[m] * dmax[m] * f

        # Factual draw: hurdle then conditional reduction, capped at real Delta_max.
        z = g.random(len(panel)) < pr
        delta_pos = np.clip(dmax * frac + g.normal(0, self.noise_sd_bp / 100.0, len(panel)),
                            0, dmax)
        true_delta = np.where(z, delta_pos, 0.0)

        out = panel.copy()
        out["repricing_prob_true"] = pr
        out["true_mu_delta"] = mu                       # counterfactual ground truth
        out["true_delta"] = true_delta                  # factual synthetic label
        out["final_rate"] = out["r_initial_q50"] - true_delta
        out["label_source"] = "synthetic"
        out["synthetic_version"] = SYNTHETIC_VERSION
        out["dgp_family"] = self.family
        out["selection_bias_theta"] = self.selection_bias
        return out


def apply_selection_bias(initial_rates: pd.DataFrame, profiles: pd.DataFrame,
                         theta: float, seed: int = 0) -> pd.DataFrame:
    """Make the available counteroffer strength depend on borrower quality (theta).

    Strong borrowers obtain stronger competing offers, so X -> C and X -> final
    rate simultaneously. This is the offline selection-bias stress test: at
    theta=0 the best comparable offer is used as-is; as theta grows, weaker
    borrowers systematically see a *shrunken* effective counteroffer gap.
    """
    g = rng(seed + 99)
    df = initial_rates.merge(profiles[["borrower_id", "credit_quality"]], on="borrower_id")
    # Best comparable offer = min competing bank initial across OTHER banks.
    best = (df.groupby(["borrower_id"])["r_initial_q50"].transform("min"))
    raw_gap = (df["r_initial_q50"] - best).clip(lower=0)
    # theta shrinks the realised gap for low-quality borrowers.
    shrink = 1.0 - theta * 0.05 * (1.0 - df["credit_quality"])
    noise = 1.0 + g.normal(0, 0.05, len(df))
    df["counteroffer_gap_bp"] = (raw_gap * 100.0 * shrink.clip(0.1, 1.0) * noise).clip(lower=0)
    return df
