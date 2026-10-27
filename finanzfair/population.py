"""Borrower population == real covariates X, mapped to the product schema.

No borrower is fabricated. For V0 research the population is a sample of *real*
HMDA borrowers (income, LTV, DTI, loan amount, term are real fields). Two fields
are not in HMDA and are handled explicitly rather than invented:

  * credit_quality  — a transparent, deterministic composite of real fields
                      (low DTI, low LTV, high income rank). Documented as a
                      proxy; replaced by real SCHUFA/credit score once
                      PHF/HFCS/NSMO/Freddie provide it.
  * fixed_period_years — a product input (German rate-fixation choice), not a
                      borrower attribute in HMDA. Defaults to the German modal
                      10y as a documented assumption.

When PHF/HFCS files are supplied, ``profiles_from_microdata`` uses the real
German/EU household distribution instead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _credit_quality_proxy(df: pd.DataFrame) -> np.ndarray:
    """Deterministic composite of real affordability fields, scaled to [0,1]."""
    dti = df["dti"].rank(pct=True)
    ltv = df["ltv"].rank(pct=True)
    inc = df["income_eur_proxy"].rank(pct=True)
    score = 0.4 * (1 - dti) + 0.3 * (1 - ltv) + 0.3 * inc
    return score.to_numpy()


def profiles_from_hmda(hmda_clean: pd.DataFrame, n: int | None = None,
                       fixed_period_years: int = 10, seed: int = 0) -> pd.DataFrame:
    """Map real HMDA rows to the FinanzFair borrower/loan/property schema."""
    d = hmda_clean
    if n is not None and n < len(d):
        d = d.sample(n, random_state=seed)
    d = d.reset_index(drop=True)
    q = _credit_quality_proxy(d)
    band = np.select([q > 0.75, q > 0.5, q > 0.28],
                     ["excellent", "good", "fair"], default="weak")
    out = pd.DataFrame({
        "borrower_id": [f"b{seed}_{i:06d}" for i in range(len(d))],
        "gross_household_income": d["income_eur_proxy"].round(0),
        "loan_amount": d["loan_amount"].round(0),
        "property_value": d["property_value"].round(0),
        "LTV": (d["ltv"] / 100.0).round(4),
        "DTI": (d["dti"] / 100.0).round(4),
        "term_years": d["term_years"].round(0),
        "fixed_period_years": fixed_period_years,
        "loan_purpose_label": d.get("loan_purpose_label", "purchase"),
        "credit_quality": np.round(q, 4),
        "credit_band": band,
        "income_eur_proxy": d["income_eur_proxy"],  # kept for the pricing surface
        "ltv": d["ltv"],
        "dti": d["dti"],
        "covariate_source": "hmda_real_us",
    })
    return out


def profiles_from_microdata(micro: pd.DataFrame, fixed_period_years: int = 10
                            ) -> pd.DataFrame:
    """Map real PHF/HFCS household microdata to the borrower schema.

    Requires the harmonised columns produced by ``ingest.microdata``. Fields that
    the survey genuinely lacks (e.g. loan structure for a hypothetical purchase)
    remain NaN for the researcher to set as product inputs; nothing is invented.
    """
    out = pd.DataFrame()
    out["borrower_id"] = [f"m_{i:06d}" for i in range(len(micro))]
    if "gross_income" in micro:
        out["gross_household_income"] = pd.to_numeric(micro["gross_income"], errors="coerce")
    out["fixed_period_years"] = fixed_period_years
    out["covariate_source"] = micro.get("source", "microdata_real")
    return out
