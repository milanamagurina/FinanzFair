"""Loaders for credentialed real microdata: Bundesbank PHF, ECB HFCS, Freddie Mac.

These datasets are REAL but require registration / a signed use agreement, so
they cannot be fetched autonomously. These loaders read the files *you* download
into ``data/external/`` and harmonise them to a common borrower schema. They
never fabricate rows: if a file is missing they raise with the exact download
instructions. Nothing downstream silently substitutes synthetic data for them.

Expected drop locations (override via arguments):

    data/external/phf/        Bundesbank PHF Scientific Use File (Stata/CSV)
    data/external/hfcs/       ECB HFCS harmonised microdata (Stata/CSV)
    data/external/freddie/    Freddie Mac Single-Family Loan-Level (txt)
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

INSTRUCTIONS = {
    "phf": ("Bundesbank Panel on Household Finances (PHF).\n"
            "  1. Apply for the Scientific Use File at the Bundesbank RDSC.\n"
            "  2. Place the household file (.dta or .csv) in data/external/phf/.\n"
            "  Provides the real German borrower distribution P_Germany(X)."),
    "hfcs": ("ECB Household Finance and Consumption Survey (HFCS).\n"
             "  1. Register for the HFCS research dataset via the ECB.\n"
             "  2. Place the harmonised D/H files in data/external/hfcs/.\n"
             "  Provides cross-country euro-area borrower distributions."),
    "freddie": ("Freddie Mac Single-Family Loan-Level Dataset.\n"
                "  1. Create a free account and accept the license.\n"
                "  2. Place the origination 'historical_data_*' files in data/external/freddie/.\n"
                "  Provides a second realized-pricing surface for validation."),
}


def _require(folder: Path, key: str) -> list[Path]:
    files = [p for p in folder.glob("**/*")
             if p.suffix.lower() in {".csv", ".dta", ".txt", ".parquet", ".zip"}]
    if not files:
        raise FileNotFoundError(
            f"No {key.upper()} files found in {folder}.\n{INSTRUCTIONS[key]}")
    return files


def load_phf(folder: str | Path = "data/external/phf") -> pd.DataFrame:
    """Load Bundesbank PHF and map to the harmonised borrower schema."""
    files = _require(Path(folder), "phf")
    df = _read_any(files[0])
    return _harmonise_household(df, source="phf_real")


def load_hfcs(folder: str | Path = "data/external/hfcs",
              countries: tuple[str, ...] = ("DE", "BE", "FR", "NL", "AT", "ES", "IT")) -> pd.DataFrame:
    """Load ECB HFCS harmonised microdata (optionally filtered by country)."""
    files = _require(Path(folder), "hfcs")
    df = _read_any(files[0])
    if "country" in df.columns:
        df = df[df["country"].isin(countries)]
    return _harmonise_household(df, source="hfcs_real")


def load_freddie(folder: str | Path = "data/external/freddie") -> pd.DataFrame:
    """Load Freddie Mac loan-level origination file (pipe-delimited, no header)."""
    files = _require(Path(folder), "freddie")
    # Freddie origination layout: documented column order; read the key fields.
    cols = ["credit_score", "first_payment_date", "first_time_buyer", "maturity_date",
            "msa", "mi_pct", "units", "occupancy", "cltv", "dti", "orig_upb", "ltv",
            "orig_rate"]
    df = pd.read_csv(files[0], sep="|", header=None, low_memory=False)
    df = df.iloc[:, :len(cols)]
    df.columns = cols
    df["source"] = "freddie_real"
    return df


def _read_any(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".dta":
        return pd.read_stata(path, convert_categoricals=False)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, low_memory=False)


def _harmonise_household(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Best-effort rename to the common borrower schema; keep originals too.

    Survey variable names differ across waves/releases, so this maps the most
    common aliases and leaves unmapped columns intact for the researcher.
    """
    alias = {
        "gross_income": ["di2000", "hg0100", "gross_income", "income_gross"],
        "net_income": ["di2000i", "net_income", "income_net"],
        "liquid_assets": ["da2101", "financial_assets", "deposits"],
        "total_debt": ["dl1000", "total_liabilities", "debt"],
        "employment": ["pe0100", "employment_status", "empl"],
    }
    out = df.copy()
    for target, cands in alias.items():
        for c in cands:
            if c in out.columns and target not in out.columns:
                out[target] = out[c]
                break
    out["source"] = source
    return out
