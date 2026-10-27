# FinanzFair V0 — Mortgage Rate & Negotiation Intelligence

A real-data research environment that estimates, for a German mortgage borrower, a **distribution** of
the achievable final rate per bank and ranks the Top-5 banks by expected final rate (not by nominal
discount).

```
P(R_final | X, Bank, Market_t, CompetingOffers),   R_final = R_initial − Δ
```

## The design contract (what is real, what is synthetic)

Everything the rate **level** is built from is real published data. The **only** generated quantity is
*how much of the empirically feasible pricing room a negotiation extracts*, as a function of counteroffer
strength — the one layer with no public dataset. It always carries `label_source='synthetic'`, so
synthetic and observed rows never mix silently.

| Layer | Source | Status |
|---|---|---|
| Market state `Market_t` | ECB Data Portal (German MIR rate by fixation, policy rate, €STR, AAA yields) | **REAL** |
| Realized pricing surface | HMDA loan-level (CFPB/FFIEC): `rate_spread`, income, LTV, DTI, term, lender | **REAL** |
| Bank offers | `thebanks.eu` scraper — German lenders, append-only snapshots | **REAL** |
| Borrower covariates `X` | HMDA rows now; PHF/HFCS when you provide them | **REAL** |
| Empirical room `Δmax` | HMDA spread dispersion → room to aggressive (Q10) pricing | **REAL** |
| **Extracted fraction of Δmax** | Richards response over counteroffer strength | **SEMI-SYNTHETIC** |

The absolute German rate level comes from real ECB/Bundesbank series; HMDA contributes only the
benchmark-relative **shape** and dispersion (cross-country caveat: US data are research priors, never
German labels).

## Quickstart

```bash
conda activate py313_env         # numpy, pandas, scipy, scikit-learn, matplotlib, requests, bs4, pyarrow
python run_pipeline.py           # fetch real data → train → Top-5 + evaluation
jupyter lab prototype.ipynb      # narrated end-to-end with charts
```

First run fetches and caches real data under `data/raw/` (append-only, timestamped for reproducibility).

## Package layout

```
finanzfair/
  config.py            versions, quantiles, reproducibility (RunContext)
  ingest/
    ecb.py             real ECB SDMX market panel (point-in-time)
    bundesbank.py      real BBK dispersion / renegotiated-loan series (generic SDMX)
    hmda.py            real HMDA loan-level pricing surface data
    scraper.py         real German bank offers, append-only snapshot store
    microdata.py       PHF / HFCS / Freddie Mac loaders (read files you provide; no fabrication)
  pricing.py           EmpiricalPricingSurface (quantile GBM) + initial-rate assembly + Δmax cap
  population.py        real borrower covariates → product schema
  banks.py             bank spreads from real offers; data-density penalty
  negotiation.py       the one semi-synthetic layer: Richards/stacked/threshold/rigid + hurdle + true μ(c,x)
  product.py           ReductionHurdleModel, final-rate Monte Carlo, Top-K ranking
  evaluate.py          pinball/coverage, Brier/ECE/reliability, NDCG, MISE, policy error
```

## Credentialed datasets

PHF, HFCS and Freddie Mac require registration. Download them and drop the files here; the loaders in
`finanzfair.ingest.microdata` read them (and raise with exact instructions if missing — nothing is
substituted synthetically):

```
data/external/phf/        Bundesbank PHF Scientific Use File  → German borrower distribution
data/external/hfcs/       ECB HFCS harmonised microdata       → EU cross-country borrowers
data/external/freddie/    Freddie Mac loan-level              → second pricing surface (validation)
```

## Roadmap

- **V1 (product):** CatBoost quantile initial-offer model + hierarchical-Bayesian bank-effect challenger;
  conformal calibration.
- **V2 (real feedback):** replace/calibrate the synthetic layer with real user-reported outcomes
  (`label_source` ∈ {user_reported, verified_offer, verified_final_offer}); hierarchical bank partial pooling.
- **V3 (causal):** counterfactual dose-response once identification is defensible.

## Caveats

Synthetic negotiation ≠ evidence; advertised minimum ≠ bank floor; user-reported data are selected;
counteroffers must be normalised for fixation/amount/LTV/fees; bank behaviour drifts over time. See
the final section of `prototype.ipynb`.
