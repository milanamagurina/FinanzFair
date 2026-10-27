#!/usr/bin/env python
"""End-to-end FinanzFair V0 pipeline on real data (CLI entry point).

Fetches real market/pricing/bank data, builds the initial-offer surface, applies
the semi-synthetic negotiation layer, trains the reduction model, and prints a
Top-5 recommendation plus held-out evaluation. See prototype.ipynb for the
narrated version with charts.

    python run_pipeline.py --n-borrowers 2000 --states DC MD VA --year 2023
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


def main() -> None:
    ap = argparse.ArgumentParser(description="FinanzFair V0 pipeline")
    ap.add_argument("--n-borrowers", type=int, default=2000)
    ap.add_argument("--states", nargs="+", default=["DC", "MD", "VA"])
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--theta", type=float, default=5.0, help="selection-bias level")
    ap.add_argument("--dgp", default="richards", help="negotiation DGP family")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    from finanzfair.ingest import ecb, hmda, scraper
    from finanzfair import pricing, population, banks, negotiation, product, evaluate

    print("[1/7] Real market state (ECB)…")
    market = ecb.build_market_panel(ecb.fetch_market_series(start="2018-01"))
    mrow = ecb.latest_market_row(market)
    bench = float(mrow["de_mortgage_rate_all_irf"])
    fix_bench = {5: float(mrow.get("de_mortgage_rate_1to5y", bench)),
                 10: float(mrow["de_mortgage_rate_5to10y"]),
                 15: float(mrow["de_mortgage_rate_over10y"]),
                 20: float(mrow["de_mortgage_rate_over10y"]),
                 30: float(mrow["de_mortgage_rate_over10y"])}
    print(f"      German benchmark {bench:.2f}% as of {mrow['month']}")

    print("[2/7] Real pricing surface (HMDA)…")
    clean = hmda.clean_hmda(hmda.fetch_hmda([args.year], args.states, (1,)))
    surface = pricing.EmpiricalPricingSurface().fit(clean)
    print(f"      {surface.n_train_:,} real loans")

    print("[3/7] Real bank offers (scraper)…")
    offers = scraper.scrape_and_store()
    if offers is None or offers.empty:
        offers = scraper.load_snapshots()
    bspreads = banks.bank_spreads_from_offers(offers, benchmark_rate=bench)
    print(f"      {len(bspreads)} German banks")

    print("[4/7] Borrower population + initial rates…")
    prof = population.profiles_from_hmda(clean, n=args.n_borrowers, seed=args.seed)
    initial = pricing.empirical_floor(
        pricing.assemble_initial_rates(prof, surface, bspreads, bench, fix_bench))

    print("[5/7] Semi-synthetic negotiation layer…")
    ic = negotiation.apply_selection_bias(initial, prof, theta=args.theta, seed=args.seed)
    panel = ic.merge(prof[["borrower_id", "LTV", "DTI", "gross_household_income"]], on="borrower_id")
    dgp = negotiation.NegotiationDGP(family=args.dgp, selection_bias=args.theta, seed=args.seed)
    bpm = dgp.sample_bank_params(sorted(panel.bank_id.unique()))
    sim = dgp.simulate(panel, bpm)

    print("[6/7] Reduction model + final-rate distribution + ranking…")
    bids = prof.borrower_id.unique().copy()
    np.random.default_rng(args.seed).shuffle(bids)
    train_ids = set(bids[:int(0.7 * len(bids))])
    tr, te = sim[sim.borrower_id.isin(train_ids)], sim[~sim.borrower_id.isin(train_ids)]
    model = product.ReductionHurdleModel().fit(tr)
    pred = model.predict(te)
    fin = product.final_rate_distribution(te, pred)
    full = pd.concat([te.reset_index(drop=True), pred.reset_index(drop=True),
                      fin.reset_index(drop=True)], axis=1)
    ranked = product.rank_banks(full, lam=0.5, top_k=5,
                                penalties=banks.data_density_penalty(bspreads))
    one = ranked[ranked.borrower_id == ranked.borrower_id.iloc[0]]
    print("\n=== Top-5 recommendation (one borrower) ===")
    print(product.format_recommendation(one).to_string(index=False))

    print("\n[7/7] Held-out evaluation…")
    for k, v in evaluate.summarize_reduction(te, pred).items():
        print(f"      {k:16s} {v:.4f}")
    c_grid = np.linspace(0, 50, 26)
    mise = evaluate.mise(dgp.true_mu_grid(te, c_grid, bpm), model.predict_mu(te, c_grid), c_grid)
    print(f"      MISE             {mise:.4f}")


if __name__ == "__main__":
    main()
