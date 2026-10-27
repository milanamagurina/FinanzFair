"""FinanzFair V0 — mortgage rate & negotiation intelligence (real-data build).

Design contract (see the architecture doc):

    REAL      borrower covariates, market state, bank offers, realized pricing.
    SYNTHETIC only the single unobserved layer: how much of the empirically
              feasible pricing room a negotiation actually extracts.

Nothing in the pricing / market / borrower path is fabricated. The only
generated quantity is the *fraction of the empirical Delta_max extracted* as a
function of counteroffer strength (``finanzfair.negotiation``), and it is always
carried with an explicit ``label_source`` so synthetic and observed rows never
mix silently.
"""

from __future__ import annotations

from . import config

__all__ = ["config"]
__version__ = config.SIMULATION_VERSION
