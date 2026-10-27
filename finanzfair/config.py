"""Global configuration, versions, and reproducibility helpers for FinanzFair V0.

Everything that affects a simulated observation is versioned here so that a
training row can be reproduced exactly from (simulation_version, seed, dgp
parameters) as required by the architecture's reproducibility section.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np

# --- Versions (bump when the generating process changes) -------------------
SIMULATION_VERSION = "v0.1.0"
MODEL_VERSION = "v0.1.0"
MARKET_SNAPSHOT_VERSION = "v0.1.0"
BORROWER_DIST_VERSION = "phf_proxy_v0.1.0"

# --- Prediction quantiles --------------------------------------------------
QUANTILES = (0.10, 0.50, 0.90)
# Interval used for user-facing "80% interval" (Q10..Q90).
INTERVAL_LOW, INTERVAL_HIGH = 0.10, 0.90

# --- Label provenance (never silently mix synthetic and observed) ----------
LABEL_SOURCES = ("synthetic", "user_reported", "verified_offer", "verified_final_offer")

# --- Units -----------------------------------------------------------------
# Rates are stored in percentage points (e.g. 3.72 == 3.72%).
# Reductions (delta) are stored in percentage points as well; 1 bp = 0.01 pp.
BP = 0.01  # one basis point expressed in percentage points


def rng(seed: int | None = None) -> np.random.Generator:
    """Single source of randomness. Always go through this for reproducibility."""
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class RunContext:
    """Stamped onto every simulated table so experiments are reproducible."""

    seed: int
    simulation_version: str = SIMULATION_VERSION
    model_version: str = MODEL_VERSION
    market_snapshot_version: str = MARKET_SNAPSHOT_VERSION
    borrower_dist_version: str = BORROWER_DIST_VERSION
    extra: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, str]:
        d = {
            "seed": self.seed,
            "simulation_version": self.simulation_version,
            "model_version": self.model_version,
            "market_snapshot_version": self.market_snapshot_version,
            "borrower_dist_version": self.borrower_dist_version,
        }
        d.update(self.extra)
        return d
