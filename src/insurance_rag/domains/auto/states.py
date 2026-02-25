"""State-specific configuration for Auto Insurance.

Covers the top US auto insurance markets with tort system, minimum
liability limits, and PIP/no-fault requirements.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class StateAutoConfig:
    """Auto insurance regulatory profile for a US state."""

    code: str
    name: str
    tort_system: str  # "tort", "no-fault", or "choice"
    min_liability: str  # e.g. "25/50/25" (BI per person / BI per accident / PD)
    pip_required: bool
    um_uim_required: bool  # uninsured/underinsured motorist
    notes: str = ""


TOP_MARKETS: list[str] = ["CA", "TX", "FL", "NY", "IL", "PA", "OH"]

STATE_CONFIGS: dict[str, StateAutoConfig] = {
    "CA": StateAutoConfig(
        code="CA",
        name="California",
        tort_system="tort",
        min_liability="15/30/5",
        pip_required=False,
        um_uim_required=True,
        notes="Must offer UM/UIM; no PIP but MedPay optional",
    ),
    "TX": StateAutoConfig(
        code="TX",
        name="Texas",
        tort_system="tort",
        min_liability="30/60/25",
        pip_required=True,
        um_uim_required=True,
        notes="PIP (Personal Injury Protection) required; UM/UIM offered",
    ),
    "FL": StateAutoConfig(
        code="FL",
        name="Florida",
        tort_system="no-fault",
        min_liability="10/20/10",
        pip_required=True,
        um_uim_required=False,
        notes="No-fault state; PIP $10k required; BI liability not mandatory but recommended",
    ),
    "NY": StateAutoConfig(
        code="NY",
        name="New York",
        tort_system="no-fault",
        min_liability="25/50/10",
        pip_required=True,
        um_uim_required=True,
        notes="No-fault state; PIP $50k (basic); SUM (UM/UIM) required at $25/50",
    ),
    "IL": StateAutoConfig(
        code="IL",
        name="Illinois",
        tort_system="tort",
        min_liability="25/50/20",
        pip_required=False,
        um_uim_required=True,
        notes="UM/UIM required; no PIP",
    ),
    "PA": StateAutoConfig(
        code="PA",
        name="Pennsylvania",
        tort_system="choice",
        min_liability="15/30/5",
        pip_required=True,
        um_uim_required=False,
        notes="Choice between full tort and limited tort; first-party medical benefits required",
    ),
    "OH": StateAutoConfig(
        code="OH",
        name="Ohio",
        tort_system="tort",
        min_liability="25/50/25",
        pip_required=False,
        um_uim_required=True,
        notes="UM/UIM required; no PIP or no-fault",
    ),
    "NJ": StateAutoConfig(
        code="NJ",
        name="New Jersey",
        tort_system="choice",
        min_liability="15/30/5",
        pip_required=True,
        um_uim_required=True,
        notes="Choice no-fault; PIP $15k minimum; verbal/limitation on lawsuit threshold",
    ),
    "MI": StateAutoConfig(
        code="MI",
        name="Michigan",
        tort_system="no-fault",
        min_liability="50/100/10",
        pip_required=True,
        um_uim_required=False,
        notes="Strong no-fault; PIP options from $50k to unlimited; mini-tort up to $3k",
    ),
    "GA": StateAutoConfig(
        code="GA",
        name="Georgia",
        tort_system="tort",
        min_liability="25/50/25",
        pip_required=False,
        um_uim_required=False,
        notes="Add-on no-fault optional; UM required if BI purchased",
    ),
}
