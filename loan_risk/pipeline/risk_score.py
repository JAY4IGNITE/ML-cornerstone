"""Transparent risk score + band mapping (03_ML_REQUIREMENTS.md / 05_API_CONTRACT).

The score is explicitly defined and must never be presented as a guaranteed
outcome:
    risk_score = round(default_probability * 100)   # 0..100, monotonic in prob
Bands are configured probability intervals (config.risk_score.bands).

Definitions are exposed via `score_definition()` so the API/frontend/model-card
can display exactly how the number is derived.
"""
from __future__ import annotations

from typing import Any

from ..config import Config


def probability_to_score(prob: float) -> int:
    """Map default probability to a 0..100 risk score: round(prob * 100)."""
    prob = max(0.0, min(1.0, float(prob)))
    return int(round(prob * 100.0))



def probability_to_band(prob: float, cfg: Config) -> str:
    prob = max(0.0, min(1.0, float(prob)))
    bands = cfg.get("risk_score", {}).get("bands", [])
    if not bands:
        # Fallback default calibrated bands
        if prob < 0.06:
            return "Low"
        elif prob < 0.15:
            return "Moderate"
        elif prob < 0.25:
            return "High"
        return "Critical"

    for band in bands:
        if band["min_prob"] <= prob < band["max_prob"]:
            return band["name"]
    return bands[-1]["name"]


def probability_to_decision(prob: float, threshold: float = 0.166) -> dict[str, Any]:
    """Derive binary prediction (TARGET=0 vs 1) and loan recommendation.

    Threshold defaults to 0.166 (the validation-optimal F1 operating point).
    """
    prob = max(0.0, min(1.0, float(prob)))
    predicted_target = 1 if prob >= threshold else 0
    target_label = "Payment Difficulty / Default" if predicted_target == 1 else "Repayment Expected"

    if prob < 0.06:
        recommendation = "Approved (Low Default Risk)"
        badge_color = "#059669"  # emerald
        action = "Eligible for prime financing, standard loan limits, and optimal interest rates."
    elif prob < 0.15:
        recommendation = "Conditional / Manual Review"
        badge_color = "#d97706"  # amber
        action = "Acceptable risk profile. Recommend secondary income and employment verification."
    elif prob < 0.25:
        recommendation = "High Risk / Strict Terms"
        badge_color = "#ea580c"  # orange
        action = "Elevated default probability. Requires creditworthy co-signer or 25%+ collateral."
    else:
        recommendation = "Declined (Critical Default Risk)"
        badge_color = "#dc2626"  # red
        action = "Exceeds acceptable default risk threshold. Substantial repayment difficulty predicted."

    return {
        "predicted_target": predicted_target,
        "target_label": target_label,
        "recommendation": recommendation,
        "badge_color": badge_color,
        "action": action,
        "threshold": threshold,
    }


def score_definition(cfg: Config) -> dict[str, Any]:
    return {
        "score_formula": "round(default_probability * 100)",
        "score_range": [0, 100],
        "monotonic": "risk_score increases monotonically with default_probability",
        "bands": [
            {"name": b["name"],
             "probability_range": [b["min_prob"], min(b["max_prob"], 1.0)]}
            for b in cfg.get("risk_score", {}).get("bands", [])
        ],
        "disclaimer": (
            "The risk score is an analytical estimate of default probability, "
            "designed to assist credit analysts with empirical data-driven insights."
        ),
    }
