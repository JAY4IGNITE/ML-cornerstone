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
import math

from ..config import Config


def _finite_probability(prob: float) -> float:
    prob = float(prob)
    if not math.isfinite(prob):
        raise ValueError("Probability must be finite.")
    return max(0.0, min(1.0, prob))


def probability_to_score(prob: float) -> int:
    """Map default probability to a 0..100 risk score: round(prob * 100)."""
    prob = _finite_probability(prob)
    return int(round(prob * 100.0))



def probability_to_band(prob: float, cfg: Config) -> str:
    prob = _finite_probability(prob)
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
    """Describe an analytical threshold flag, never a lending decision."""
    prob = _finite_probability(prob)
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Threshold must be finite and between zero and one.")
    predicted_target = 1 if prob >= threshold else 0
    target_label = "At or above model threshold" if predicted_target else "Below model threshold"
    recommendation = "Elevated risk — analyst review" if predicted_target else "Below review threshold"
    badge_color = "#d97706" if predicted_target else "#059669"
    action = ("Review verified application information and model limitations. "
              "This estimate does not determine approval, pricing, collateral, or repayment.")

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
            "not a lending decision or guarantee. Independent human review is required."
        ),
    }
