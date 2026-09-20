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
    prob = max(0.0, min(1.0, float(prob)))
    return int(round(prob * 100))


def probability_to_band(prob: float, cfg: Config) -> str:
    prob = max(0.0, min(1.0, float(prob)))
    for band in cfg["risk_score"]["bands"]:
        if band["min_prob"] <= prob < band["max_prob"]:
            return band["name"]
    return cfg["risk_score"]["bands"][-1]["name"]


def score_definition(cfg: Config) -> dict[str, Any]:
    return {
        "score_formula": "risk_score = round(default_probability * 100)",
        "score_range": [0, 100],
        "monotonic": "risk_score increases with default_probability",
        "bands": [
            {"name": b["name"],
             "probability_range": [b["min_prob"], min(b["max_prob"], 1.0)]}
            for b in cfg["risk_score"]["bands"]
        ],
        "disclaimer": (
            "The risk score is an analytical estimate of default probability, "
            "NOT a guaranteed outcome and NOT an autonomous lending decision."
        ),
    }
