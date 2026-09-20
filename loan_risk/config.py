"""Configuration loader.

Single source of truth for paths, seeds, thresholds and hyperparameters.
No tunable value is hardcoded in logic modules (Execution Rule #7).

Resolution order for the config file:
    1. explicit path passed to ``load_config``
    2. ``$LOAN_RISK_CONFIG`` environment variable
    3. ``<repo_root>/config/config.yaml``
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    """Repo root = two levels up from this file (loan_risk/config.py)."""
    return Path(__file__).resolve().parents[1]


def _default_config_path() -> Path:
    env = os.environ.get("LOAN_RISK_CONFIG")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "config" / "config.yaml"


@dataclass(frozen=True)
class Config:
    """Typed wrapper around the parsed YAML with path resolution helpers."""

    raw: dict[str, Any]
    root: Path = field(default_factory=repo_root)

    # -- generic access -------------------------------------------------
    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    # -- path helpers (all resolved relative to repo root) --------------
    def path(self, key: str) -> Path:
        """Resolve a key under ``paths:`` to an absolute Path."""
        rel = self.raw["paths"][key]
        p = Path(rel)
        return p if p.is_absolute() else (self.root / p)

    @property
    def seed(self) -> int:
        return int(self.raw["project"]["random_seed"])

    @property
    def target(self) -> str:
        return str(self.raw["dataset"]["target"])

    @property
    def id_column(self) -> str:
        return str(self.raw["dataset"]["id_column"])

    def ensure_dirs(self) -> None:
        """Create the writable output directories if missing."""
        for key in ("data_raw", "data_synthetic", "data_processed",
                    "artifacts", "reports"):
            self.path(key).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=8)
def load_config(path: str | os.PathLike[str] | None = None) -> Config:
    """Load and cache the configuration."""
    cfg_path = Path(path).expanduser().resolve() if path else _default_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Config file not found at {cfg_path}. Set $LOAN_RISK_CONFIG or "
            f"create config/config.yaml."
        )
    with cfg_path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"Config at {cfg_path} did not parse to a mapping.")
    return Config(raw=raw)
