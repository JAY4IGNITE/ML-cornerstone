"""One-command acquisition of the real Home Credit Default Risk dataset.

Real data is NOT committed to the repo (it is a Kaggle competition dataset and
git-ignored). This script fetches it into ``paths.data_raw`` so that flipping
``dataset.source: real`` in config.yaml makes the pipeline train on it with no
code changes.

Requires either:
  * the Kaggle CLI + API token (``~/.kaggle/kaggle.json`` or KAGGLE_USERNAME /
    KAGGLE_KEY env vars), and having accepted the competition rules, OR
  * a manual download placed in data/raw/ (the script then just verifies files).

We never fabricate the data. If credentials/rules are missing we print exact,
actionable instructions and exit non-zero (Execution Rule #10: report blockers).
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ..config import Config, load_config

COMPETITION = "home-credit-default-risk"
EXPECTED_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "installments_payments.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "HomeCredit_columns_description.csv",
]


def _have_kaggle_cli() -> bool:
    try:
        subprocess.run(["kaggle", "--version"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def _present(raw_dir: Path) -> list[str]:
    return [f for f in EXPECTED_FILES if (raw_dir / f).exists()]


def download(cfg: Config, force: bool = False) -> int:
    raw_dir = cfg.path("data_raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    primary = cfg["dataset"]["primary_file"]
    if (raw_dir / primary).exists() and not force:
        print(f"[download] {primary} already present in {raw_dir}. "
              f"Use --force to re-download.")
        _report_presence(raw_dir)
        return 0

    if not _have_kaggle_cli():
        _print_manual_instructions(raw_dir)
        return 2

    print(f"[download] downloading competition '{COMPETITION}' via Kaggle CLI ...")
    try:
        subprocess.run(
            ["kaggle", "competitions", "download", "-c", COMPETITION,
             "-p", str(raw_dir)],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"[download] Kaggle CLI failed (exit {exc.returncode}).", file=sys.stderr)
        _print_manual_instructions(raw_dir)
        return exc.returncode or 2

    # unzip the bundle(s)
    import zipfile
    for zip_path in raw_dir.glob("*.zip"):
        print(f"[download] extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(raw_dir)

    _report_presence(raw_dir)
    if not (raw_dir / primary).exists():
        print(f"[download] WARNING: expected {primary} not found after extraction.",
              file=sys.stderr)
        return 2
    print(f"[download] done. Set dataset.source: real in config.yaml to use it.")
    return 0


def _report_presence(raw_dir: Path) -> None:
    present = _present(raw_dir)
    missing = [f for f in EXPECTED_FILES if f not in present]
    print(f"[download] present ({len(present)}): {', '.join(present) or 'none'}")
    if missing:
        print(f"[download] not present ({len(missing)}): {', '.join(missing)}")


def _print_manual_instructions(raw_dir: Path) -> None:
    print(
        "\n[download] BLOCKER — cannot fetch real data automatically.\n"
        "The Kaggle CLI/credentials are not available, or competition rules\n"
        "have not been accepted. To obtain the real dataset:\n\n"
        "  Option A (CLI):\n"
        "    1. pip install kaggle\n"
        "    2. Create an API token at https://www.kaggle.com/settings/account\n"
        "       and place kaggle.json at %USERPROFILE%\\.kaggle\\kaggle.json\n"
        "       (or set KAGGLE_USERNAME / KAGGLE_KEY env vars).\n"
        "    3. Accept the rules at\n"
        "       https://www.kaggle.com/competitions/home-credit-default-risk/rules\n"
        "    4. Re-run: python -m loan_risk.data.download\n\n"
        "  Option B (manual):\n"
        f"    Download application_train.csv from the competition Data tab and\n"
        f"    place it (and any supporting files) in:\n      {raw_dir}\n\n"
        "Until then the pipeline runs on the labeled SYNTHETIC fixture\n"
        "(dataset.source: synthetic).\n",
        file=sys.stderr,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download real Home Credit data from Kaggle.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--force", action="store_true", help="Re-download even if present.")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    return download(cfg, force=args.force)


if __name__ == "__main__":
    raise SystemExit(main())
