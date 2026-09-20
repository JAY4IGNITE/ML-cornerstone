"""Convenience helper: download the Home Credit competition bundle via kagglehub.

This is a minimal standalone alternative that fetches the dataset into the local
kagglehub cache. It does NOT place the file where the pipeline expects
(``data/raw/application_train.csv``) and ``kagglehub`` is not in any requirements
file. The supported acquisition path is ``python -m loan_risk.data.download``;
copy ``application_train.csv`` from the printed cache path into ``data/raw/`` if
you use this script instead.
"""
import kagglehub

# Login to Kaggle
kagglehub.login()

# Download competition dataset
path = kagglehub.competition_download(
    "home-credit-default-risk"
)

print("Dataset path:", path)