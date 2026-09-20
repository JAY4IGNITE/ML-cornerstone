import kagglehub

# Login to Kaggle
kagglehub.login()

# Download competition dataset
path = kagglehub.competition_download(
    "home-credit-default-risk"
)

print("Dataset path:", path)