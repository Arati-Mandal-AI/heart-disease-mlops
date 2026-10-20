"""
Task 1: Data Acquisition - Heart Disease UCI
Downloads from UCI repo, saves to data/heart_disease.csv
"""
import pathlib, pandas as pd
from ucimlrepo import fetch_ucirepo

# Alternative direct fetch - using ucimlrepo package, fallback to sklearn fetch
def main():
    pathlib.Path("data").mkdir(exist_ok=True)
    try:
        # UCI Heart Disease dataset id=45
        from sklearn.datasets import fetch_openml
        # Heart Disease UCI via openml for robustness
        df = fetch_openml(name="heart-disease", version=1, as_frame=True).frame
        # Some versions call target 'class' - normalize
        print(f"Fetched via openml: {df.shape}")
        df.to_csv("data/heart_disease.csv", index=False)
    except Exception as e:
        print(f"OpenML fetch failed {e}, trying manual URL...")
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
        cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
        df = pd.read_csv(url, names=cols)
        # Handle missing '?' as NaN
        df.replace('?', pd.NA, inplace=True)
        df.to_csv("data/heart_disease.csv", index=False)
    print("Saved data/heart_disease.csv")

if __name__ == "__main__":
    main()
