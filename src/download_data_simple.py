import pandas as pd, pathlib
pathlib.Path("data").mkdir(exist_ok=True)
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
cols = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal","target"]
df = pd.read_csv(url, names=cols, na_values=['?'])
df.to_csv("data/heart_disease.csv", index=False)
print(f"Saved {len(df)} rows, columns {list(df.columns)}")
