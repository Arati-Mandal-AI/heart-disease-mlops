import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib

NUMERIC = ["age","trestbps","chol","thalach","oldpeak"]
CATEGORICAL = ["sex","cp","fbs","restecg","exang","slope","ca","thal"]

def get_preprocessor():
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, NUMERIC),
        ("cat", cat_pipe, CATEGORICAL)
    ])
    return pre

def load_data(path="data/heart_disease.csv"):
    df = pd.read_csv(path)
    # Cleveland data: target 0-4, binarize to 0 = no disease, 1 = disease
    if df["target"].max() > 1:
        df["target"] = (df["target"] > 0).astype(int)
    X = df.drop(columns=["target"])
    y = df["target"]
    return X, y
