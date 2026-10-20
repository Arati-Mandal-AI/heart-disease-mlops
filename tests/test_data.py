import pandas as pd
from src.preprocess import load_data, get_preprocessor

def test_data_loads():
    X, y = load_data("data/heart_disease.csv")
    assert len(X) > 100
    assert "age" in X.columns

def test_preprocessor():
    X, y = load_data("data/heart_disease.csv")
    pre = get_preprocessor()
    Xt = pre.fit_transform(X.head(20))
    assert Xt.shape[0] == 20
