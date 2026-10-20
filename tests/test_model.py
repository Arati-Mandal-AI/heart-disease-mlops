from src.preprocess import load_data, get_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def test_model_train_predict():
    X, y = load_data("data/heart_disease.csv")
    pre = get_preprocessor()
    pipe = Pipeline([("pre", pre), ("clf", RandomForestClassifier(n_estimators=10, random_state=42))])
    pipe.fit(X.head(100), y.head(100))
    preds = pipe.predict(X.head(5))
    assert len(preds) == 5
    proba = pipe.predict_proba(X.head(5))
    assert proba.shape[1] == 2
