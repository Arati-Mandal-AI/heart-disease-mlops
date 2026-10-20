"""
Lab1 pattern: CartVista churn -> adapted to Heart Disease
Logs params, metrics, model to MLflow sqlite store
"""
import argparse, hashlib, pathlib
import pandas as pd, mlflow, mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from preprocess import get_preprocessor, load_data
from sklearn.pipeline import Pipeline

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "heart-disease"

def build_model(kind, args):
    if kind == "logreg":
        return LogisticRegression(C=args.C, max_iter=1000)
    if kind == "rf":
        return RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42)
    raise ValueError(kind)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["logreg","rf"], required=True)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--max-depth", type=int, default=6)
    args = ap.parse_args()

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    X, y = load_data()
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    pre = get_preprocessor()

    with mlflow.start_run(run_name=f"{args.model}-run"):
        mlflow.log_param("model_kind", args.model)
        if args.model == "logreg":
            mlflow.log_param("C", args.C)
        else:
            mlflow.log_params({"n_estimators": args.n_estimators, "max_depth": args.max_depth})
        
        data_fp = hashlib.md5(pathlib.Path("data/heart_disease.csv").read_bytes()).hexdigest()[:10] if pathlib.Path("data/heart_disease.csv").exists() else "no-file"
        mlflow.log_param("data_fingerprint", data_fp)

        clf = build_model(args.model, args)
        pipe = Pipeline([("preprocess", pre), ("clf", clf)])
        pipe.fit(X_tr, y_tr)

        proba = pipe.predict_proba(X_te)[:,1]
        preds = pipe.predict(X_te)

        metrics = {
            "test_auc": roc_auc_score(y_te, proba),
            "test_accuracy": accuracy_score(y_te, preds),
            "test_f1": f1_score(y_te, preds),
            "test_precision": precision_score(y_te, preds),
            "test_recall": recall_score(y_te, preds)
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(pipe, artifact_path="model")
        print(f"Logged run {mlflow.active_run().info.run_id[:8]} auc={metrics['test_auc']:.4f} acc={metrics['test_accuracy']:.4f}")

if __name__ == "__main__":
    main()
