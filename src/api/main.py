"""
Lab3 pattern: FastAPI microservice loading models:/heart-disease@champion
+ Lab4 monitoring: Prometheus metrics
"""
import time, pathlib
import mlflow, pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field
from mlflow import MlflowClient
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME, ALIAS = "heart-disease", "champion"

app = FastAPI(title="Heart Disease Prediction Service", version="1.0")

# Prometheus metrics - Lab4
PREDICTIONS = Counter("heart_predictions_total","Predictions by outcome",["outcome"])
LATENCY = Histogram("heart_inference_seconds","Latency", buckets=[.005,.01,.025,.05,.1,.25])
AGE_GAUGE = Gauge("heart_input_age","Age of last request")
app.mount("/metrics", make_asgi_app())

class Patient(BaseModel):
    age: int = Field(ge=0, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3)
    trestbps: int = Field(ge=50, le=300)
    chol: int = Field(ge=50, le=700)
    fbs: int = Field(ge=0, le=1)
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(ge=50, le=250)
    exang: int = Field(ge=0, le=1)
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4)
    thal: int = Field(ge=0, le=3)

state = {}

@app.on_event("startup")
def load_model():
    mlflow.set_tracking_uri(TRACKING_URI)
    # Try registry, fallback to exported folder for Docker
    try:
        state["model"] = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{ALIAS}")
        mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, ALIAS)
        state["version"], state["run_id"] = mv.version, mv.run_id
        print(f"Loaded {MODEL_NAME} v{mv.version}")
    except Exception as e:
        print(f"Registry load failed {e}, trying local export")
        # inside docker we bake model_export
        for cand in ["/app/model_export","model_export","./model_export"]:
            p = pathlib.Path(cand)
            if p.exists():
                state["model"] = mlflow.pyfunc.load_model(str(p))
                state["version"], state["run_id"] = "exported","exported"
                break

@app.get("/health")
def health():
    return {"status":"ok","model_loaded": "model" in state}

@app.get("/model-info")
def model_info():
    return {"model": MODEL_NAME, "alias": ALIAS, "version": state.get("version"), "run_id": state.get("run_id")}

@app.post("/predict")
def predict(patient: Patient):
    t0 = time.perf_counter()
    df = pd.DataFrame([patient.model_dump()])
    with LATENCY.time():
        pred = int(state["model"].predict(df)[0])
        proba = float(state["model"].predict_proba(df)[0,1]) if hasattr(state["model"], "predict_proba") else 0.0
    PREDICTIONS.labels(outcome=str(pred)).inc()
    AGE_GAUGE.set(patient.age)
    ms = (time.perf_counter()-t0)*1000
    return {"prediction": pred, "probability": round(proba,4), "model_version": state.get("version"), "inference_ms": round(ms,2)}
