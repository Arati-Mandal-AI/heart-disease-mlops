# Heart Disease MLOps Assignment - End-to-End Pipeline
Follows BITS MLOps Labs 1-5 pattern applied to Heart Disease UCI Dataset.

## Quick Start
1. python src/download_data.py
2. python src/train_tracked.py --model rf --n-estimators 200 --max-depth 8
3. mlflow ui --backend-store-uri sqlite:///mlflow.db
4. python src/register_model.py
5. docker build -t heart-disease-api:v1 .
6. docker run -p 8000:8000 heart-disease-api:v1
7. Test: curl http://localhost:8000/health

Structure:
- src/ = data + training + api
- tests/ = unit tests
- .github/workflows/ci.yml = CI
- k8s/ = Kubernetes manifests
- monitoring/ = Prometheus + Grafana + traffic generator + PSI
