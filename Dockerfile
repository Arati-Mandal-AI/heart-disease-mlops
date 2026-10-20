FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --default-timeout=100 --retries 10 --no-cache-dir -r requirements.txt
COPY model_export/ /app/model_export/
COPY src/ /app/src/
COPY mlflow.db /app/mlflow.db
COPY src/api/main.py /app/main.py
EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
