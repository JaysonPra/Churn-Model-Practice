FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY mlflow.db .
COPY mlruns/ ./mlruns/

ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

EXPOSE 8000

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]