# Spam Email Detection - MLOps POC

This project trains a text classifier for email categorization, logs experiments in MLflow, and serves predictions through a FastAPI endpoint.

## What This Project Does

1. Reads and prepares the dataset from `spam_ham_dataset.csv`.
2. Trains a scikit-learn pipeline (`TF-IDF + LogisticRegression`).
3. Computes evaluation metrics and saves them to `artifacts/metrics.json`.
4. Saves the trained model to `models/spam_classifier.joblib`.
5. Logs runs, params, and metrics to MLflow (`mlruns/`).
6. Serves real-time predictions with FastAPI + Uvicorn.

## Project Structure

- `src/data_preparation/data_pipeline.py`: Data loading, column detection, balancing, split.
- `src/model_training/train.py`: Training entrypoint and MLflow logging.
- `src/deployment/predict.py`: CLI prediction entrypoint.
- `src/deployment/api.py`: FastAPI real-time inference service.
- `models/`: Saved model artifact (`.joblib`).
- `artifacts/`: Generated metrics JSON.
- `mlruns/`: MLflow tracking files.
- `tests/`: Test suite.

## Prerequisites

- Windows PowerShell
- Python 3.13 (recommended in this setup)

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train Model

```powershell
python -m src.model_training.train
```

Optional hyperparameters:

```powershell
python -m src.model_training.train --test-size 0.2 --random-state 42 --max-features 5000 --c 1.0
```

## Run CLI Prediction

```powershell
python -m src.deployment.predict --text "Congratulations! You won a free iPhone. Click now!"
```

## Run API (POC)

Start server:

```powershell
python -m uvicorn src.deployment.api:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"
```

Prediction:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body '{"text":"Free money waiting for you. Click now."}'
```

## MLflow Tracking

Start MLflow UI using the same backend path used by training:

```powershell
python -m mlflow ui --backend-store-uri "file:///D:/My%20Learning/MlOps-Projects/Detecting%20Spam%20Emails/mlruns" --host 127.0.0.1 --port 5000
```

Open in browser:

- http://127.0.0.1:5000

## Notes

- If your dataset has more than two classes, training supports multiclass metrics automatically.
- The MLflow filesystem deprecation message is a warning, not a training failure.
- The model artifact extension is `.joblib` because training uses `joblib.dump`.

## Troubleshooting

If `mlflow` module is missing:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If API cannot find model:

1. Run training first.
2. Confirm `models/spam_classifier.joblib` exists.

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```
