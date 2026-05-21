# Spam Email Detection - MLOps Starter

This project is a practical MLOps starter for spam detection using your dataset `spam_ham_dataset.csv`.

## Project Structure

- `src/business_problem/` - Problem definition artifacts
- `src/data_collection/` - Data source and ingestion notes/code
- `src/data_preparation/data_pipeline.py` - Dataset loading, column detection, split
- `src/feature_engineering/` - Feature logic notes/code
- `src/model_training/train.py` - Train model and log experiments with MLflow
- `src/model_evaluation/` - Evaluation reports/code
- `src/deployment/predict.py` - Predict spam/ham for new text
- `src/monitoring/` - Monitoring notes/code
- `models/` - Saved model artifacts
- `artifacts/` - Metrics and generated outputs
- `mlruns/` - MLflow local tracking store
- `params.yaml` - Training defaults

## 1) Create environment (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## 2) Train model

```powershell
python -m src.model_training.train
```

Optional custom params:

```powershell
python -m src.model_training.train --test-size 0.2 --random-state 42 --max-features 5000 --c 1.0
```

## 3) Predict new message

```powershell
python -m src.deployment.predict --text "Congratulations! You won a free iPhone. Click now!"
```

## 4) Run real-time API (POC)

Start the API server:

```powershell
python -m uvicorn src.deployment.api:app --host 127.0.0.1 --port 8000 --reload
```

Health check:

```powershell
Invoke-RestMethod -Method Get -Uri "http://127.0.0.1:8000/health"
```

Prediction request:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" -ContentType "application/json" -Body '{"text":"Free money waiting for you. Click now."}'
```

## 5) Track experiments in MLflow

```powershell
python -m mlflow ui --backend-store-uri "file:///D:/My%20Learning/MlOps-Projects/Detecting%20Spam%20Emails/mlruns" --host 127.0.0.1 --port 5000
```

Then open: http://127.0.0.1:5000

## Suggested MLOps Next Steps

1. Add `dvc` for data and model versioning.
2. Add unit tests for preprocessing and prediction.
3. Add CI pipeline (GitHub Actions) for training checks.
4. Containerize with Docker for reproducible deployment.
5. Expose prediction API with FastAPI.

## GitHub Actions Pipeline (Production-Oriented)

This repo now includes CI/CD workflows:

- `.github/workflows/ci.yml`
	- Runs on every push and pull request to `main`/`master`
	- Performs Python compile checks
	- Runs unit tests
	- Trains model and uploads artifacts (`trained-model`, `training-metrics`)

- `.github/workflows/release.yml`
	- Runs on tag pushes like `v1.0.0` or manual trigger (`workflow_dispatch`)
	- Trains model
	- Builds and pushes Docker image to GHCR:
		- `ghcr.io/<owner>/spam-email-api:<tag>`

### How to trigger release

1. Commit and push changes to GitHub.
2. Create a tag and push it:

```powershell
git tag v1.0.0
git push origin v1.0.0
```

Or trigger manually from the GitHub Actions UI using `Release API Image`.
