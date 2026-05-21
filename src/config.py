from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "spam_ham_dataset.csv"
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"
MODEL_PATH = MODELS_DIR / "spam_classifier.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"
