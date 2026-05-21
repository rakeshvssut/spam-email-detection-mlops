from __future__ import annotations

import argparse
import json

import joblib
import mlflow
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

from src.config import ARTIFACTS_DIR, DATA_PATH, METRICS_PATH, MLRUNS_DIR, MODEL_PATH, MODELS_DIR
from src.data_preparation.data_pipeline import load_and_prepare_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train spam classifier")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--c", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLRUNS_DIR.resolve().as_uri())
    mlflow.set_experiment("spam-email-classifier")

    x_train, x_test, y_train, y_test = load_and_prepare_data(
        str(DATA_PATH), test_size=args.test_size, random_state=args.random_state
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=args.max_features)),
            ("clf", LogisticRegression(C=args.c, max_iter=500)),
        ]
    )

    with mlflow.start_run():
        mlflow.log_params(
            {
                "test_size": args.test_size,
                "random_state": args.random_state,
                "max_features": args.max_features,
                "c": args.c,
            }
        )

        model.fit(x_train, y_train)
        preds = model.predict(x_test)

        unique_labels = np.unique(y_test)
        is_binary = len(unique_labels) == 2
        avg = "binary" if is_binary else "weighted"

        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "precision": float(precision_score(y_test, preds, average=avg, zero_division=0)),
            "recall": float(recall_score(y_test, preds, average=avg, zero_division=0)),
            "f1": float(f1_score(y_test, preds, average=avg, zero_division=0)),
        }

        if not is_binary:
            metrics["precision_macro"] = float(
                precision_score(y_test, preds, average="macro",
                                zero_division=0)
            )
            metrics["recall_macro"] = float(
                recall_score(y_test, preds, average="macro", zero_division=0)
            )
            metrics["f1_macro"] = float(
                f1_score(y_test, preds, average="macro", zero_division=0))

        mlflow.log_metrics(metrics)

        joblib.dump(model, MODEL_PATH)
        with open(METRICS_PATH, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

        mlflow.log_artifact(str(MODEL_PATH))
        mlflow.log_artifact(str(METRICS_PATH))

    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")


if __name__ == "__main__":
    main()
