from __future__ import annotations

import argparse

import joblib

from src.config import MODEL_PATH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict spam vs ham")
    parser.add_argument("--text", type=str, required=True, help="Email message to classify")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}. Run training first.")

    model = joblib.load(MODEL_PATH)
    pred = int(model.predict([args.text])[0])
    label = "spam" if pred == 1 else "ham"

    print({"prediction": pred, "label": label})


if __name__ == "__main__":
    main()
