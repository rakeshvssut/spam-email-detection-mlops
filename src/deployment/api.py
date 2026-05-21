from __future__ import annotations

from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import MODEL_PATH

app = FastAPI(title="Spam Email Classifier API", version="1.0.0")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1,
                      description="Email message text to classify")


class PredictResponse(BaseModel):
    prediction: int
    label: Literal["spam", "ham"]


def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run training first.")
    return joblib.load(MODEL_PATH)


@app.get("/health")
def health() -> dict[str, str]:
    if not MODEL_PATH.exists():
        return {"status": "warning", "message": "Model not trained yet"}
    return {"status": "ok", "message": "Service is healthy"}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    try:
        model = _load_model()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    pred = int(model.predict([payload.text])[0])
    label = "spam" if pred == 1 else "ham"
    return PredictResponse(prediction=pred, label=label)
