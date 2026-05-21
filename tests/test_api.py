from __future__ import annotations

from src.deployment.api import PredictRequest, predict


class DummyModel:
    def predict(self, items: list[str]) -> list[int]:
        return [1 if "free" in items[0].lower() else 0]


def test_predict_endpoint(monkeypatch) -> None:
    from src.deployment import api as api_module

    monkeypatch.setattr(api_module, "_load_model", lambda: DummyModel())
    body = predict(PredictRequest(text="Free iPhone for you")).model_dump()

    assert body["prediction"] == 1
    assert body["label"] == "spam"
