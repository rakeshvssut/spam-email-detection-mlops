from __future__ import annotations

import json

from src.model_training.train import main


def test_train_main_creates_metrics_file(monkeypatch) -> None:

    monkeypatch.setattr("sys.argv", ["train.py", "--test-size", "0.2", "--random-state", "42"])

    # Smoke test entrypoint; project dataset is used by design.
    main()

    with open("artifacts/metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
