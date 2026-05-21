from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data_preparation.data_pipeline import load_and_prepare_data


def test_load_and_prepare_data_returns_split(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "message": [
                "win free cash",
                "project update",
                "meeting at 10",
                "claim your prize",
                "invoice attached",
                "urgent reward",
            ],
            "label": ["spam", "ham", "ham", "spam", "ham", "spam"],
        }
    )
    csv_path = tmp_path / "sample.csv"
    df.to_csv(csv_path, index=False)

    x_train, x_test, y_train, y_test = load_and_prepare_data(
        str(csv_path), test_size=0.33, random_state=42
    )

    assert len(x_train) > 0
    assert len(x_test) > 0
    assert len(y_train) == len(x_train)
    assert len(y_test) == len(x_test)
    assert set(y_train.unique()).issubset({0, 1})
    assert set(y_test.unique()).issubset({0, 1})
