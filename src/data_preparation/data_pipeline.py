from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

COMMON_TEXT_COLUMNS = ["text", "message", "email", "content", "body"]
COMMON_LABEL_COLUMNS = ["label", "class", "target", "category", "type"]


def _find_column(df: pd.DataFrame, candidates: list[str], fallback_kind: str) -> str:
    lowered = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]

    if fallback_kind == "label":
        for col in df.columns:
            if df[col].nunique(dropna=True) <= 10:
                return col
    elif fallback_kind == "text":
        object_cols = [col for col in df.columns if df[col].dtype == "object"]
        if object_cols:
            return object_cols[0]

    raise ValueError(f"Could not infer {fallback_kind} column from dataset")


def _balance_dataset(df: pd.DataFrame, label_col: str, random_state: int) -> pd.DataFrame:
    """Downsample majority class to match minority class count."""
    counts = df[label_col].value_counts()
    minority_count = counts.min()

    balanced_parts = [
        group.sample(n=minority_count, random_state=random_state)
        for _, group in df.groupby(label_col)
    ]
    return pd.concat(balanced_parts).reset_index(drop=True)


def load_and_prepare_data(
    csv_path: str, test_size: float, random_state: int
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("Dataset is empty")

    text_col = _find_column(df, COMMON_TEXT_COLUMNS, fallback_kind="text")
    label_col = _find_column(df, COMMON_LABEL_COLUMNS, fallback_kind="label")

    # Normalize labels before balancing so groupby works on clean values
    df[label_col] = df[label_col].astype(str).str.strip().str.lower()

    # Balance: downsample majority class (Ham) to match minority class (Spam)
    df = _balance_dataset(df, label_col, random_state)

    x = df[text_col].astype(str).fillna("")
    y_raw = df[label_col]

    mapping = {"spam": 1, "ham": 0, "0": 0, "1": 1}
    y = y_raw.map(mapping)
    if y.isna().any():
        # Fallback: encode distinct labels deterministically.
        labels = sorted(y_raw.unique())
        encoded = {label: i for i, label in enumerate(labels)}
        y = y_raw.map(encoded)

    print(f"Balanced dataset: {y.value_counts().to_dict()}")

    return train_test_split(x, y, test_size=test_size, random_state=random_state, stratify=y)
