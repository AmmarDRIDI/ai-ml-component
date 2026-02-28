"""Dataset loading, cleaning, and deterministic splitting."""
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


def load_raw() -> pd.DataFrame:
    """Return the breast-cancer dataset as a tidy DataFrame."""
    bunch = load_breast_cancer()
    df = pd.DataFrame(bunch.data, columns=bunch.feature_names)
    df["target"] = bunch.target
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows that contain any NaN values (real cleaning step)."""
    before = len(df)
    df = df.dropna()
    after = len(df)
    if before != after:
        print(f"[data_prep] dropped {before - after} rows with NaN values.")
    return df.reset_index(drop=True)


def split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) with deterministic random state.

    *val_size* is the fraction of the **original** dataset reserved for
    validation; *test_size* is the fraction reserved for testing.
    """
    X = df.drop(columns=["target"])
    y = df["target"]

    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y
    )

    # Adjusted validation fraction relative to the remaining data
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=val_fraction, random_state=seed, stratify=y_tmp
    )

    train_df = X_train.copy()
    train_df["target"] = y_train.values

    val_df = X_val.copy()
    val_df["target"] = y_val.values

    test_df = X_test.copy()
    test_df["target"] = y_test.values

    return train_df, val_df, test_df


def load_and_prepare(
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper: load → clean → split."""
    df = load_raw()
    df = clean(df)
    return split(df, test_size=test_size, val_size=val_size, seed=seed)
