"""Tests for data_prep.py – verifies the cleaning step removes NaNs."""
import numpy as np
import pandas as pd
import pytest

from ml_component.src.data_prep import clean, load_raw, load_and_prepare


def test_clean_removes_nans():
    """Rows with NaN values must be dropped by clean()."""
    df = load_raw()
    # Inject NaNs into a copy
    dirty = df.copy()
    dirty.iloc[0, 0] = np.nan
    dirty.iloc[5, 3] = np.nan

    cleaned = clean(dirty)

    assert cleaned.isna().sum().sum() == 0, "clean() should remove all NaN rows"
    assert len(cleaned) == len(df) - 2, "Exactly the two injected rows should be dropped"


def test_clean_no_nans_unchanged():
    """If there are no NaNs the dataset length should not change."""
    df = load_raw()
    cleaned = clean(df.copy())
    assert len(cleaned) == len(df)


def test_load_and_prepare_split_sizes():
    """Train/val/test sizes should be consistent with requested fractions."""
    train_df, val_df, test_df = load_and_prepare(test_size=0.2, val_size=0.2, seed=42)
    total = len(train_df) + len(val_df) + len(test_df)
    assert total > 0
    # Test split should be roughly 20 %
    assert abs(len(test_df) / total - 0.2) < 0.05


def test_splits_contain_target_and_features():
    """Each split DataFrame must contain both feature columns and the 'target' column."""
    train_df, val_df, test_df = load_and_prepare(seed=42)
    for split in (train_df, val_df, test_df):
        assert "target" in split.columns
        assert len(split.columns) > 1, "Split must also contain feature columns"
