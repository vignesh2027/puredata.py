"""Shared fixtures for puredata test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def empty_df():
    return pd.DataFrame()


@pytest.fixture
def single_col_df():
    return pd.DataFrame({"a": [1, 2, 3, np.nan, 5]})


@pytest.fixture
def all_null_col_df():
    return pd.DataFrame({"a": [np.nan] * 10, "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})


@pytest.fixture
def all_duplicate_df():
    row = {"x": 1, "y": "hello"}
    return pd.DataFrame([row] * 50)


@pytest.fixture
def dirty_df():
    """Realistic dirty DataFrame with multiple issues."""
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        "age": np.random.randint(18, 80, n).astype(float),
        "income": np.random.normal(50000, 15000, n),
        "name": ["  Alice  ", "Bob", "alice", "ALICE", "bob", "BOB"] * (n // 6) + ["Carol"] * (n % 6),
        "gender": (["Male", "male", "M", "MALE", "Female", "female", "F", "FEMALE"] * 25)[:n],
        "date_joined": (["2020-01-15", "15/01/2020", "January 15, 2020", "01-15-2020"] * 50)[:n],
        "score": np.random.uniform(0, 100, n),
        "label": np.random.randint(0, 2, n).astype(float),
    })
    # Inject nulls
    df.loc[np.random.choice(n, 20, replace=False), "age"] = np.nan
    df.loc[np.random.choice(n, 15, replace=False), "income"] = np.nan
    df.loc[np.random.choice(n, 5, replace=False), "gender"] = np.nan
    # Inject outliers
    df.loc[0, "income"] = 9_999_999
    df.loc[1, "income"] = -500_000
    # Inject duplicates
    dup_rows = df.iloc[:5].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    return df


@pytest.fixture
def train_df():
    np.random.seed(0)
    n = 500
    return pd.DataFrame({
        "feature_a": np.random.normal(10, 2, n),
        "feature_b": np.random.uniform(0, 100, n),
        "category": np.random.choice(["cat", "dog", "bird"], n),
        "label": np.random.randint(0, 2, n),
    })


@pytest.fixture
def prod_df_clean(train_df):
    """Production data with no issues — should pass all checks."""
    np.random.seed(1)
    n = 100
    return pd.DataFrame({
        "feature_a": np.random.normal(10, 2, n),
        "feature_b": np.random.uniform(0, 100, n),
        "category": np.random.choice(["cat", "dog", "bird"], n),
        "label": np.random.randint(0, 2, n),
    })


@pytest.fixture
def prod_df_drifted(train_df):
    """Production data with severe drift — should fail checks."""
    np.random.seed(99)
    n = 100
    return pd.DataFrame({
        "feature_a": np.random.normal(50, 20, n),   # massive drift
        "feature_b": np.random.uniform(200, 500, n),  # out of range
        "category": np.random.choice(["cat", "dog", "bird", "fish", "rabbit"], n),  # new values
        "label": np.random.randint(0, 2, n),
    })
