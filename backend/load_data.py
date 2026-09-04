"""
backend/load_data.py — IEEE-CIS Fraud Detection Data Loader
============================================================
Loads the IEEE-CIS Fraud Detection dataset (Kaggle / Vesta Corporation),
applies documented missing-value imputation, performs a time-based
train/val/test split, and caches processed splits to Parquet.

Dataset: https://www.kaggle.com/c/ieee-fraud-detection
- Real anonymized e-commerce transactions (~590K rows, ~3.5% fraud rate)
- We use only train_transaction.csv (skip the identity join for simplicity)

Usage:
    python backend/load_data.py
"""

import os
import json
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_PATH = os.path.join(PROJECT_ROOT, "ieee-fraud-detection", "train_transaction.csv")
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

# Columns to keep — curated subset of the 394 available columns.
# Rationale: include the target, temporal key, transaction amount, card info,
# address/distance features, counting features (C1-C14), time-delta features
# (D1-D15), match flags (M1-M9), and email domain. Skip V-columns (339 anon
# Vesta-engineered features) to keep the model interpretable and training fast.
TARGET = "isFraud"
TEMPORAL_KEY = "TransactionDT"

NUMERIC_COLS = [
    "TransactionAmt",
    "card1", "card2", "card3", "card5",
    "addr1", "addr2",
    "dist1", "dist2",
    "C1", "C2", "C3", "C4", "C5", "C6", "C7",
    "C8", "C9", "C10", "C11", "C12", "C13", "C14",
    "D1", "D2", "D3", "D4", "D5", "D6", "D7",
    "D8", "D9", "D10", "D11", "D12", "D13", "D14", "D15",
]

CATEGORICAL_COLS = [
    "ProductCD",
    "card4", "card6",
    "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9",
    "P_emaildomain",
]

ALL_FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS
ALL_COLS = [TARGET, TEMPORAL_KEY] + ALL_FEATURE_COLS

# Split ratios (by sorted time order)
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
# TEST_FRAC = 0.20 (remainder)


def load_and_process():
    """Load raw CSV, subset columns, impute missing values, split by time."""

    print(f"[load_data] Reading raw data from {RAW_DATA_PATH} ...")
    df = pd.read_csv(RAW_DATA_PATH, usecols=ALL_COLS)
    print(f"[load_data] Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"[load_data] Fraud rate: {df[TARGET].mean():.4f} ({df[TARGET].sum():,} fraud / {len(df):,} total)")

    # ------------------------------------------------------------------
    # Time-based split (ordered by TransactionDT)
    # ------------------------------------------------------------------
    # Rationale: In production fraud detection, we always predict *forward*
    # in time. A random split would leak future transaction patterns into
    # training data, inflating apparent performance. A temporal split gives
    # an honest estimate of how the model will perform on unseen future data.
    df = df.sort_values(TEMPORAL_KEY).reset_index(drop=True)
    n = len(df)
    train_end = int(n * TRAIN_FRAC)
    val_end = int(n * (TRAIN_FRAC + VAL_FRAC))

    df_train = df.iloc[:train_end].copy()
    df_val = df.iloc[train_end:val_end].copy()
    df_test = df.iloc[val_end:].copy()

    print(f"[load_data] Split sizes — train: {len(df_train):,}, val: {len(df_val):,}, test: {len(df_test):,}")
    print(f"[load_data] Fraud rates — train: {df_train[TARGET].mean():.4f}, "
          f"val: {df_val[TARGET].mean():.4f}, test: {df_test[TARGET].mean():.4f}")

    # ------------------------------------------------------------------
    # Missing-value imputation
    # ------------------------------------------------------------------
    # Strategy:
    #   - Numeric: fill with MEDIAN computed on the TRAINING set only.
    #     Using train-only statistics prevents information leakage from
    #     validation/test sets.
    #   - Categorical: fill with the literal string "missing".
    #     This preserves the information that the value was absent, which
    #     can itself be a fraud signal (e.g., missing email domain).
    #
    # We do NOT drop any rows. The IEEE-CIS dataset has structurally sparse
    # columns by design (e.g., dist1/dist2 are missing for ~95% of rows).
    # Dropping rows with any NaN would eliminate most of the dataset and
    # severely bias the fraud rate.

    # Compute medians on training data only
    train_medians = df_train[NUMERIC_COLS].median().to_dict()

    # Apply imputation to all splits
    for split_df in [df_train, df_val, df_test]:
        for col in NUMERIC_COLS:
            split_df[col] = split_df[col].fillna(train_medians[col])
        for col in CATEGORICAL_COLS:
            split_df[col] = split_df[col].fillna("missing")

    # Verify no NaNs remain in feature columns
    for name, split_df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        nan_count = split_df[ALL_FEATURE_COLS].isna().sum().sum()
        assert nan_count == 0, f"[load_data] BUG: {nan_count} NaNs remain in {name} split!"

    print("[load_data] Imputation complete — zero NaNs in all splits")

    return df_train, df_val, df_test, train_medians


def save_splits(df_train, df_val, df_test, train_medians):
    """Cache processed splits to Parquet and save column config."""
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    df_train.to_parquet(os.path.join(ARTIFACTS_DIR, "train.parquet"), index=False)
    df_val.to_parquet(os.path.join(ARTIFACTS_DIR, "val.parquet"), index=False)
    df_test.to_parquet(os.path.join(ARTIFACTS_DIR, "test.parquet"), index=False)

    # Save column configuration and imputation values so the frontend
    # and responder can reconstruct a feature vector for new transactions
    # without re-reading the training data.
    config = {
        "target": TARGET,
        "temporal_key": TEMPORAL_KEY,
        "numeric_cols": NUMERIC_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "all_feature_cols": ALL_FEATURE_COLS,
        "train_medians": {k: float(v) for k, v in train_medians.items()},
    }
    config_path = os.path.join(ARTIFACTS_DIR, "column_config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"[load_data] Saved processed splits and config to {ARTIFACTS_DIR}/")


def load_cached_splits():
    """Load previously cached Parquet splits (fast path for the app)."""
    df_train = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "train.parquet"))
    df_val = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "val.parquet"))
    df_test = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "test.parquet"))

    with open(os.path.join(ARTIFACTS_DIR, "column_config.json")) as f:
        config = json.load(f)

    return df_train, df_val, df_test, config


def splits_exist():
    """Check whether cached splits already exist on disk."""
    return all(
        os.path.exists(os.path.join(ARTIFACTS_DIR, f))
        for f in ["train.parquet", "val.parquet", "test.parquet", "column_config.json"]
    )


if __name__ == "__main__":
    if splits_exist():
        print("[load_data] Cached splits found — loading from disk ...")
        df_train, df_val, df_test, config = load_cached_splits()
        print(f"[load_data] Loaded train={len(df_train):,}, val={len(df_val):,}, test={len(df_test):,}")
    else:
        print("[load_data] No cached splits — processing raw data ...")
        df_train, df_val, df_test, train_medians = load_and_process()
        save_splits(df_train, df_val, df_test, train_medians)
    print("[load_data] Done.")
