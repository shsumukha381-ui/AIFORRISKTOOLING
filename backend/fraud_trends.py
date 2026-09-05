"""
backend/fraud_trends.py -- Aggregate Fraud Trend Analysis & Spike Detection
=============================================================================
Computes time-windowed fraud rates from the held-out test set and flags
statistical spikes (bins where the fraud rate exceeds a rolling baseline
by more than z_threshold standard deviations).

This is pure aggregation on already-labeled data -- no model inference needed.

Usage:
    python backend/fraud_trends.py
"""

import os
import json
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

# Minimum transactions per bin for reliable fraud-rate estimates.
# Bins below this threshold produce noisy fraud rates and unreliable
# z-scores -- they are excluded from spike detection (is_spike forced
# to False) and flagged as low_volume in the output so the frontend
# can dim/annotate them rather than report a misleading spike off a
# handful of transactions.
MIN_BIN_VOLUME = 20


def compute_fraud_trends(test_df, window_hours=1, z_threshold=2.0,
                         rolling_window=24, min_periods=5):
    """
    Compute hourly fraud rates and detect spikes via z-score.

    Parameters
    ----------
    test_df : DataFrame
        Must include: TransactionDT (seconds from reference), isFraud
        (real label), ProductCD.
    window_hours : int
        Bin width in hours (default 1).
    z_threshold : float
        Number of standard deviations above the rolling baseline to flag
        as a spike.
    rolling_window : int
        Number of trailing bins for the rolling baseline (default 24).
    min_periods : int
        Minimum bins required for a valid rolling statistic.

    Returns
    -------
    overall : DataFrame
        Per-hour-bin aggregated fraud rate with spike flags.
    per_product : DataFrame
        Same, broken out by ProductCD.
    """
    df = test_df.copy()

    # Convert TransactionDT (seconds from dataset reference) into
    # hour-of-dataset bins.  window_hours lets us use wider bins if
    # the dataset is sparse.
    seconds_per_bin = 3600 * window_hours
    df["hour_bin"] = (df["TransactionDT"] // seconds_per_bin).astype(int)

    # ------------------------------------------------------------------
    # Overall trend
    # ------------------------------------------------------------------
    overall = (
        df.groupby("hour_bin")
        .agg(n_transactions=("isFraud", "size"),
             n_fraud=("isFraud", "sum"))
        .sort_index()
    )
    overall["fraud_rate"] = overall["n_fraud"] / overall["n_transactions"]

    # Flag low-volume bins (see MIN_BIN_VOLUME docstring above)
    overall["low_volume"] = overall["n_transactions"] < MIN_BIN_VOLUME

    # Rolling baseline: trailing window, shift(1) excludes the current
    # bin to prevent look-ahead leakage in the z-score calculation.
    overall["baseline_mean"] = (
        overall["fraud_rate"]
        .shift(1)
        .rolling(rolling_window, min_periods=min_periods)
        .mean()
    )
    overall["baseline_std"] = (
        overall["fraud_rate"]
        .shift(1)
        .rolling(rolling_window, min_periods=min_periods)
        .std()
    )

    # Z-score: how many standard deviations above the rolling baseline
    overall["z_score"] = (
        (overall["fraud_rate"] - overall["baseline_mean"])
        / overall["baseline_std"]
    )

    # A bin is a spike only if it exceeds the z-threshold AND has enough
    # volume to be statistically meaningful.
    overall["is_spike"] = (
        (overall["z_score"] > z_threshold)
        & (~overall["low_volume"])
        & overall["baseline_mean"].notna()
    )

    # ------------------------------------------------------------------
    # Per-ProductCD trend (same logic, grouped)
    # ------------------------------------------------------------------
    per_product = (
        df.groupby(["hour_bin", "ProductCD"])
        .agg(n_transactions=("isFraud", "size"),
             n_fraud=("isFraud", "sum"))
        .sort_index()
    )
    per_product["fraud_rate"] = (
        per_product["n_fraud"] / per_product["n_transactions"]
    )
    per_product["low_volume"] = per_product["n_transactions"] < MIN_BIN_VOLUME

    # Apply rolling baseline + z-score per ProductCD group
    def _add_rolling_stats(group):
        group = group.sort_index()
        group["baseline_mean"] = (
            group["fraud_rate"]
            .shift(1)
            .rolling(rolling_window, min_periods=min_periods)
            .mean()
        )
        group["baseline_std"] = (
            group["fraud_rate"]
            .shift(1)
            .rolling(rolling_window, min_periods=min_periods)
            .std()
        )
        group["z_score"] = (
            (group["fraud_rate"] - group["baseline_mean"])
            / group["baseline_std"]
        )
        group["is_spike"] = (
            (group["z_score"] > z_threshold)
            & (~group["low_volume"])
            & group["baseline_mean"].notna()
        )
        return group

    per_product = (
        per_product
        .groupby("ProductCD", group_keys=False)
        .apply(_add_rolling_stats)
    )

    return overall, per_product


def _safe_value(v):
    """Convert numpy/pandas types to JSON-safe Python types."""
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 6) if not np.isnan(v) else None
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, float) and np.isnan(v):
        return None
    return v


def save_trends(overall, per_product):
    """Persist computed trends to JSON for the frontend."""

    overall_records = []
    for hour_bin, row in overall.iterrows():
        overall_records.append({
            "hour_bin": int(hour_bin),
            "n_transactions": _safe_value(row["n_transactions"]),
            "n_fraud": _safe_value(row["n_fraud"]),
            "fraud_rate": _safe_value(row["fraud_rate"]),
            "low_volume": _safe_value(row["low_volume"]),
            "baseline_mean": _safe_value(row["baseline_mean"]),
            "baseline_std": _safe_value(row["baseline_std"]),
            "z_score": _safe_value(row["z_score"]),
            "is_spike": _safe_value(row["is_spike"]),
        })

    # Per-product records (multi-index: hour_bin, ProductCD)
    per_product_records = []
    for (hour_bin, product_cd), row in per_product.iterrows():
        per_product_records.append({
            "hour_bin": int(hour_bin),
            "ProductCD": str(product_cd),
            "n_transactions": _safe_value(row["n_transactions"]),
            "n_fraud": _safe_value(row["n_fraud"]),
            "fraud_rate": _safe_value(row["fraud_rate"]),
            "low_volume": _safe_value(row["low_volume"]),
            "baseline_mean": _safe_value(row["baseline_mean"]),
            "baseline_std": _safe_value(row["baseline_std"]),
            "z_score": _safe_value(row["z_score"]),
            "is_spike": _safe_value(row["is_spike"]),
        })

    # Summary statistics for quick display
    n_spikes = sum(1 for r in overall_records if r["is_spike"])
    n_bins = len(overall_records)
    spike_records = [r for r in overall_records if r["is_spike"]]

    output = {
        "n_bins": n_bins,
        "n_spikes_overall": n_spikes,
        "window_hours": None,  # will be set by caller if known
        "overall": overall_records,
        "per_product": per_product_records,
        "spikes_summary": spike_records,
    }

    output_path = os.path.join(ARTIFACTS_DIR, "fraud_trends.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"[fraud_trends] Saved {n_bins} bins ({n_spikes} spikes) to {output_path}")
    return output


def get_transactions_for_bin(test_df, hour_bin, window_hours=4):
    """
    Return the actual test-set rows whose TransactionDT falls into the
    given hour_bin.

    Parameters
    ----------
    test_df : DataFrame
        The full test set (must include TransactionDT and isFraud).
    hour_bin : int
        The bin identifier (as stored in fraud_trends.json).
    window_hours : int
        Bin width in hours — must match the value used when
        compute_fraud_trends() was called.

    Returns
    -------
    DataFrame
        Rows belonging to that bin, with original columns intact.
    """
    seconds_per_bin = 3600 * window_hours
    df = test_df.copy()
    df["hour_bin"] = (df["TransactionDT"] // seconds_per_bin).astype(int)
    return df[df["hour_bin"] == hour_bin].drop(columns=["hour_bin"])


def main():
    print("[fraud_trends] Loading test set ...")
    df_test = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "test.parquet"))
    print(f"[fraud_trends] Test set: {len(df_test):,} rows, "
          f"fraud rate: {df_test['isFraud'].mean():.4f}")

    dt_range = df_test["TransactionDT"].max() - df_test["TransactionDT"].min()
    span_hours = dt_range / 3600
    print(f"[fraud_trends] Time span: {span_hours:.1f} hours "
          f"({span_hours / 24:.1f} days)")

    # Use 4-hour bins to keep the chart readable (~250 bins for 1000 hours)
    # and ensure most bins have enough volume for reliable rates.
    # 1-hour bins would give ~1005 bins averaging 117 tx/bin, which is fine
    # statistically but very noisy visually.  4-hour is a good tradeoff.
    window_hours = 4
    avg_per_bin = len(df_test) / (span_hours / window_hours)
    print(f"[fraud_trends] Using {window_hours}-hour bins "
          f"(~{avg_per_bin:.0f} transactions/bin expected)")

    overall, per_product = compute_fraud_trends(
        df_test, window_hours=window_hours, z_threshold=2.0
    )

    n_low_vol = (overall["low_volume"]).sum()
    n_spikes = overall["is_spike"].sum()
    print(f"[fraud_trends] {len(overall)} bins total, "
          f"{n_low_vol} low-volume, {n_spikes} spikes detected")

    result = save_trends(overall, per_product)
    result["window_hours"] = window_hours
    # Re-save with window_hours populated
    output_path = os.path.join(ARTIFACTS_DIR, "fraud_trends.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print("[fraud_trends] Done.")


if __name__ == "__main__":
    main()
