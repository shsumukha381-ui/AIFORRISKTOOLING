"""
backend/benchmark_latency.py — Single-Row Inference Latency Benchmark
======================================================================
Measures how long a single predict_proba call takes through the full
calibrated model (XGBoost + isotonic regression wrapper), which is the
exact path a production checkout-time call would take.

Reports p50, p95, p99, mean, and max latencies in milliseconds, saved
to artifacts/latency_benchmark.json for consumption by the frontend
dashboard and README.

Usage:
    python backend/benchmark_latency.py
"""

import os
import sys
import time
import json
import platform
import numpy as np
import pandas as pd
import pickle

# Ensure project root is on the path so train_model imports work
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

# Import CalibratedModel so pickle can deserialize the calibrated model
from backend.train_model import CalibratedModel  # noqa: F401

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

N_RUNS = 1000
N_WARMUP = 20


def get_hardware_note():
    """Build a human-readable hardware description for the benchmark."""
    cpu = platform.processor() or platform.machine()
    system = platform.system()
    python_ver = platform.python_version()
    return f"{cpu}, {system}, Python {python_ver}, no GPU"


def load_model_and_test_data():
    """
    Load the calibrated model and produce preprocessed test-set feature
    vectors, mirroring the exact pipeline from train_model.py.
    """
    # Load the calibrated model (XGBoost + isotonic wrapper)
    with open(os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"), "rb") as f:
        calibrated_model = pickle.load(f)

    # Load encoder and column config
    with open(os.path.join(ARTIFACTS_DIR, "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)

    with open(os.path.join(ARTIFACTS_DIR, "column_config.json")) as f:
        config = json.load(f)

    # Load the raw test split (before feature encoding)
    df_test = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "test.parquet"))

    numeric_cols = config["numeric_cols"]
    categorical_cols = config["categorical_cols"]

    # Reproduce the same feature encoding as train_model.py:
    # numeric values (already imputed) + one-hot encoded categoricals
    X_num = df_test[numeric_cols].values
    X_cat = encoder.transform(df_test[categorical_cols])
    X_test = np.hstack([X_num, X_cat])

    return calibrated_model, X_test


def run_benchmark():
    print("[benchmark] Loading calibrated model and test data ...")
    calibrated_model, X_test = load_model_and_test_data()
    print(f"[benchmark] Test set shape: {X_test.shape}")

    # Sample rows (or use all if fewer than N_RUNS)
    n_available = X_test.shape[0]
    n_sample = min(N_RUNS, n_available)
    rng = np.random.RandomState(42)
    sample_indices = rng.choice(n_available, size=n_sample, replace=False)
    sample_rows = X_test[sample_indices]

    # Warm-up runs: exclude from measurement.
    # First calls are slower due to lazy initialization, memory allocation,
    # and potential JIT-like effects in the XGBoost C++ backend.
    print(f"[benchmark] Running {N_WARMUP} warm-up iterations ...")
    for i in range(min(N_WARMUP, n_sample)):
        row = sample_rows[i].reshape(1, -1)
        calibrated_model.predict_proba(row)

    # Timed runs: single-row inference (one transaction at a time),
    # matching the real checkout-time use case.
    print(f"[benchmark] Running {n_sample} timed single-row inferences ...")
    latencies_ms = []
    for i in range(n_sample):
        row = sample_rows[i].reshape(1, -1)
        start = time.perf_counter()
        calibrated_model.predict_proba(row)
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000)

    latencies_ms = np.array(latencies_ms)

    hardware_note = get_hardware_note()

    result = {
        "n_runs": int(len(latencies_ms)),
        "p50_ms": round(float(np.percentile(latencies_ms, 50)), 3),
        "p95_ms": round(float(np.percentile(latencies_ms, 95)), 3),
        "p99_ms": round(float(np.percentile(latencies_ms, 99)), 3),
        "mean_ms": round(float(latencies_ms.mean()), 3),
        "max_ms": round(float(latencies_ms.max()), 3),
        "hardware_note": hardware_note,
    }

    # Sanity check: XGBoost single-row inference is typically low single-digit
    # to low double-digit ms. Flag if p95 is unexpectedly high.
    if result["p95_ms"] > 100:
        print(f"[benchmark] WARNING: p95 latency is {result['p95_ms']:.1f} ms "
              f"-- unexpectedly high for XGBoost single-row inference.")
        print(f"[benchmark]   Possible causes: warm-up not excluded, batch overhead, "
              f"or isotonic calibration overhead.")
    else:
        print(f"[benchmark] OK: Latency looks reasonable for single-row XGBoost inference.")

    # Save results
    output_path = os.path.join(ARTIFACTS_DIR, "latency_benchmark.json")
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n[benchmark] === Results ===")
    print(f"  Runs:     {result['n_runs']}")
    print(f"  p50:      {result['p50_ms']:.3f} ms")
    print(f"  p95:      {result['p95_ms']:.3f} ms")
    print(f"  p99:      {result['p99_ms']:.3f} ms")
    print(f"  Mean:     {result['mean_ms']:.3f} ms")
    print(f"  Max:      {result['max_ms']:.3f} ms")
    print(f"  Hardware: {hardware_note}")
    print(f"\n[benchmark] Saved to {output_path}")

    return result


if __name__ == "__main__":
    run_benchmark()
