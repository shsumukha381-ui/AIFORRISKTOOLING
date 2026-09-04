"""
backend/train_model.py — Model Training Pipeline
=================================================
Trains two models on the IEEE-CIS fraud detection data:
  1. Logistic Regression baseline (interpretability check)
  2. XGBoost main model (production-grade performance)

Both models use one-hot encoded categoricals and the imputed numerics
from load_data.py. Saves trained models, encoder, and predicted
probabilities for all splits to the artifacts directory.

Usage:
    python backend/train_model.py
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
from xgboost import XGBClassifier


class CalibratedModel:
    """
    Lightweight wrapper: raw XGBoost + isotonic calibrator.

    sklearn 1.8 removed cv='prefit' from CalibratedClassifierCV, so we
    calibrate manually.  The wrapper exposes:
      - predict_proba(X) → calibrated probabilities
      - .estimator        → the raw XGBClassifier (for SHAP / importance)
    """

    def __init__(self, estimator, calibrator):
        self.estimator = estimator
        self.calibrator = calibrator

    @classmethod
    def fit_calibration(cls, xgb_model, X_val, y_val):
        """Fit an isotonic calibrator on validation-set raw probabilities."""
        raw_probs = xgb_model.predict_proba(X_val)[:, 1]
        iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        iso.fit(raw_probs, y_val)
        return cls(estimator=xgb_model, calibrator=iso)

    def predict_proba(self, X):
        """Return calibrated [P(legit), P(fraud)] array."""
        import numpy as np
        raw = self.estimator.predict_proba(X)[:, 1]
        calibrated = self.calibrator.predict(raw)
        return np.column_stack([1.0 - calibrated, calibrated])

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")


def load_splits_and_config():
    """Load cached Parquet splits and column configuration."""
    df_train = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "train.parquet"))
    df_val = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "val.parquet"))
    df_test = pd.read_parquet(os.path.join(ARTIFACTS_DIR, "test.parquet"))

    with open(os.path.join(ARTIFACTS_DIR, "column_config.json")) as f:
        config = json.load(f)

    return df_train, df_val, df_test, config


def prepare_features(df_train, df_val, df_test, config):
    """
    One-hot encode categoricals and combine with numerics.
    Fits encoder on training data only to prevent leakage.
    """
    numeric_cols = config["numeric_cols"]
    categorical_cols = config["categorical_cols"]
    target = config["target"]

    # One-hot encode categoricals — fit on train, transform all splits.
    # handle_unknown='ignore' ensures unseen categories in val/test
    # don't crash the pipeline (they just get all-zeros for that feature).
    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    encoder.fit(df_train[categorical_cols])

    feature_sets = {}
    for name, df in [("train", df_train), ("val", df_val), ("test", df_test)]:
        # Numeric features (already imputed, no NaNs)
        X_num = df[numeric_cols].values

        # One-hot encoded categorical features
        X_cat = encoder.transform(df[categorical_cols])

        # Combine
        X = np.hstack([X_num, X_cat])
        y = df[target].values

        feature_sets[name] = (X, y)

    # Build feature name list for interpretability
    cat_feature_names = encoder.get_feature_names_out(categorical_cols).tolist()
    all_feature_names = numeric_cols + cat_feature_names

    return feature_sets, encoder, all_feature_names


def train_logistic_regression(X_train, y_train):
    """
    Logistic Regression baseline with class-weight balancing.
    Purpose: interpretability check — if LR coefficients show a single
    feature dominating, that's a leakage signal to investigate.
    """
    print("[train] Training Logistic Regression baseline ...")
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )),
    ])
    lr_pipeline.fit(X_train, y_train)
    print(f"[train] LR training complete — {lr_pipeline.named_steps['lr'].n_iter_[0]} iterations")
    return lr_pipeline


def train_xgboost(X_train, y_train, X_val, y_val):
    """
    XGBoost main model with early stopping on the validation set.
    scale_pos_weight handles class imbalance (~28:1 negative:positive ratio).
    eval_metric='aucpr' aligns training with the competition's primary metric.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos

    print(f"[train] Training XGBoost — scale_pos_weight={scale_pos_weight:.2f} "
          f"(neg={n_neg:,}, pos={n_pos:,}) ...")

    xgb = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        random_state=42,
        n_jobs=-1,
        verbosity=1,
    )

    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=50,
    )

    best_iter = xgb.best_iteration
    best_score = xgb.best_score
    print(f"[train] XGBoost training complete — best iteration: {best_iter}, "
          f"best val aucpr: {best_score:.4f}")

    return xgb


def save_artifacts(lr, xgb_calibrated, xgb_raw, encoder, feature_names,
                   feature_sets, df_train, df_val, df_test, config):
    """Save models, encoder, feature names, and predictions."""

    # Save models
    with open(os.path.join(ARTIFACTS_DIR, "lr_model.pkl"), "wb") as f:
        pickle.dump(lr, f)
    # Calibrated model — used for all downstream scoring (predict_proba)
    with open(os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"), "wb") as f:
        pickle.dump(xgb_calibrated, f)
    # Raw model — used for feature importance and SHAP contributions
    with open(os.path.join(ARTIFACTS_DIR, "xgb_model_raw.pkl"), "wb") as f:
        pickle.dump(xgb_raw, f)
    with open(os.path.join(ARTIFACTS_DIR, "encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    # Save feature names
    with open(os.path.join(ARTIFACTS_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    # Save LR coefficients for feature importance analysis
    lr_coefs = dict(zip(feature_names, lr.named_steps['lr'].coef_[0].tolist()))
    with open(os.path.join(ARTIFACTS_DIR, "lr_coefficients.json"), "w") as f:
        json.dump(lr_coefs, f, indent=2)

    # Generate and save predicted probabilities for all splits.
    # These are used by evaluate.py and the frontend without retraining.
    # IMPORTANT: XGBoost probabilities come from the CALIBRATED model.
    for split_name, df_orig in [("train", df_train), ("val", df_val), ("test", df_test)]:
        X, y = feature_sets[split_name]

        proba_lr = lr.predict_proba(X)[:, 1]
        proba_xgb = xgb_calibrated.predict_proba(X)[:, 1]

        # Build predictions DataFrame with columns needed for evaluation:
        # - y_true: actual fraud label
        # - y_prob_xgb / y_prob_lr: predicted fraud probabilities (calibrated)
        # - TransactionAmt: needed for cost-sensitive threshold analysis
        # - ProductCD: needed for segmented metrics
        pred_df = pd.DataFrame({
            "y_true": y,
            "y_prob_xgb": proba_xgb,
            "y_prob_lr": proba_lr,
            "TransactionAmt": df_orig["TransactionAmt"].values,
            "ProductCD": df_orig["ProductCD"].values,
        })
        pred_df.to_parquet(os.path.join(ARTIFACTS_DIR, f"{split_name}_probas.parquet"), index=False)

    print(f"[train] Saved models, encoder, feature names, and predictions to {ARTIFACTS_DIR}/")

    # Save 5 demo cases from the test set for the frontend:
    # 1 clearly fraudulent, 1 clearly legitimate, 1 borderline, 1 small legit, 1 moderate flag
    _save_demo_cases(df_test, feature_sets["test"], xgb_calibrated, config)


def _save_demo_cases(df_test, test_data, model, config):
    """Select 5 representative test-set rows for the frontend demo dropdown."""
    X_test, y_test = test_data
    probas = model.predict_proba(X_test)[:, 1]

    # We'll pick cases based on fraud probability and actual label
    df_demo_source = df_test.copy()
    df_demo_source["_fraud_prob"] = probas
    df_demo_source["_idx"] = range(len(df_demo_source))

    demos = []

    # Case 1: Clearly legitimate (low prob, actually not fraud)
    legit = df_demo_source[(df_demo_source["isFraud"] == 0) & (df_demo_source["_fraud_prob"] < 0.05)]
    if len(legit) > 0:
        demos.append(("Demo 1: Typical legitimate purchase", legit.sample(1, random_state=42)))

    # Case 2: Clearly fraudulent (high prob, actually fraud)
    fraud = df_demo_source[(df_demo_source["isFraud"] == 1) & (df_demo_source["_fraud_prob"] > 0.8)]
    if len(fraud) > 0:
        demos.append(("Demo 2: High-value suspicious transaction", fraud.sample(1, random_state=42)))
    else:
        # Fallback: highest-prob fraud case
        fraud_all = df_demo_source[df_demo_source["isFraud"] == 1].nlargest(1, "_fraud_prob")
        demos.append(("Demo 2: High-value suspicious transaction", fraud_all))

    # Case 3: Borderline / ambiguous (prob near 0.3-0.6, could be either)
    borderline = df_demo_source[
        (df_demo_source["_fraud_prob"] >= 0.15) &
        (df_demo_source["_fraud_prob"] <= 0.55)
    ]
    if len(borderline) > 0:
        demos.append(("Demo 3: Borderline / ambiguous signals", borderline.sample(1, random_state=42)))

    # Case 4: Small legitimate transaction
    small_legit = df_demo_source[
        (df_demo_source["isFraud"] == 0) &
        (df_demo_source["_fraud_prob"] < 0.03) &
        (df_demo_source["TransactionAmt"] < 30)
    ]
    if len(small_legit) > 0:
        demos.append(("Demo 4: Small legitimate transaction", small_legit.sample(1, random_state=42)))

    # Case 5: Flagged unusual pattern (above threshold but moderate)
    flagged = df_demo_source[
        (df_demo_source["_fraud_prob"] >= 0.5) &
        (df_demo_source["_fraud_prob"] <= 0.85)
    ]
    if len(flagged) > 0:
        demos.append(("Demo 5: Flagged unusual pattern", flagged.sample(1, random_state=42)))

    # Save demo cases
    feature_cols = config["all_feature_cols"]
    demo_list = []
    for label, row_df in demos:
        row = row_df.iloc[0]
        demo_entry = {
            "label": label,
            "features": {col: _convert_value(row[col]) for col in feature_cols},
            "actual_fraud": int(row["isFraud"]),
        }
        demo_list.append(demo_entry)

    with open(os.path.join(ARTIFACTS_DIR, "demo_cases.json"), "w") as f:
        json.dump(demo_list, f, indent=2)

    print(f"[train] Saved {len(demo_list)} demo cases for frontend")


def _convert_value(val):
    """Convert numpy/pandas types to JSON-serializable Python types."""
    if isinstance(val, (np.integer,)):
        return int(val)
    elif isinstance(val, (np.floating,)):
        return float(val)
    elif isinstance(val, (np.bool_,)):
        return bool(val)
    else:
        return val


def main():
    print("[train] Loading cached splits ...")
    df_train, df_val, df_test, config = load_splits_and_config()

    print("[train] Preparing features ...")
    feature_sets, encoder, feature_names = prepare_features(
        df_train, df_val, df_test, config
    )
    print(f"[train] Feature matrix shape: {feature_sets['train'][0].shape}")

    X_train, y_train = feature_sets["train"]
    X_val, y_val = feature_sets["val"]

    # Model 1: Logistic Regression baseline
    lr = train_logistic_regression(X_train, y_train)

    # Model 2: XGBoost main model
    xgb_raw = train_xgboost(X_train, y_train, X_val, y_val)

    # Model 3: Calibrate XGBoost with isotonic regression on validation set.
    # The raw model systematically overstates fraud probabilities (ECE ~0.17).
    # Isotonic calibration corrects this so predicted probabilities are
    # trustworthy as actual fraud rates — essential for the cost-threshold
    # math to be meaningful.
    print("[train] Calibrating XGBoost with isotonic regression on validation set ...")
    xgb_calibrated = CalibratedModel.fit_calibration(xgb_raw, X_val, y_val)
    print("[train] Calibration complete.")

    # Save everything
    save_artifacts(lr, xgb_calibrated, xgb_raw, encoder, feature_names,
                   feature_sets, df_train, df_val, df_test, config)

    print("[train] Pipeline complete.")


if __name__ == "__main__":
    main()
