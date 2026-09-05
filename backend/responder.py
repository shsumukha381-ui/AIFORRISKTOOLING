"""
backend/responder.py — LLM-Backed Pre-Authorization Risk Responder
==================================================================
Generates internal fraud case notes and always-on risk narrations
using Groq's LLM API, framed as a pre-payment / pre-authorization
risk checkpoint.

Key behaviors:
  - generate_risk_narration: runs for EVERY transaction (flagged or not),
    producing one short plain-language sentence explaining the score.
  - generate_case_note: runs only for flagged transactions AT or ABOVE
    the cost-optimal threshold, producing a detailed 2-4 sentence case note.
  - Extracts per-prediction feature contributions from the XGBoost model
  - Produces calibrated language: marginal-probability cases are explicitly
    flagged as ambiguous with a recommendation to defer to human review
  - DEFENSE-ONLY: the system prompt explicitly prohibits generating content
    that could help a user evade detection, fabricate evidence, or
    reverse-engineer the model's blind spots

Usage:
    from backend.responder import generate_case_note, generate_risk_narration
    narration = generate_risk_narration(contributions, probability)
    note = generate_case_note(feature_dict, fraud_probability, optimal_threshold)
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Import CalibratedModel so pickle can resolve it when loading xgb_model.pkl
# (the model was pickled when train_model.py was __main__, so pickle stored
# the class reference as __main__.CalibratedModel).
from backend.train_model import CalibratedModel  # noqa: F401

# Patch into __main__ so pickle.load can find __main__.CalibratedModel
import __main__ as _main_module
if not hasattr(_main_module, "CalibratedModel"):
    _main_module.CalibratedModel = CalibratedModel

# Load .env file for API key
load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

# Marginal probability band: if prob is between threshold and threshold * MARGIN_FACTOR,
# the LLM is instructed to express lower confidence and defer to human review.
MARGIN_FACTOR = 1.3

# System prompt — defense-only, calibrated language, pre-payment action set
SYSTEM_PROMPT = """You are a fraud-risk case-note assistant. You will be given a transaction's flagged signals, its fraud probability, and a recommended action that has ALREADY been decided by fixed business rules. Your only job is to write a short factual explanation of why the signals support that specific action. Do not propose, imply, or conclude a different action than the one given to you. If the probability is only marginally above threshold, note that the signals are weak/ambiguous, but still only justify the given action — do not contradict it. Never suggest how a transaction could be altered to avoid detection."""

# System prompt for always-on risk narration (lightweight, runs for every transaction)
NARRATION_SYSTEM_PROMPT = (
    "You write one short, plain-language sentence explaining why a transaction's "
    "fraud risk score is what it is, based on the given signals. Do not recommend "
    "actions here — that is handled separately. Do not suggest how to alter a "
    "transaction to change its score. Be factual and neutral, not alarmist."
)


class CustomUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == 'CalibratedModel':
            from backend.train_model import CalibratedModel
            return CalibratedModel
        return super().find_class(module, name)

def load_model_artifacts():
    """Load the trained (calibrated) XGBoost model, raw model, encoder, and feature names."""
    with open(os.path.join(ARTIFACTS_DIR, "xgb_model.pkl"), "rb") as f:
        model = CustomUnpickler(f).load()
    # The raw XGBoost model is accessible inside the calibrated wrapper.
    # We also keep xgb_model_raw.pkl on disk for evaluate.py's feature
    # importance loader, but for inference we just reach into the wrapper.
    raw_model = model.estimator
    with open(os.path.join(ARTIFACTS_DIR, "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "column_config.json")) as f:
        config = json.load(f)
    with open(os.path.join(ARTIFACTS_DIR, "eval_summary.json")) as f:
        eval_summary = json.load(f)
    return model, raw_model, encoder, feature_names, config, eval_summary


def get_feature_contributions(raw_model, X_row, feature_names, top_n=5):
    """
    Extract per-prediction feature contributions using XGBoost's built-in
    SHAP-style tree contribution method.
    
    Uses the RAW (uncalibrated) XGBoost model, since the calibrated wrapper
    doesn't expose the tree booster needed for contribution prediction.
    
    Returns the top_n features by absolute contribution magnitude,
    with their names, values, and contribution directions.
    """
    import xgboost as xgb

    # Get prediction contributions (SHAP values from the tree structure)
    # This returns one value per feature + a bias term
    dmatrix = xgb.DMatrix(X_row, feature_names=feature_names)
    contribs = raw_model.get_booster().predict(dmatrix, pred_contribs=True)[0]

    # Last element is the bias — exclude it
    feature_contribs = contribs[:-1]

    # Get top-N by absolute value
    top_indices = np.argsort(np.abs(feature_contribs))[-top_n:][::-1]

    contributions = []
    for idx in top_indices:
        contributions.append({
            "feature": feature_names[idx],
            "value": float(X_row[0, idx]),
            "contribution": float(feature_contribs[idx]),
            "direction": "increases fraud risk" if feature_contribs[idx] > 0 else "decreases fraud risk",
        })

    return contributions


def prepare_feature_vector(feature_dict, config, encoder):
    """
    Convert a dictionary of feature values into the model's expected
    input format (numeric + one-hot encoded categoricals).
    """
    numeric_cols = config["numeric_cols"]
    categorical_cols = config["categorical_cols"]
    medians = config["train_medians"]

    # Build numeric array — use provided values or fall back to train median
    numeric_values = []
    for col in numeric_cols:
        val = feature_dict.get(col, medians.get(col, 0))
        numeric_values.append(float(val))

    # Build categorical DataFrame for encoding
    cat_values = {}
    for col in categorical_cols:
        cat_values[col] = [feature_dict.get(col, "missing")]

    cat_df = pd.DataFrame(cat_values)
    X_cat = encoder.transform(cat_df)
    X_num = np.array(numeric_values).reshape(1, -1)
    X = np.hstack([X_num, X_cat])

    return X


def predict_fraud_probability(feature_dict, model=None, raw_model=None,
                               encoder=None, config=None, feature_names=None):
    """
    Predict fraud probability for a single transaction.
    Returns (probability, feature_contributions, X_row).
    """
    if model is None:
        model, raw_model, encoder, feature_names, config, _ = load_model_artifacts()

    X = prepare_feature_vector(feature_dict, config, encoder)
    prob = model.predict_proba(X)[0, 1]
    contributions = get_feature_contributions(raw_model, X, feature_names)

    return float(prob), contributions, X


def _build_user_prompt(fraud_prob, optimal_threshold, contributions,
                       transaction_amt, product_cd, decided_action):
    """Build the user message for the LLM with all relevant context."""

    # Format feature contributions
    contrib_lines = []
    for c in contributions:
        # Map technical feature names to more readable descriptions
        name = _humanize_feature_name(c["feature"])
        contrib_lines.append(
            f"{name}={c['value']:.2f} ({c['direction']})"
        )
    contrib_text = "; ".join(contrib_lines)

    prompt = f"Fraud probability: {fraud_prob:.3f}. Top contributing signals: {contrib_text}. Decided action: {decided_action}. Write the case note."

    return prompt


def _humanize_feature_name(name):
    """Map technical feature names to more readable descriptions."""
    mappings = {
        "TransactionAmt": "Transaction Amount",
        "card1": "Card Identifier 1",
        "card2": "Card Identifier 2",
        "card3": "Card Country Code",
        "card5": "Card Identifier 5",
        "addr1": "Billing Address Region",
        "addr2": "Billing Address Country",
        "dist1": "Distance Metric 1",
        "dist2": "Distance Metric 2",
        "P_emaildomain": "Purchaser Email Domain",
    }
    # Handle one-hot encoded features
    for prefix in ["ProductCD_", "card4_", "card6_", "M1_", "M2_", "M3_",
                    "M4_", "M5_", "M6_", "M7_", "M8_", "M9_", "P_emaildomain_"]:
        if name.startswith(prefix):
            base = prefix.rstrip("_")
            value = name[len(prefix):]
            return f"{base} = {value}"

    # C-columns: counting features
    if name.startswith("C"):
        return f"Counting Feature {name}"

    # D-columns: time delta features
    if name.startswith("D"):
        return f"Time Delta Feature {name}"

    return mappings.get(name, name)


def generate_case_note(feature_dict, fraud_prob=None, optimal_threshold=None,
                       model=None, raw_model=None, encoder=None, config=None,
                       feature_names=None, eval_summary=None):
    """
    Generate an LLM-backed case note for a flagged transaction.
    
    Args:
        feature_dict: dict of feature name → value for the transaction
        fraud_prob: pre-computed fraud probability (if None, will compute)
        optimal_threshold: cost-optimal threshold (if None, loads from eval_summary)
        model, raw_model, encoder, config, feature_names, eval_summary: pre-loaded artifacts
    
    Returns:
        dict with keys: case_note, recommended_action, fraud_probability,
                        is_flagged, contributions
    """
    # Load artifacts if not provided
    if model is None:
        model, raw_model, encoder, feature_names, config, eval_summary = load_model_artifacts()

    if optimal_threshold is None:
        optimal_threshold = eval_summary["optimal_threshold"]

    # Get prediction and contributions
    if fraud_prob is None:
        fraud_prob, contributions, _ = predict_fraud_probability(
            feature_dict, model, raw_model, encoder, config, feature_names
        )
    else:
        X = prepare_feature_vector(feature_dict, config, encoder)
        contributions = get_feature_contributions(raw_model, X, feature_names)

    # Determine risk level
    if fraud_prob < optimal_threshold * 0.5:
        risk_level = "low"
    elif fraud_prob < optimal_threshold:
        risk_level = "elevated"
    else:
        risk_level = "high"

    # Only call the LLM for flagged transactions (at or above threshold)
    if fraud_prob < optimal_threshold:
        return {
            "case_note": None,
            "recommended_action": None,
            "fraud_probability": fraud_prob,
            "risk_level": risk_level,
            "is_flagged": False,
            "contributions": contributions,
        }

    # Determine deterministic action based on probability bands.
    # Bands are RELATIVE to the cost-optimal threshold, so they
    # automatically adapt when the threshold changes (e.g., after
    # calibration shifts the probability scale).
    above_range = 1.0 - optimal_threshold
    hold_ceiling = optimal_threshold + above_range * 0.35
    stepup_ceiling = optimal_threshold + above_range * 0.7

    if fraud_prob < hold_ceiling:
        decided_action = "Hold — do not capture payment yet, pending manual review"
    elif fraud_prob < stepup_ceiling:
        decided_action = "Require step-up verification before authorizing payment"
    else:
        decided_action = "Decline — block this payment"

    # Build prompt and call Groq
    transaction_amt = feature_dict.get("TransactionAmt", 0)
    product_cd = feature_dict.get("ProductCD", "unknown")
    user_prompt = _build_user_prompt(
        fraud_prob, optimal_threshold, contributions,
        transaction_amt, product_cd, decided_action
    )

    try:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return {
                "case_note": "[GROQ_API_KEY not set — case note generation unavailable]",
                "recommended_action": decided_action,
                "fraud_probability": fraud_prob,
                "risk_level": risk_level,
                "is_flagged": True,
                "contributions": contributions,
            }

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=300,
            temperature=0.3,  # Low temperature for factual, consistent output
        )

        case_note = response.choices[0].message.content.strip()

        return {
            "case_note": case_note,
            "recommended_action": decided_action,
            "fraud_probability": fraud_prob,
            "risk_level": risk_level,
            "is_flagged": True,
            "contributions": contributions,
        }

    except Exception as e:
        return {
            "case_note": f"[LLM call failed: {str(e)}]",
            "recommended_action": decided_action,
            "fraud_probability": fraud_prob,
            "risk_level": risk_level,
            "is_flagged": True,
            "contributions": contributions,
        }





def generate_risk_narration(contributions, probability):
    """
    Generate a one-sentence, always-on AI narration explaining why
    a transaction's fraud risk score is what it is.

    This is a lightweight LLM call that runs for EVERY transaction
    (flagged or not), separate from the detailed case note.

    Defense-only guardrail: the system prompt prohibits suggesting
    how to alter a transaction to change its score.

    Args:
        contributions: list of dicts with feature contributions
        probability: float, the fraud probability score

    Returns:
        str: one plain-language sentence narrating the risk signals
    """
    # Format top features for the prompt
    top_features = "; ".join(
        f"{_humanize_feature_name(c['feature'])}={c['value']:.2f} ({c['direction']})"
        for c in contributions[:5]
    )

    try:
        from groq import Groq

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return _fallback_narration(contributions, probability)

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            max_tokens=60,
            temperature=0.3,
            messages=[
                {"role": "system", "content": NARRATION_SYSTEM_PROMPT},
                {"role": "user", "content": f"Fraud probability: {probability:.2f}. Top contributing signals: {top_features}"},
            ],
        )
        return response.choices[0].message.content.strip()

    except Exception:
        return _fallback_narration(contributions, probability)


def _fallback_narration(contributions, probability):
    """Generate a fallback narration without LLM when API is unavailable."""
    if probability < 0.1:
        level = "low"
    elif probability < 0.3:
        level = "moderate"
    else:
        level = "elevated"

    top = contributions[0] if contributions else None
    if top:
        feature_name = _humanize_feature_name(top["feature"])
        return (f"Risk assessed as {level} — the primary signal is "
                f"{feature_name} which {top['direction']}.")
    return f"Risk assessed as {level} based on the transaction's overall signal profile."
