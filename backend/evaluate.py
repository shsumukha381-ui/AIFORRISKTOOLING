"""
backend/evaluate.py — Model Evaluation Pipeline
================================================
Produces rigorous evaluation artifacts for the fraud detection model:
  1. PR-AUC with baseline comparison
  2. Confusion matrix at 0.5 threshold
  3. Cost-sensitive threshold sweep with optimal threshold identification
  4. Segmented metrics by TransactionAmt bucket and ProductCD
  5. Feature importance / LR coefficient leakage check
  6. Calibration (reliability) curve with Expected Calibration Error (ECE)

All metrics are computed on the HELD-OUT TEST SET (touched exactly once).
The validation set is used only for threshold selection via the cost sweep,
then the chosen threshold is verified on test.

Usage:
    python backend/evaluate.py
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import calibration_curve

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

# Cost constants for the threshold sweep
# FN_COST: missed fraud costs the transaction amount + a flat chargeback fee
CHARGEBACK_FEE = 25.0  # Flat fee per missed fraud (chargeback processing)
# FP_COST: each false positive costs analyst review time (flat)
FP_REVIEW_COST = 5.0  # Flat cost per false positive review

# Plot styling
plt.rcParams.update({
    "figure.facecolor": "#1a1a2e",
    "axes.facecolor": "#16213e",
    "axes.edgecolor": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#e0e0e0",
    "ytick.color": "#e0e0e0",
    "text.color": "#e0e0e0",
    "grid.color": "#2a2a4a",
    "grid.alpha": 0.5,
    "font.size": 11,
})


def load_predictions():
    """Load predicted probabilities and actuals for all splits."""
    splits = {}
    for name in ["train", "val", "test"]:
        path = os.path.join(ARTIFACTS_DIR, f"{name}_probas.parquet")
        splits[name] = pd.read_parquet(path)
    return splits


def load_feature_importance():
    """Load XGBoost feature importances and LR coefficients."""
    with open(os.path.join(ARTIFACTS_DIR, "feature_names.json")) as f:
        feature_names = json.load(f)

    with open(os.path.join(ARTIFACTS_DIR, "lr_coefficients.json")) as f:
        lr_coefs = json.load(f)

    # XGBoost feature importance from the RAW model (not the calibrated wrapper,
    # which doesn't expose feature_importances_ directly)
    import pickle
    with open(os.path.join(ARTIFACTS_DIR, "xgb_model_raw.pkl"), "rb") as f:
        xgb = pickle.load(f)

    xgb_importance = xgb.feature_importances_

    return feature_names, lr_coefs, xgb_importance


# ---------------------------------------------------------------------------
# 1. PR-AUC
# ---------------------------------------------------------------------------
def plot_pr_curve(splits):
    """Plot Precision-Recall curve with baseline comparison."""
    test = splits["test"]
    y_true = test["y_true"].values
    fraud_rate = y_true.mean()

    fig, ax = plt.subplots(figsize=(8, 6))

    # XGBoost PR curve
    prec_xgb, rec_xgb, _ = precision_recall_curve(y_true, test["y_prob_xgb"])
    pr_auc_xgb = auc(rec_xgb, prec_xgb)
    ax.plot(rec_xgb, prec_xgb, color="#7c3aed", linewidth=2,
            label=f"XGBoost (PR-AUC = {pr_auc_xgb:.4f})")

    # Logistic Regression PR curve
    prec_lr, rec_lr, _ = precision_recall_curve(y_true, test["y_prob_lr"])
    pr_auc_lr = auc(rec_lr, prec_lr)
    ax.plot(rec_lr, prec_lr, color="#06b6d4", linewidth=2, linestyle="--",
            label=f"Logistic Regression (PR-AUC = {pr_auc_lr:.4f})")

    # Random baseline = fraud rate
    ax.axhline(y=fraud_rate, color="#ef4444", linestyle=":", linewidth=1.5,
               label=f"Random baseline (precision = {fraud_rate:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve — Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACTS_DIR, "pr_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[evaluate] PR-AUC — XGBoost: {pr_auc_xgb:.4f}, LR: {pr_auc_lr:.4f}")
    return pr_auc_xgb, pr_auc_lr


# ---------------------------------------------------------------------------
# 2. Confusion Matrix at 0.5 threshold
# ---------------------------------------------------------------------------
def plot_confusion_matrix(splits):
    """Plot confusion matrix heatmap at default 0.5 threshold."""
    test = splits["test"]
    y_true = test["y_true"].values
    y_pred = (test["y_prob_xgb"].values >= 0.5).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Heatmap
    im = ax.imshow(cm, interpolation="nearest", cmap="BuPu")
    fig.colorbar(im, ax=ax, shrink=0.8)

    # Labels
    labels = ["Legitimate", "Fraud"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix @ 0.5 threshold\n"
                 f"Precision={prec:.3f}  Recall={rec:.3f}  F1={f1:.3f}",
                 fontsize=12, fontweight="bold")

    # Annotate cells with counts and percentages
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = count / total * 100
            ax.text(j, i, f"{count:,}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=12,
                    color="white" if count > cm.max() / 2 else "black")

    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACTS_DIR, "confusion_matrix.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[evaluate] Confusion matrix @ 0.5 — Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    return prec, rec, f1, cm


# ---------------------------------------------------------------------------
# 3. Cost-Sensitive Threshold Sweep
# ---------------------------------------------------------------------------
def cost_threshold_sweep(splits):
    """
    Sweep thresholds and compute total expected cost for each.
    
    Cost model:
      - FN (missed fraud): merchant loses TransactionAmt + chargeback fee
      - FP (false alarm): flat review labor cost
    
    The optimal threshold minimizes total expected cost.
    We use the VALIDATION set to find the optimal threshold,
    then verify on the TEST set.
    """
    # Find optimal threshold on validation set
    val = splits["val"]
    val_y_true = val["y_true"].values
    val_probs = val["y_prob_xgb"].values
    val_amounts = val["TransactionAmt"].values

    thresholds = np.arange(0.01, 0.91, 0.01)
    val_costs = []

    for t in thresholds:
        y_pred = (val_probs >= t).astype(int)
        fn_mask = (val_y_true == 1) & (y_pred == 0)
        fp_mask = (val_y_true == 0) & (y_pred == 1)

        fn_cost = (val_amounts[fn_mask] + CHARGEBACK_FEE).sum()
        fp_cost = fp_mask.sum() * FP_REVIEW_COST
        total_cost = fn_cost + fp_cost
        val_costs.append(total_cost)

    val_costs = np.array(val_costs)
    optimal_idx = np.argmin(val_costs)
    optimal_threshold = thresholds[optimal_idx]

    # Sanity check: if optimal lands at extremes, investigate
    if optimal_idx <= 1 or optimal_idx >= len(thresholds) - 2:
        print(f"[evaluate] WARNING: Optimal threshold ({optimal_threshold:.2f}) is at sweep extreme!")
        print(f"[evaluate] This may indicate the cost constants need adjustment.")

    # Now evaluate on TEST set
    test = splits["test"]
    y_true = test["y_true"].values
    probs = test["y_prob_xgb"].values
    amounts = test["TransactionAmt"].values

    test_costs = []
    for t in thresholds:
        y_pred = (probs >= t).astype(int)
        fn_mask = (y_true == 1) & (y_pred == 0)
        fp_mask = (y_true == 0) & (y_pred == 1)

        fn_cost = (amounts[fn_mask] + CHARGEBACK_FEE).sum()
        fp_cost = fp_mask.sum() * FP_REVIEW_COST
        total_cost = fn_cost + fp_cost
        test_costs.append(total_cost)

    test_costs = np.array(test_costs)

    # Cost at optimal threshold and at naive 0.5
    idx_optimal = np.argmin(np.abs(thresholds - optimal_threshold))
    idx_naive = np.argmin(np.abs(thresholds - 0.5))
    cost_at_optimal = test_costs[idx_optimal]
    cost_at_naive = test_costs[idx_naive]
    cost_savings = cost_at_naive - cost_at_optimal

    # Metrics at optimal threshold on test set
    y_pred_optimal = (probs >= optimal_threshold).astype(int)
    prec_opt = precision_score(y_true, y_pred_optimal, zero_division=0)
    rec_opt = recall_score(y_true, y_pred_optimal, zero_division=0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(thresholds, test_costs / 1e6, color="#7c3aed", linewidth=2, label="Total cost")

    ax.axvline(x=optimal_threshold, color="#22c55e", linestyle="--", linewidth=2,
               label=f"Optimal threshold = {optimal_threshold:.2f} (cost: ${cost_at_optimal/1e6:.2f}M)")
    ax.axvline(x=0.5, color="#ef4444", linestyle=":", linewidth=2,
               label=f"Naive 0.5 threshold (cost: ${cost_at_naive/1e6:.2f}M)")

    ax.set_xlabel("Decision Threshold", fontsize=12)
    ax.set_ylabel("Total Expected Cost ($M)", fontsize=12)
    ax.set_title(f"Cost-Sensitive Threshold Analysis — Test Set\n"
                 f"Savings: ${cost_savings/1e6:.2f}M ({cost_savings/cost_at_naive*100:.1f}%) "
                 f"by using optimal vs. naive threshold",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper center", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACTS_DIR, "cost_threshold.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[evaluate] Optimal threshold: {optimal_threshold:.2f}")
    print(f"[evaluate] Cost @ optimal: ${cost_at_optimal:,.0f}, Cost @ 0.5: ${cost_at_naive:,.0f}")
    print(f"[evaluate] Savings: ${cost_savings:,.0f} ({cost_savings/cost_at_naive*100:.1f}%)")

    return {
        "optimal_threshold": float(optimal_threshold),
        "cost_at_optimal": float(cost_at_optimal),
        "cost_at_naive": float(cost_at_naive),
        "cost_savings": float(cost_savings),
        "cost_savings_pct": float(cost_savings / cost_at_naive * 100) if cost_at_naive > 0 else 0,
        "precision_at_optimal": float(prec_opt),
        "recall_at_optimal": float(rec_opt),
    }


# ---------------------------------------------------------------------------
# 4. Segmented Metrics
# ---------------------------------------------------------------------------
def plot_segmented_metrics(splits):
    """
    Compute and plot PR-AUC and recall by:
      a) TransactionAmt bucket
      b) ProductCD category
    """
    test = splits["test"]
    y_true = test["y_true"].values
    y_prob = test["y_prob_xgb"].values
    amounts = test["TransactionAmt"].values
    products = test["ProductCD"].values

    # --- a) By TransactionAmt bucket ---
    amt_bins = [0, 50, 150, 500, float("inf")]
    amt_labels = ["$0–$50", "$50–$150", "$150–$500", "$500+"]
    amt_bucket = pd.cut(amounts, bins=amt_bins, labels=amt_labels, right=True)

    amt_metrics = {}
    for label in amt_labels:
        mask = (amt_bucket == label)
        if mask.sum() < 10 or y_true[mask].sum() < 2:
            continue
        prec_seg, rec_seg, _ = precision_recall_curve(y_true[mask], y_prob[mask])
        pr_auc_seg = auc(rec_seg, prec_seg)
        rec_at_05 = recall_score(y_true[mask], (y_prob[mask] >= 0.5).astype(int), zero_division=0)
        amt_metrics[label] = {"pr_auc": float(pr_auc_seg), "recall_at_05": float(rec_at_05),
                              "n_samples": int(mask.sum()), "n_fraud": int(y_true[mask].sum())}

    # --- b) By ProductCD ---
    product_metrics = {}
    for prod in sorted(set(products)):
        mask = (products == prod)
        if mask.sum() < 10 or y_true[mask].sum() < 2:
            continue
        prec_seg, rec_seg, _ = precision_recall_curve(y_true[mask], y_prob[mask])
        pr_auc_seg = auc(rec_seg, prec_seg)
        rec_at_05 = recall_score(y_true[mask], (y_prob[mask] >= 0.5).astype(int), zero_division=0)
        product_metrics[prod] = {"pr_auc": float(pr_auc_seg), "recall_at_05": float(rec_at_05),
                                 "n_samples": int(mask.sum()), "n_fraud": int(y_true[mask].sum())}

    # Plot — two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Amount buckets
    if amt_metrics:
        labels_a = list(amt_metrics.keys())
        prauc_a = [amt_metrics[k]["pr_auc"] for k in labels_a]
        recall_a = [amt_metrics[k]["recall_at_05"] for k in labels_a]

        x_a = np.arange(len(labels_a))
        width = 0.35
        ax1.bar(x_a - width / 2, prauc_a, width, color="#7c3aed", alpha=0.9, label="PR-AUC")
        ax1.bar(x_a + width / 2, recall_a, width, color="#06b6d4", alpha=0.9, label="Recall@0.5")
        ax1.set_xticks(x_a)
        ax1.set_xticklabels(labels_a, rotation=45, ha="right", fontsize=10)
        ax1.set_title("Metrics by Transaction Amount", fontsize=12, fontweight="bold")
        ax1.set_ylabel("Score", fontsize=11)
        ax1.legend(fontsize=9)
        ax1.set_ylim([0, 1])
        ax1.grid(True, alpha=0.3, axis="y")

    # ProductCD
    if product_metrics:
        labels_p = list(product_metrics.keys())
        prauc_p = [product_metrics[k]["pr_auc"] for k in labels_p]
        recall_p = [product_metrics[k]["recall_at_05"] for k in labels_p]

        x_p = np.arange(len(labels_p))
        ax2.bar(x_p - width / 2, prauc_p, width, color="#7c3aed", alpha=0.9, label="PR-AUC")
        ax2.bar(x_p + width / 2, recall_p, width, color="#06b6d4", alpha=0.9, label="Recall@0.5")
        ax2.set_xticks(x_p)
        ax2.set_xticklabels([f"Product {l}" for l in labels_p], rotation=45, ha="right", fontsize=10)
        ax2.set_title("Metrics by Product Category", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Score", fontsize=11)
        ax2.legend(fontsize=9)
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Segmented Model Performance — Test Set", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACTS_DIR, "segmented_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"[evaluate] Segmented metrics — {len(amt_metrics)} amount buckets, {len(product_metrics)} products")
    return {"by_amount": amt_metrics, "by_product": product_metrics}


# ---------------------------------------------------------------------------
# 5. Feature Importance / Leakage Check
# ---------------------------------------------------------------------------
def plot_feature_importance():
    """
    Plot top-20 feature importances (XGBoost) and LR coefficients.
    Purpose: leakage sanity check — if TransactionDT or a single anonymized
    column dominates disproportionately, it signals potential data leakage.
    """
    feature_names, lr_coefs, xgb_importance = load_feature_importance()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # XGBoost top-20
    xgb_sorted_idx = np.argsort(xgb_importance)[-20:]
    xgb_top_names = [feature_names[i] for i in xgb_sorted_idx]
    xgb_top_vals = xgb_importance[xgb_sorted_idx]

    ax1.barh(range(len(xgb_top_names)), xgb_top_vals, color="#7c3aed", alpha=0.9)
    ax1.set_yticks(range(len(xgb_top_names)))
    ax1.set_yticklabels(xgb_top_names, fontsize=9)
    ax1.set_xlabel("Feature Importance (gain)", fontsize=11)
    ax1.set_title("XGBoost — Top 20 Features", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3, axis="x")

    # LR top-20 by absolute coefficient
    lr_items = sorted(lr_coefs.items(), key=lambda x: abs(x[1]), reverse=True)[:20]
    lr_names = [item[0] for item in lr_items]
    lr_vals = [item[1] for item in lr_items]
    colors = ["#ef4444" if v > 0 else "#22c55e" for v in lr_vals]

    ax2.barh(range(len(lr_names)), lr_vals, color=colors, alpha=0.9)
    ax2.set_yticks(range(len(lr_names)))
    ax2.set_yticklabels(lr_names, fontsize=9)
    ax2.set_xlabel("Coefficient (red=fraud, green=legit)", fontsize=11)
    ax2.set_title("Logistic Regression — Top 20 Coefficients", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3, axis="x")

    # Check for leakage signals
    top_feature = feature_names[np.argmax(xgb_importance)]
    top_importance = xgb_importance.max()
    second_importance = np.sort(xgb_importance)[-2]
    if top_importance > 5 * second_importance:
        print(f"[evaluate] ⚠ LEAKAGE WARNING: '{top_feature}' dominates with importance "
              f"{top_importance:.4f} vs second-place {second_importance:.4f}")
    else:
        print(f"[evaluate] Feature importance check: no single-feature dominance detected")

    fig.suptitle("Feature Importance — Leakage Sanity Check", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACTS_DIR, "feature_importance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    return xgb_top_names, xgb_top_vals


# ---------------------------------------------------------------------------
# 6. Calibration / Reliability Curve
# ---------------------------------------------------------------------------
def plot_calibration_curve(splits, n_bins=10):
    """
    Plot a calibration (reliability) diagram for XGBoost and LR.

    For each model the predicted probabilities are grouped into quantile
    bins and the *observed* fraud rate in each bin is plotted against the
    *mean predicted* probability.  A perfectly calibrated model would
    track the diagonal.

    Also computes the Expected Calibration Error (ECE): the
    sample-weighted mean absolute gap between predicted and observed
    rates across bins.
    """
    test = splits["test"]
    y_true = test["y_true"].values

    fig, ax = plt.subplots(figsize=(8, 7))

    calibration_data = {}  # keyed by model name

    for label, col, color, ls in [
        ("XGBoost",              "y_prob_xgb", "#7c3aed", "-"),
        ("Logistic Regression",  "y_prob_lr",  "#06b6d4", "--"),
    ]:
        y_prob = test[col].values

        # strategy="quantile" → equal-count bins, tolerant of skewed probs
        fraction_of_positives, mean_predicted = calibration_curve(
            y_true, y_prob, n_bins=n_bins, strategy="quantile"
        )

        # Expected Calibration Error (sample-weighted)
        # Weight each bin by the number of samples that fell into it.
        bin_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
        bin_indices = np.digitize(y_prob, bin_edges[1:-1])  # 0..n_bins-1
        bin_counts = np.array([np.sum(bin_indices == i) for i in range(n_bins)])
        # Only use bins that calibration_curve actually returned
        n_actual = len(fraction_of_positives)
        bin_counts = bin_counts[:n_actual]
        total = bin_counts.sum() if bin_counts.sum() > 0 else 1
        ece = float(np.sum(bin_counts * np.abs(fraction_of_positives - mean_predicted)) / total)

        ax.plot(mean_predicted, fraction_of_positives,
                marker="o", markersize=6, color=color, linewidth=2,
                linestyle=ls, label=f"{label} (ECE = {ece:.4f})")

        calibration_data[label] = {
            "ece": ece,
            "bins": [
                {"mean_predicted": float(mp), "fraction_positive": float(fp)}
                for mp, fp in zip(mean_predicted, fraction_of_positives)
            ],
        }

    # Perfect calibration diagonal
    ax.plot([0, 1], [0, 1], color="#ef4444", linestyle=":", linewidth=1.5,
            label="Perfect calibration")

    ax.set_xlabel("Mean Predicted Probability", fontsize=12)
    ax.set_ylabel("Observed Fraction of Positives", fontsize=12)
    ax.set_title("Calibration (Reliability) Curve — Test Set", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=10)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(ARTIFACTS_DIR, "calibration_curve.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    ece_xgb = calibration_data["XGBoost"]["ece"]
    ece_lr  = calibration_data["Logistic Regression"]["ece"]
    print(f"[evaluate] Calibration — ECE XGBoost: {ece_xgb:.4f}, ECE LR: {ece_lr:.4f}")
    return calibration_data


def main():
    print("[evaluate] Loading predictions ...")
    splits = load_predictions()

    fraud_rate = splits["test"]["y_true"].mean()
    print(f"[evaluate] Test set: {len(splits['test']):,} rows, fraud rate: {fraud_rate:.4f}")

    # 1. PR-AUC
    pr_auc_xgb, pr_auc_lr = plot_pr_curve(splits)

    # 2. Confusion matrix
    prec_05, rec_05, f1_05, cm = plot_confusion_matrix(splits)

    # 3. Cost-sensitive threshold sweep
    cost_results = cost_threshold_sweep(splits)

    # 4. Segmented metrics
    segmented = plot_segmented_metrics(splits)

    # 5. Feature importance
    plot_feature_importance()

    # 6. Calibration / reliability curve
    calibration = plot_calibration_curve(splits)

    # Save summary JSON (source of truth for the frontend)
    summary = {
        "pr_auc_xgb": float(pr_auc_xgb),
        "pr_auc_lr": float(pr_auc_lr),
        "precision_at_05": float(prec_05),
        "recall_at_05": float(rec_05),
        "f1_at_05": float(f1_05),
        "fraud_rate": float(fraud_rate),
        "confusion_matrix": cm.tolist(),
        **cost_results,
        "segmented_metrics": segmented,
        "calibration": calibration,
    }

    summary_path = os.path.join(ARTIFACTS_DIR, "eval_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[evaluate] === Summary ===")
    print(f"  PR-AUC (XGBoost): {pr_auc_xgb:.4f}")
    print(f"  PR-AUC (LR):      {pr_auc_lr:.4f}")
    print(f"  Precision@0.5:     {prec_05:.4f}")
    print(f"  Recall@0.5:        {rec_05:.4f}")
    print(f"  F1@0.5:            {f1_05:.4f}")
    print(f"  Optimal threshold: {cost_results['optimal_threshold']:.2f}")
    print(f"  Cost savings:      ${cost_results['cost_savings']:,.0f} "
          f"({cost_results['cost_savings_pct']:.1f}%)")
    print(f"  ECE (XGBoost):     {calibration['XGBoost']['ece']:.4f}")
    print(f"  ECE (LR):          {calibration['Logistic Regression']['ece']:.4f}")
    print(f"\n[evaluate] All artifacts saved to {ARTIFACTS_DIR}/")


if __name__ == "__main__":
    main()
