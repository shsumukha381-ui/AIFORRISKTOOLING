# 🛡️ Fraud Detector & Risk Case-Note Responder

**A real-time, cost-aware fraud risk check for e-commerce payments, with a defense-only AI explanation layer for risk analysts.**

## The Problem
Merchants lose billions annually to online payment fraud. However, naive machine learning models often optimize for the wrong metric (overall accuracy) and ignore the highly asymmetric costs of fraud: a false negative (missed fraud) costs the merchant the lost item plus chargeback fees, while a false positive (false alarm) only costs the time required for manual review. Furthermore, most public fraud detection demos rely on perfectly balanced, synthetic datasets that produce unrealistic 99% accuracy rates, setting false expectations for production environments.

## What We Built
This project is a complete, production-styled pipeline addressing those realities:
- **A per-transaction fraud detector** (XGBoost, calibrated) trained and evaluated strictly on real, imbalanced IEEE-CIS/Vesta e-commerce transaction data.
- **A cost-sensitive decision layer** that sets hold/review thresholds by mathematically minimizing real expected costs, rather than relying on a default 0.5 probability cutoff.
- **A defense-only AI responder** (Groq-backed) that generates internal case notes and answers analyst questions, but never decides actions itself and is strictly guardrailed against helping evade detection.
- **An aggregate Fraud Trends view** that detects statistical spikes in fraud rates over time, complementing the real-time per-transaction detector.

## Live Demo
- **[LIVE DEMO LINK]**(https://aiforsisktooling.streamlit.app/)
- **[DEMO VIDEO LINK]**(https://drive.google.com/file/d/1pESfCszb_o1ZpAGdRwijw5xJkgATNWko/view?usp=sharing)

## Headline Results

*All metrics are computed on the strictly held-out test set (touched exactly once).*

| Metric | Value |
|--------|-------|
| **PR-AUC (XGBoost)** | 0.4562 (Base rate: 0.0344 &mdash; **~13.2x lift over random**) |
| **Precision @ Optimal Threshold** | 0.1208 |
| **Recall @ Optimal Threshold** | 0.8157 |
| **Cost-Optimal Threshold** | 0.03 |
| **Cost Savings (vs. naive 0.5)** | **$315,715 (55.2% reduction in expected cost)** |
| **Calibration (ECE)** | 0.1657 (raw) &rarr; **0.0060 (calibrated)** |
| **Inference Latency (single-row)** | **p50: 1.271 ms** / **p95: 2.581 ms** |

*(Hardware note for latency: AMD64 Family 25 Model 68, Windows, Python 3.14.0, no GPU)*

**Why are these numbers credible?** 
1. **Precision is intentionally low at the optimal threshold.** In this domain, a false negative (missing a $500 fraudulent purchase) is vastly more expensive than a false positive (a $5 manual review). The cost-minimizing threshold therefore aggressively flags borderline cases, willingly accepting a lower precision to achieve an 81.5% recall.
2. **We use PR-AUC, not ROC-AUC.** Because the dataset is highly imbalanced (~3.4% fraud), ROC-AUC would look misleadingly high (often >0.90) by rewarding the model for correctly identifying easy true-negatives. Precision-Recall AUC focuses strictly on the minority class that actually matters.

## Architecture

`Data Input` &rarr; `XGBoost Detector` &rarr; `Isotonic Calibration` &rarr; `Cost-Optimal Threshold` &rarr; `Deterministic Action` &rarr; *(if flagged)* `AI Case Note / Chatbot`

**Crucial architectural choice:** The LLM **never decides the action**&mdash;it only explains a decision that has already been made by deterministic statistical rules. Early in development, we found a bug where the LLM-authored text occasionally contradicted the UI's displayed action. By isolating the AI to a purely explanatory layer acting on the deterministic model's output, we eliminate the risk of AI hallucination overriding a critical business decision.

## Data
We use the **IEEE-CIS Fraud Detection Dataset** (Vesta Corporation, via Kaggle), containing real, anonymized e-commerce transactions. 
- **Size:** ~590,540 rows total.
- **Labels:** Real `isFraud` labels (overall ~3.5% fraud rate).
- **Integrity:** No synthetic, SMOTE, or simulated data was used in the final model. 
- **Splits:** We strictly use a time-based train/val/test split. Fraud detection in production is a forward-in-time prediction problem; a random shuffle would leak future patterns into the training set and falsely inflate performance.

## Scoping Decisions
- **Why fraud, not returns or chargebacks?** Real, labeled chargeback dispute-outcome data is not publicly available, and returns-risk represents a different loss profile. We chose to go deep on one realistic, rigorously evaluated pipeline rather than building shallow proof-of-concepts for several.
- **Why a per-transaction detector?** Per-transaction scoring is what enables real-time *pre-payment* decisioning. The Fraud Trends tab was built specifically to also satisfy the "fraud-spike detector" track prompt, ensuring both individual and aggregate angles are covered.
- **Retrospective vs. Streaming:** The current Fraud Trends tab performs a historical analysis over the test set's timestamps. It is not a live streaming system (though a production version would simply add continuous ingestion against a live transaction stream).
- **Geographic Generalization:** While IEEE-CIS is not India-specific, the fraud signal classes modeled (card-not-present indicators, address/device mismatches, transaction velocity) generalize directly to Indian e-commerce and BFSI card-not-present fraud patterns.

## Defense-Only Guardrails
- **Anti-Evasion:** The responder and chatbot strictly refuse to suggest how a transaction could be altered, structured, or timed to lower its score. This is actively tested, not just prompted-for.
- **Architectural Sandbox:** The deterministic action-decision pipeline prevents the AI from ever overriding or modifying a risk decision.
- **Grounded Responses:** The chatbot is contextually grounded *only* in the provided real metrics and transaction data, explicitly stating when it lacks information rather than inventing it.

## Known Limitations / Future Work
- **Concept Drift:** Fraud patterns evolve rapidly. In production, this model would require continuous monitoring and periodic retraining.
- **Cold-Start Problem:** New customers or cards with no history receive median-imputed features, inherently reducing signal quality.
- **Anonymized Columns:** Many provided features (`C`, `D`, `M`, `V` columns) are masked by Vesta. While we built a feature glossary to aid interpretability, domain-specific insight is limited compared to unmasked internal data.
- **Abuse-Ring / Graph Detection:** Coordinated abuse rings are poorly captured by per-transaction features. Graph-based network analysis was explicitly scoped out here but is flagged as high-value future work.

## Setup and Run Instructions

### 1. Prerequisites
- Python 3.9+
- A Kaggle account and API token (`kaggle.json`) to download the dataset.
- A Groq API key for the AI explanation layer.

### 2. Environment Setup
```bash
pip install -r requirements.txt
cp .env.example .env
# Edit .env and insert your GROQ_API_KEY=your_key_here
```

### 3. Pipeline Execution
Run the commands in this exact sequence to reproduce the artifacts and launch the app:

```bash
# 1. Download & process data, cache splits
python backend/load_data.py

# 2. Train LR baseline & XGBoost, apply isotonic calibration
python backend/train_model.py

# 3. Generate evaluation metrics, cost thresholds, and plots
python backend/evaluate.py

# 4. Measure single-row inference latency
python backend/benchmark_latency.py

# 5. Compute aggregate fraud trends + spikes
python backend/fraud_trends.py

# 6. Launch the Streamlit application
python -m streamlit run frontend/app.py
```

## Project Structure

```
./
    .env
    .env.example
    .gitignore
    README.md
    requirements.txt
    artifacts/
        calibration_curve.png
        column_config.json
        confusion_matrix.png
        cost_threshold.png
        demo_cases.json
        encoder.pkl
        eval_summary.json
        feature_importance.png
        feature_names.json
        fraud_trends.json
        latency_benchmark.json
        lr_coefficients.json
        lr_model.pkl
        pr_curve.png
        segmented_metrics.png
        xgb_model.pkl
        xgb_model_raw.pkl
    backend/
        benchmark_latency.py
        chat_analyst.py
        evaluate.py
        feature_glossary.py
        fraud_trends.py
        load_data.py
        responder.py
        train_model.py
    frontend/
        app.py
```
