"""
frontend/app.py — Streamlit Frontend for Fraud Detector
========================================================
Two-tab Streamlit application:
  Tab 1: Transaction Analyzer (manual entry, pre-loaded demo, CSV upload)
  Tab 2: Model Evaluation (metrics from saved backend artifacts)

Usage:
    streamlit run frontend/app.py
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import streamlit as st

# Add project root to path for backend imports
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")

from dotenv import load_dotenv
load_dotenv()

from backend.responder import (
    load_model_artifacts,
    predict_fraud_probability,
    predict_fraud_probability_batch,
    generate_case_note,
    generate_risk_narration,
    prepare_feature_vector,
)
from backend.feature_glossary import lookup_feature

# ---------------------------------------------------------------------------
# Page config & theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Fraud Detector & Risk Case-Note Responder",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ---------------------------------------------------------------------------
# Custom CSS — dark theme + gradient banner + styling
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global dark theme */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Top bar styling */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.5rem 0;
        margin-bottom: 0.5rem;
    }
    .top-bar h1 {
        color: #ffffff;
        font-size: 1.6rem;
        margin: 0;
        font-weight: 700;
    }
    .deploy-btn {
        background: linear-gradient(135deg, #4f46e5, #7c3aed);
        color: white;
        border: none;
        padding: 0.45rem 1.2rem;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
        cursor: default;
        letter-spacing: 0.02em;
    }
    
    /* Gradient banner */
    .gradient-banner {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #a855f7 100%);
        padding: 1.2rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 20px rgba(124, 58, 237, 0.3);
    }
    .gradient-banner p {
        color: #f0f0f0;
        font-size: 1.05rem;
        margin: 0;
        font-weight: 400;
        letter-spacing: 0.01em;
    }
    
    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #1e1e2f 0%, #252540 100%);
        border: 1px solid #3a3a5c;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        box-shadow: 0 2px 12px rgba(0,0,0,0.3);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(124, 58, 237, 0.2);
    }
    .metric-card .label {
        color: #a0a0b8;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.4rem;
    }
    .metric-card .value {
        color: #ffffff;
        font-size: 1.8rem;
        font-weight: 700;
        line-height: 1.2;
    }
    
    /* Risk badges */
    .risk-badge {
        display: inline-block;
        padding: 0.4rem 1.2rem;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.95rem;
        letter-spacing: 0.03em;
    }
    .risk-low {
        background: linear-gradient(135deg, #065f46, #047857);
        color: #a7f3d0;
        border: 1px solid #059669;
    }
    .risk-elevated {
        background: linear-gradient(135deg, #1e40af, #2563eb);
        color: #bfdbfe;
        border: 1px solid #3b82f6;
    }
    .risk-high {
        background: linear-gradient(135deg, #78350f, #92400e);
        color: #fcd34d;
        border: 1px solid #d97706;
    }
    
    /* Action pill badge */
    .action-pill {
        display: inline-block;
        padding: 0.35rem 1rem;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    .action-review {
        background: #1e40af;
        color: #bfdbfe;
        border: 1px solid #3b82f6;
    }
    .action-stepup {
        background: #78350f;
        color: #fcd34d;
        border: 1px solid #d97706;
    }
    .action-decline {
        background: #7f1d1d;
        color: #fca5a5;
        border: 1px solid #ef4444;
    }
    
    /* Case note card */
    .case-note-card {
        background: #1a1a2e;
        border: 1px solid #4f46e5;
        border-left: 4px solid #7c3aed;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        box-shadow: 0 2px 12px rgba(79, 70, 229, 0.15);
    }
    .case-note-card h4 {
        color: #a78bfa;
        margin-top: 0;
        margin-bottom: 0.6rem;
        font-size: 0.95rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .case-note-card p {
        color: #d1d5db;
        font-size: 0.95rem;
        line-height: 1.6;
    }
    
    /* Score display */
    .score-display {
        text-align: center;
        padding: 1.5rem;
        background: linear-gradient(135deg, #1a1a2e, #252540);
        border-radius: 12px;
        border: 1px solid #3a3a5c;
        margin: 1rem 0;
    }
    .score-display .score-value {
        font-size: 3rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .score-display .score-label {
        font-size: 0.85rem;
        color: #a0a0b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 0.3rem;
    }
    
    /* Contributions table */
    .contrib-table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 0.8rem;
    }
    .contrib-table th {
        background: #252540;
        color: #a0a0b8;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.5rem 0.8rem;
        text-align: left;
        border-bottom: 1px solid #3a3a5c;
    }
    .contrib-table td {
        padding: 0.45rem 0.8rem;
        font-size: 0.88rem;
        color: #d1d5db;
        border-bottom: 1px solid #2a2a3e;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background-color: #1a1a2e;
        border-radius: 10px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #a0a0b8;
        font-weight: 600;
        font-size: 0.95rem;
        padding: 0.6rem 1.5rem;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
        color: white !important;
    }
    
    /* Section headers */
    .section-header {
        color: #a78bfa;
        font-size: 1.1rem;
        font-weight: 600;
        margin: 1.5rem 0 0.8rem 0;
        padding-bottom: 0.4rem;
        border-bottom: 2px solid #4f46e5;
    }
    
    /* No-action card */
    .no-action-card {
        background: linear-gradient(135deg, #064e3b, #065f46);
        border: 1px solid #059669;
        border-radius: 10px;
        padding: 1.2rem 1.5rem;
        margin-top: 1rem;
        text-align: center;
    }
    .no-action-card p {
        color: #a7f3d0;
        font-size: 1.05rem;
        font-weight: 600;
        margin: 0;
    }
    
    /* AI narration line */
    .narration-line {
        background: linear-gradient(135deg, #1a1a2e, #1e1e3a);
        border: 1px solid #3a3a5c;
        border-left: 3px solid #a78bfa;
        border-radius: 8px;
        padding: 0.8rem 1.2rem;
        margin: 1rem 0;
        font-size: 0.92rem;
        color: #c4b5fd;
        line-height: 1.5;
        font-style: italic;
    }
    .narration-line .narration-icon {
        font-style: normal;
        margin-right: 0.4rem;
    }

    /* Hide Streamlit's default hamburger menu and footer */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Plot container */
    .plot-container {
        background: #1a1a2e;
        border: 1px solid #2a2a4a;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load artifacts (cached via st.cache_resource)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load all backend artifacts once and cache them."""
    model, raw_model, encoder, feature_names, config, eval_summary = load_model_artifacts()
    
    # Load demo cases
    demo_path = os.path.join(ARTIFACTS_DIR, "demo_cases.json")
    with open(demo_path) as f:
        demo_cases = json.load(f)
    
    return model, raw_model, encoder, feature_names, config, eval_summary, demo_cases


def load_eval_summary():
    """Load evaluation summary JSON."""
    with open(os.path.join(ARTIFACTS_DIR, "eval_summary.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div class="top-bar">
    <h1>🛡️ Fraud Detector & Risk Case-Note Responder</h1>
    <span class="deploy-btn">Deploy</span>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="gradient-banner">
    <p>AI-powered fraud risk scoring with defense-only case-note responder.</p>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Main Application Content
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["🔍 Transaction Analyzer", "📊 Model Evaluation", "📈 Fraud Trends"])


# ===========================================================================
# TAB 1: Transaction Analyzer
# ===========================================================================
with tab1:
    model, raw_model, encoder, feature_names, config, eval_summary, demo_cases = load_artifacts()
    optimal_threshold = eval_summary["optimal_threshold"]
    
    # Input method selector
    input_method = st.radio(
        "Input Method",
        ["Manual entry", "Pre-loaded demo", "Upload CSV row"],
        horizontal=True,
        key="input_method",
    )
    
    st.markdown("---")
    
    if "current_feature_dict" not in st.session_state:
        st.session_state.current_feature_dict = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "tx_context" not in st.session_state:
        st.session_state.tx_context = None

    feature_dict = st.session_state.current_feature_dict
    
    # ----- Manual Entry -----
    if input_method == "Manual entry":
        st.markdown('<p class="section-header">Transaction Details</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            transaction_amt = st.number_input(
                "Transaction Amount ($)", min_value=0.01, max_value=100000.0,
                value=100.0, step=10.0, key="manual_amt"
            )
            
            product_cd = st.selectbox(
                "Product Category",
                options=["W", "H", "C", "S", "R"],
                format_func=lambda x: {
                    "W": "W — Digital goods",
                    "H": "H — Physical goods",
                    "C": "C — Consumables",
                    "S": "S — Services",
                    "R": "R — Recurring",
                }.get(x, x),
                key="manual_product"
            )
            
            days_since_activity = st.number_input(
                "Days Since Account Activity",
                min_value=0, max_value=1000, value=10, step=1,
                help="Proxy for D1 column — days since last account activity",
                key="manual_d1"
            )
        
        with col2:
            st.markdown("**Transaction Signals**")
            
            addr_match = st.checkbox("Billing/Shipping Address Match", value=True, key="manual_addr")
            email_recognized = st.checkbox("Email Domain Recognized", value=True, key="manual_email")
            distance_present = st.checkbox("Distance Flag (dist1/dist2 present)", value=False, key="manual_dist")
            card_match = st.checkbox("Card Match Flags (M-columns)", value=True, key="manual_card_match")
        
        with col3:
            card_type = st.selectbox(
                "Card Type",
                options=["visa", "mastercard", "american express", "discover"],
                key="manual_card4"
            )
            card_class = st.selectbox(
                "Card Class",
                options=["credit", "debit", "debit or credit", "charge card"],
                key="manual_card6"
            )
            customer_history = st.number_input(
                "Customer History Score",
                min_value=0, max_value=20000, value=1000, step=100,
                help="Proxy for card1 — customer frequency identifier",
                key="manual_card1"
            )
        
        if st.button("🛡️ Check Risk Before Payment", key="manual_analyze", use_container_width=True):
            # Map UI inputs to model features, using train medians for unexposed columns
            medians = config["train_medians"]
            feature_dict = {col: medians[col] for col in config["numeric_cols"]}
            feature_dict.update({col: "missing" for col in config["categorical_cols"]})
            
            # Override with user inputs
            feature_dict["TransactionAmt"] = transaction_amt
            feature_dict["ProductCD"] = product_cd
            feature_dict["D1"] = float(days_since_activity)
            feature_dict["card1"] = float(customer_history)
            feature_dict["card4"] = card_type
            feature_dict["card6"] = card_class
            
            # Map boolean signals
            feature_dict["M4"] = "T" if addr_match else "F"
            feature_dict["M5"] = "T" if email_recognized else "F"
            feature_dict["M6"] = "T" if card_match else "F"
            
            if email_recognized:
                feature_dict["P_emaildomain"] = "gmail.com"
            else:
                feature_dict["P_emaildomain"] = "missing"
            
            if distance_present:
                feature_dict["dist1"] = 50.0
            else:
                feature_dict["dist1"] = medians.get("dist1", 0)
                
            st.session_state.current_feature_dict = feature_dict
            st.session_state.chat_history = []
            st.session_state.tx_context = None
    
    # ----- Pre-loaded Demo -----
    elif input_method == "Pre-loaded demo":
        demo_labels = [d["label"] for d in demo_cases]
        selected_demo = st.selectbox(
            "Select a demo case", options=demo_labels, key="demo_select"
        )
        
        selected_idx = demo_labels.index(selected_demo)
        demo = demo_cases[selected_idx]
        
        # Show brief info about the demo case
        actual = "🔴 Fraudulent" if demo["actual_fraud"] else "🟢 Legitimate"
        st.info(f"**{selected_demo}** — Ground truth: {actual}")
        
        if st.button("🛡️ Check Risk Before Payment", key="demo_analyze", use_container_width=True):
            feature_dict = demo["features"]
            st.session_state.current_feature_dict = feature_dict
            st.session_state.chat_history = []
            st.session_state.tx_context = None
    
    # ----- Upload CSV -----
    elif input_method == "Upload CSV row":
        uploaded_file = st.file_uploader(
            "Upload a CSV file with one or more transaction rows",
            type=["csv"],
            key="csv_upload"
        )
        
        if uploaded_file is not None:
            try:
                df_upload = pd.read_csv(uploaded_file)
                if len(df_upload) == 0:
                    st.error("The uploaded CSV is empty.")
                else:
                    # Validate columns
                    required_cols = set(config["all_feature_cols"])
                    uploaded_cols = set(df_upload.columns)
                    missing_cols = required_cols - uploaded_cols
                    
                    if missing_cols:
                        st.error(f"Missing required columns: {', '.join(sorted(missing_cols))}")
                        st.info(f"Expected columns: {', '.join(sorted(required_cols))}")
                    else:
                        st.success(f"CSV validated — {len(df_upload)} transaction(s) found.")
                        
                        if st.button("🛡️ Score All Transactions", key="csv_analyze", use_container_width=True):
                            with st.spinner(f"Scoring {len(df_upload)} transactions..."):
                                batch_results = predict_fraud_probability_batch(
                                    df_upload, model, raw_model, encoder, config,
                                    feature_names, optimal_threshold,
                                )
                            st.session_state["csv_batch_results"] = batch_results
                            st.session_state["csv_batch_df"] = df_upload
                            # Clear per-row LLM caches from a previous upload
                            st.session_state.pop("csv_narration_cache", None)
                            st.session_state.pop("csv_casenote_cache", None)
                            st.session_state.current_feature_dict = None
                            st.session_state.chat_history = []
                            st.session_state.tx_context = None
            except Exception as e:
                st.error(f"Error reading CSV: {str(e)}")
        
        # --- CSV Batch Results Panel ---
        if "csv_batch_results" in st.session_state and input_method == "Upload CSV row":
            batch_results = st.session_state["csv_batch_results"]
            st.markdown("---")
            
            n_total = len(batch_results)
            n_flagged = sum(1 for r in batch_results if r["risk_tier"] != "Approved")
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Transactions Scored</div>
                    <div class="value">{n_total}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_s2:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Flagged for Review</div>
                    <div class="value" style="color: #f59e0b;">{n_flagged}</div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            # Build summary table
            summary_rows = []
            for r in batch_results:
                fd = r["feature_dict"]
                summary_rows.append({
                    "Row": r["row_idx"] + 1,
                    "Transaction Amt": fd.get("TransactionAmt", 0),
                    "Product": fd.get("ProductCD", "—"),
                    "Fraud Probability": round(r["fraud_prob"], 4),
                    "Decision": r["risk_tier"],
                })
            summary_df = pd.DataFrame(summary_rows).sort_values(
                "Fraud Probability", ascending=False
            )
            
            def _color_decision(val):
                colors = {
                    "Approved": "color: #22c55e",
                    "Flagged for Review": "color: #f59e0b; font-weight: 600",
                    "Strongly Flagged": "color: #ef4444; font-weight: 600",
                }
                return colors.get(val, "")
            
            styled_summary = summary_df.style.applymap(
                _color_decision, subset=["Decision"]
            )
            st.dataframe(styled_summary, use_container_width=True, hide_index=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            st.info(
                f"{n_total} transactions loaded and scored. "
                f"Select a row below to see the full AI explanation."
            )
            
            # Row selector for detail view
            row_options = [
                f"Row {r['row_idx']+1} — {r['fraud_prob']:.2%} — {r['risk_tier']}"
                for r in sorted(batch_results, key=lambda x: -x["fraud_prob"])
            ]
            sorted_results = sorted(batch_results, key=lambda x: -x["fraud_prob"])
            
            selected_row_label = st.selectbox(
                "View details for row:",
                options=row_options,
                key="csv_row_select",
            )
            selected_idx = row_options.index(selected_row_label)
            selected_result = sorted_results[selected_idx]
            sel_row_idx = selected_result["row_idx"]
            sel_feature_dict = selected_result["feature_dict"]
            sel_prob = selected_result["fraud_prob"]
            sel_tier = selected_result["risk_tier"]
            sel_class = selected_result["risk_class"]
            sel_color = selected_result["score_color"]
            
            # Badge caption
            if sel_prob < optimal_threshold:
                sel_caption = (
                    f"Below the cost-optimal review threshold for this system "
                    f"({optimal_threshold:.2%})."
                )
            else:
                sel_caption = (
                    f"Flagged because this exceeds the cost-optimal review threshold "
                    f"for this system ({optimal_threshold:.2%}) \u2014 not a claim that "
                    f"fraud is likely."
                )
            
            # Score display for selected row
            st.markdown(f"""
            <div class="score-display">
                <div class="score-value" style="color: {sel_color};">{sel_prob:.1%}</div>
                <div class="score-label">Fraud Probability — Row {sel_row_idx+1}</div>
                <div style="margin-top: 0.8rem;">
                    <span class="risk-badge {sel_class}">{sel_tier}</span>
                </div>
                <div style="margin-top: 0.5rem; color: #9ca3af; font-size: 0.8rem; max-width: 500px; margin-left: auto; margin-right: auto;">
                    {sel_caption}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- On-demand LLM narration (cached per row) ---
            if "csv_narration_cache" not in st.session_state:
                st.session_state["csv_narration_cache"] = {}
            narration_cache = st.session_state["csv_narration_cache"]
            
            # Compute feature contributions for selected row (needed for narration + display)
            from backend.responder import get_feature_contributions
            X_sel = prepare_feature_vector(sel_feature_dict, config, encoder)
            contributions = get_feature_contributions(raw_model, X_sel, feature_names)
            
            if sel_row_idx not in narration_cache:
                try:
                    narration = generate_risk_narration(
                        contributions, float(sel_prob), float(optimal_threshold), sel_tier
                    )
                except Exception as e:
                    narration = None
                    print(f"[app] Narration generation failed for row {sel_row_idx+1}: {e}")
                narration_cache[sel_row_idx] = narration
            else:
                narration = narration_cache[sel_row_idx]
            
            if narration:
                st.markdown(f"""
                <div class="narration-line">
                    <span class="narration-icon">🤖</span> {narration}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.caption("🤖 AI narration unavailable for this transaction.")
            
            # Top contributing features
            st.markdown('<p class="section-header">Top Contributing Signals</p>', unsafe_allow_html=True)
            import html as html_mod
            contrib_html = '<table class="contrib-table"><tr><th>Feature</th><th>Value</th><th>Impact</th><th>Description</th></tr>'
            for c in contributions:
                direction_icon = "🔴 ↑" if c["contribution"] > 0 else "🟢 ↓"
                desc = lookup_feature(c["feature"]) or "—"
                desc_escaped = html_mod.escape(desc)
                contrib_html += f'<tr><td>{c["feature"]}</td><td>{c["value"]:.2f}</td><td>{direction_icon} {c["direction"]}</td><td style="font-size:0.82rem;color:#a0a0b8;max-width:300px;">{desc_escaped}</td></tr>'
            contrib_html += '</table>'
            st.markdown(contrib_html, unsafe_allow_html=True)
            
            # Case note (only for flagged transactions, cached per row)
            if sel_prob >= optimal_threshold:
                if "csv_casenote_cache" not in st.session_state:
                    st.session_state["csv_casenote_cache"] = {}
                casenote_cache = st.session_state["csv_casenote_cache"]
                
                if sel_row_idx not in casenote_cache:
                    st.markdown('<p class="section-header">AI Case Note</p>', unsafe_allow_html=True)
                    try:
                        cn_result = generate_case_note(
                            sel_feature_dict,
                            fraud_prob=sel_prob,
                            optimal_threshold=optimal_threshold,
                            model=model,
                            raw_model=raw_model,
                            encoder=encoder,
                            config=config,
                            feature_names=feature_names,
                            eval_summary=eval_summary,
                        )
                    except Exception as e:
                        cn_result = {"case_note": f"[Case note generation failed: {e}]",
                                     "recommended_action": "Hold — pending manual review"}
                        print(f"[app] Case note failed for row {sel_row_idx+1}: {e}")
                    casenote_cache[sel_row_idx] = cn_result
                else:
                    st.markdown('<p class="section-header">AI Case Note</p>', unsafe_allow_html=True)
                    cn_result = casenote_cache[sel_row_idx]
                
                action = cn_result.get("recommended_action", "Hold — do not capture payment yet, pending manual review")
                if "decline" in action.lower():
                    action_class = "action-decline"
                elif "step-up" in action.lower():
                    action_class = "action-stepup"
                else:
                    action_class = "action-review"
                
                st.markdown(f"""
                <div class="case-note-card">
                    <h4>🤖 Generated Case Note</h4>
                    <p>{cn_result.get('case_note', 'Case note unavailable.')}</p>
                    <div style="margin-top: 0.8rem;">
                        <span style="color: #6b7280; font-size: 0.8rem; margin-right: 0.5rem;">Recommended Action:</span>
                        <span class="action-pill {action_class}">{action}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="no-action-card">
                    <p>✅ Approved — proceed with payment.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Update transaction context for the chatbot
            action_for_chat = "Approved"
            if sel_prob >= optimal_threshold:
                action_for_chat = cn_result.get("recommended_action", "Hold") if sel_prob >= optimal_threshold else "Approved"
            
            st.session_state.tx_context = {
                "probability": float(sel_prob),
                "decided_action": action_for_chat,
                "top_features": "; ".join(f"{c['feature']}={c['value']} ({c['direction']})" for c in contributions[:5])
            }
    
    # ----- Single-Row Result Panel (Manual entry / Pre-loaded demo) -----
    if feature_dict is not None and input_method != "Upload CSV row":
        st.markdown("---")
        
        with st.spinner("Analyzing transaction..."):
            # Get prediction
            fraud_prob, contributions, X = predict_fraud_probability(
                feature_dict, model, raw_model, encoder, config, feature_names
            )
            
            # Determine risk level and decision-oriented badge label
            above_range = 1.0 - optimal_threshold
            decline_floor = optimal_threshold + above_range * 0.7
            if fraud_prob < optimal_threshold * 0.5:
                risk_level = "low"
                risk_label = "Approved"
                risk_class = "risk-low"
                score_color = "#22c55e"
            elif fraud_prob < optimal_threshold:
                risk_level = "elevated"
                risk_label = "Approved"
                risk_class = "risk-low"
                score_color = "#22c55e"
            elif fraud_prob < decline_floor:
                risk_level = "high"
                risk_label = "Flagged for Review"
                risk_class = "risk-elevated"
                score_color = "#f59e0b"
            else:
                risk_level = "high"
                risk_label = "Strongly Flagged"
                risk_class = "risk-high"
                score_color = "#ef4444"
            
            # Build caption explaining the badge decision
            if fraud_prob < optimal_threshold:
                badge_caption = (
                    f"Below the cost-optimal review threshold for this system "
                    f"({optimal_threshold:.2%})."
                )
            else:
                badge_caption = (
                    f"Flagged because this exceeds the cost-optimal review threshold "
                    f"for this system ({optimal_threshold:.2%}) \u2014 not a claim that "
                    f"fraud is likely."
                )
            
            # Score display
            st.markdown(f"""
            <div class="score-display">
                <div class="score-value" style="color: {score_color};">{fraud_prob:.1%}</div>
                <div class="score-label">Fraud Probability</div>
                <div style="margin-top: 0.8rem;">
                    <span class="risk-badge {risk_class}">{risk_label}</span>
                </div>
                <div style="margin-top: 0.5rem; color: #9ca3af; font-size: 0.8rem; max-width: 500px; margin-left: auto; margin-right: auto;">
                    {badge_caption}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Always-on AI narration (runs for every transaction, flagged or not)
            try:
                narration = generate_risk_narration(
                    contributions, float(fraud_prob), float(optimal_threshold), risk_label
                )
            except Exception as e:
                narration = None
                print(f"[app] Narration generation failed: {e}")

            if narration:
                st.markdown(f"""
                <div class="narration-line">
                    <span class="narration-icon">🤖</span> {narration}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.caption("🤖 AI narration unavailable for this transaction.")
            
            # Top contributing features
            st.markdown('<p class="section-header">Top Contributing Signals</p>', unsafe_allow_html=True)
            
            contrib_html = '<table class="contrib-table"><tr><th>Feature</th><th>Value</th><th>Impact</th><th>Description</th></tr>'
            for c in contributions:
                direction_icon = "🔴 ↑" if c["contribution"] > 0 else "🟢 ↓"
                desc = lookup_feature(c["feature"]) or "—"
                # Escape HTML in description
                import html as html_mod
                desc_escaped = html_mod.escape(desc)
                contrib_html += f'<tr><td>{c["feature"]}</td><td>{c["value"]:.2f}</td><td>{direction_icon} {c["direction"]}</td><td style="font-size:0.82rem;color:#a0a0b8;max-width:300px;">{desc_escaped}</td></tr>'
            contrib_html += '</table>'
            st.markdown(contrib_html, unsafe_allow_html=True)
            
            # Case note (only for flagged transactions)
            if fraud_prob >= optimal_threshold:
                st.markdown('<p class="section-header">AI Case Note</p>', unsafe_allow_html=True)
                
                result = generate_case_note(
                    feature_dict,
                    fraud_prob=fraud_prob,
                    optimal_threshold=optimal_threshold,
                    model=model,
                    raw_model=raw_model,
                    encoder=encoder,
                    config=config,
                    feature_names=feature_names,
                    eval_summary=eval_summary,
                )
                
                # Action badge
                action = result.get("recommended_action", "Hold — do not capture payment yet, pending manual review")
                if "decline" in action.lower():
                    action_class = "action-decline"
                elif "step-up" in action.lower():
                    action_class = "action-stepup"
                else:
                    action_class = "action-review"
                
                st.markdown(f"""
                <div class="case-note-card">
                    <h4>🤖 Generated Case Note</h4>
                    <p>{result.get('case_note', 'Case note unavailable.')}</p>
                    <div style="margin-top: 0.8rem;">
                        <span style="color: #6b7280; font-size: 0.8rem; margin-right: 0.5rem;">Recommended Action:</span>
                        <span class="action-pill {action_class}">{action}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="no-action-card">
                    <p>✅ Approved — proceed with payment.</p>
                </div>
                """, unsafe_allow_html=True)
                
            # Update transaction context in session state for the chatbot
            action_for_chat = "Approved"
            if fraud_prob >= optimal_threshold:
                action_for_chat = result.get("recommended_action", "Hold")
                
            st.session_state.tx_context = {
                "probability": float(fraud_prob),
                "decided_action": action_for_chat,
                "top_features": "; ".join(f"{c['feature']}={c['value']} ({c['direction']})" for c in contributions[:5])
            }


    # ----- Risk Analyst Chatbot -----
    st.markdown('<p class="section-header">Ask the Risk Analyst</p>', unsafe_allow_html=True)
    
    # Build model_metrics from already-loaded eval_summary
    model_metrics = {
        "pr_auc": eval_summary.get("pr_auc_xgb"),
        "base_rate": eval_summary.get("fraud_rate"),
        "optimal_threshold": eval_summary.get("optimal_threshold"),
        "precision": eval_summary.get("precision_at_optimal"),
        "recall": eval_summary.get("recall_at_optimal"),
    }
    
    if st.button("Clear conversation", key="clear_chat"):
        st.session_state.chat_history = []
        st.rerun()
        
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    if user_q := st.chat_input("Ask about this transaction, the model's metrics, or fraud detection in general."):
        st.session_state.chat_history.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
            
        with st.chat_message("assistant"):
            with st.spinner("Analyzing context..."):
                try:
                    from backend.chat_analyst import chat_with_analyst
                    reply = chat_with_analyst(
                        user_q,
                        st.session_state.tx_context,
                        model_metrics,
                        st.session_state.chat_history[:-1],
                    )
                    st.markdown(reply)
                    st.session_state.chat_history.append({"role": "assistant", "content": reply})
                except Exception as e:
                    st.error(f"Error calling analyst chatbot: {str(e)}")


# ===========================================================================
# TAB 2: Model Evaluation
# ===========================================================================
with tab2:
    eval_data = load_eval_summary()
    
    # Headline metric cards
    st.markdown('<p class="section-header">Model Performance Summary</p>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">PR-AUC (XGBoost)</div>
            <div class="value">{eval_data['pr_auc_xgb']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Precision @ Optimal</div>
            <div class="value">{eval_data['precision_at_optimal']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Recall @ Optimal</div>
            <div class="value">{eval_data['recall_at_optimal']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        savings = eval_data.get('cost_savings', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Cost Savings (Optimal vs. Naive)</div>
            <div class="value">${savings/1e6:.2f}M</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Additional metrics row
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">PR-AUC (Logistic Reg.)</div>
            <div class="value">{eval_data['pr_auc_lr']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Optimal Threshold</div>
            <div class="value">{eval_data['optimal_threshold']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col7:
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Test Set Fraud Rate</div>
            <div class="value">{eval_data['fraud_rate']:.2%}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col8:
        savings_pct = eval_data.get('cost_savings_pct', 0)
        st.markdown(f"""
        <div class="metric-card">
            <div class="label">Cost Reduction %</div>
            <div class="value">{savings_pct:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Calibration (ECE) metric cards row
    calibration = eval_data.get('calibration', {})
    xgb_cal = calibration.get('XGBoost', {})
    lr_cal = calibration.get('Logistic Regression', {})
    ece_xgb = xgb_cal.get('ece')
    ece_lr = lr_cal.get('ece')
    
    if ece_xgb is not None:
        # Load latency benchmark if available
        latency_data = None
        latency_path = os.path.join(ARTIFACTS_DIR, "latency_benchmark.json")
        if os.path.exists(latency_path):
            with open(latency_path) as f:
                latency_data = json.load(f)
        
        col9, col10, col11, col12 = st.columns(4)
        
        with col9:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">ECE (XGBoost)</div>
                <div class="value">{ece_xgb:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col10:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">ECE (Logistic Reg.)</div>
                <div class="value">{ece_lr:.4f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col11:
            st.markdown("""
            <div class="metric-card" style="text-align:left;">
                <div class="label">What is ECE?</div>
                <div style="color:#a0a0b8; font-size:0.85rem; line-height:1.5;">
                    <b>Expected Calibration Error</b> — the average gap between what the model
                    predicts and what actually happens. Lower is better. An ECE near zero means
                    "when the model says 30% fraud, roughly 30% really are fraud," which makes
                    the cost-threshold math trustworthy.
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col12:
            if latency_data:
                p95 = latency_data["p95_ms"]
                p50 = latency_data["p50_ms"]
                hw = latency_data.get("hardware_note", "unknown hardware")
                st.markdown(f"""
                <div class="metric-card">
                    <div class="label">Inference Latency (p95)</div>
                    <div class="value">{p95:.1f} ms</div>
                    <div style="color:#a0a0b8; font-size:0.75rem; margin-top:0.4rem;">
                        p50: {p50:.1f} ms &middot; single-row
                    </div>
                    <div style="color:#6b7280; font-size:0.65rem; margin-top:0.2rem; line-height:1.3;">
                        {hw}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="metric-card">
                    <div class="label">Inference Latency (p95)</div>
                    <div class="value" style="font-size:1rem; color:#6b7280;">Not measured</div>
                    <div style="color:#a0a0b8; font-size:0.75rem; margin-top:0.4rem;">
                        Run <code>python backend/benchmark_latency.py</code>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Evaluation plots — loaded from saved images
    plot_sections = [
        ("Precision-Recall Curve", "pr_curve.png",
         "PR curve for XGBoost and Logistic Regression on the test set, with random baseline at the fraud rate."),
        ("Cost-Sensitive Threshold Analysis", "cost_threshold.png",
         "Total expected cost vs. decision threshold. The optimal threshold minimizes combined FN costs (lost transaction + chargeback) and FP costs (review labor)."),
        ("Confusion Matrix (@ 0.5 Threshold)", "confusion_matrix.png",
         "Standard confusion matrix at the default 0.5 threshold, showing counts and percentages."),
        ("Segmented Model Performance", "segmented_metrics.png",
         "PR-AUC and Recall broken down by transaction amount bucket and product category."),
        ("Feature Importance — Leakage Check", "feature_importance.png",
         "Top-20 features by XGBoost importance and Logistic Regression coefficient magnitude. Checks for single-feature dominance that might indicate data leakage."),
        ("Calibration (Reliability) Curve", "calibration_curve.png",
         "Reliability diagram: for each probability bin, the observed fraud rate is plotted against the mean predicted probability. "
         "A well-calibrated model tracks the diagonal — this is what makes the cost-threshold math trustworthy. "
         "ECE (Expected Calibration Error) is the sample-weighted mean gap between the two."),
    ]
    
    for title, filename, description in plot_sections:
        st.markdown(f'<p class="section-header">{title}</p>', unsafe_allow_html=True)
        st.caption(description)
        
        img_path = os.path.join(ARTIFACTS_DIR, filename)
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
        else:
            st.warning(f"Plot not found: {filename}. Run `python backend/evaluate.py` to generate.")
        
        st.markdown("<br>", unsafe_allow_html=True)


# ===========================================================================
# TAB 3: Fraud Trends
# ===========================================================================
with tab3:
    trends_path = os.path.join(ARTIFACTS_DIR, "fraud_trends.json")

    if not os.path.exists(trends_path):
        st.warning(
            "Fraud trends data not found. "
            "Run `python backend/fraud_trends.py` to generate."
        )
    else:
        with open(trends_path) as f:
            trends_data = json.load(f)

        st.markdown(
            '<p class="section-header">Aggregate Fraud Rate Over Time</p>',
            unsafe_allow_html=True,
        )

        st.caption(
            "Computed on the held-out test set's real transaction timestamps "
            "and labels \u2014 this view looks across transactions over time, "
            "complementing the per-transaction detector in the Transaction "
            "Analyzer tab, which scores each payment individually at checkout."
        )

        # Summary cards
        n_bins = trends_data["n_bins"]
        n_spikes = trends_data["n_spikes_overall"]

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Time Bins</div>
                <div class="value">{n_bins}</div>
            </div>
            """, unsafe_allow_html=True)
        with sc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Spikes Detected</div>
                <div class="value" style="color: {'#ef4444' if n_spikes > 0 else '#22c55e'};">{n_spikes}</div>
            </div>
            """, unsafe_allow_html=True)
        with sc3:
            # Average fraud rate across bins
            overall_records = trends_data["overall"]
            avg_fr = np.mean([r["fraud_rate"] for r in overall_records if r["fraud_rate"] is not None])
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Avg Fraud Rate</div>
                <div class="value">{avg_fr:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # View toggle
        view_mode = st.radio(
            "View",
            ["Overall", "Per ProductCD"],
            horizontal=True,
            key="trends_view_mode",
        )

        # ---------------------------------------------------------------
        # Build chart data
        # ---------------------------------------------------------------
        import plotly.graph_objects as go

        if view_mode == "Overall":
            records = trends_data["overall"]
            df_trend = pd.DataFrame(records)

            fig = go.Figure()

            # Baseline band (mean +/- 1 std)
            has_baseline = df_trend["baseline_mean"].notna()
            df_bl = df_trend[has_baseline].copy()
            if len(df_bl) > 0:
                upper = df_bl["baseline_mean"] + df_bl["baseline_std"].fillna(0)
                lower = (df_bl["baseline_mean"] - df_bl["baseline_std"].fillna(0)).clip(lower=0)

                fig.add_trace(go.Scatter(
                    x=df_bl["hour_bin"], y=upper,
                    mode="lines", line=dict(width=0),
                    showlegend=False, hoverinfo="skip",
                ))
                fig.add_trace(go.Scatter(
                    x=df_bl["hour_bin"], y=lower,
                    mode="lines", line=dict(width=0),
                    fill="tonexty",
                    fillcolor="rgba(124, 58, 237, 0.12)",
                    name="Baseline \u00b11\u03c3",
                    hoverinfo="skip",
                ))
                # Baseline mean line
                fig.add_trace(go.Scatter(
                    x=df_bl["hour_bin"], y=df_bl["baseline_mean"],
                    mode="lines",
                    line=dict(color="#7c3aed", width=1.5, dash="dash"),
                    name="Rolling baseline",
                    hovertemplate="Bin %{x}<br>Baseline: %{y:.3%}<extra></extra>",
                ))

            # Main fraud rate line
            fig.add_trace(go.Scatter(
                x=df_trend["hour_bin"], y=df_trend["fraud_rate"],
                mode="lines",
                line=dict(color="#06b6d4", width=2),
                name="Fraud rate",
                hovertemplate=(
                    "Bin %{x}<br>"
                    "Fraud rate: %{y:.3%}<br>"
                    "Transactions: %{customdata[0]}<br>"
                    "Fraud count: %{customdata[1]}<extra></extra>"
                ),
                customdata=df_trend[["n_transactions", "n_fraud"]].values,
            ))

            # Spike markers
            df_spikes = df_trend[df_trend["is_spike"] == True]  # noqa: E712
            if len(df_spikes) > 0:
                fig.add_trace(go.Scatter(
                    x=df_spikes["hour_bin"], y=df_spikes["fraud_rate"],
                    mode="markers",
                    marker=dict(color="#ef4444", size=10, symbol="diamond",
                                line=dict(color="#ffffff", width=1.5)),
                    name="Spike",
                    hovertemplate=(
                        "SPIKE \u2014 Bin %{x}<br>"
                        "Fraud rate: %{y:.3%}<br>"
                        "z-score: %{customdata[0]:.2f}<br>"
                        "Baseline: %{customdata[1]:.3%}<extra></extra>"
                    ),
                    customdata=df_spikes[["z_score", "baseline_mean"]].values,
                ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#16213e",
                title=dict(text="Fraud Rate by Time Bin (Overall)", font=dict(size=16)),
                xaxis_title="Time Bin (4-hour periods)",
                yaxis_title="Fraud Rate",
                yaxis_tickformat=".1%",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=450,
                margin=dict(l=60, r=30, t=60, b=50),
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

        else:
            # Per-ProductCD view
            records = trends_data["per_product"]
            df_pp = pd.DataFrame(records)
            products = sorted(df_pp["ProductCD"].unique())

            fig = go.Figure()

            colors = {
                "W": "#06b6d4", "H": "#f59e0b", "C": "#22c55e",
                "S": "#a78bfa", "R": "#ef4444",
            }

            for prod in products:
                df_prod = df_pp[df_pp["ProductCD"] == prod].sort_values("hour_bin")
                color = colors.get(prod, "#888888")

                fig.add_trace(go.Scatter(
                    x=df_prod["hour_bin"], y=df_prod["fraud_rate"],
                    mode="lines",
                    line=dict(color=color, width=1.5),
                    name=f"Product {prod}",
                    hovertemplate=(
                        f"Product {prod}<br>"
                        "Bin %{x}<br>"
                        "Fraud rate: %{y:.3%}<br>"
                        "Transactions: %{customdata[0]}<extra></extra>"
                    ),
                    customdata=df_prod[["n_transactions"]].values,
                ))

                # Spike markers for this product
                df_prod_spikes = df_prod[df_prod["is_spike"] == True]  # noqa: E712
                if len(df_prod_spikes) > 0:
                    fig.add_trace(go.Scatter(
                        x=df_prod_spikes["hour_bin"],
                        y=df_prod_spikes["fraud_rate"],
                        mode="markers",
                        marker=dict(color=color, size=8, symbol="diamond",
                                    line=dict(color="#ffffff", width=1.5)),
                        name=f"Spike ({prod})",
                        showlegend=False,
                        hovertemplate=(
                            f"SPIKE \u2014 Product {prod}<br>"
                            "Bin %{x}<br>"
                            "Fraud rate: %{y:.3%}<extra></extra>"
                        ),
                    ))

            fig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0e1117",
                plot_bgcolor="#16213e",
                title=dict(text="Fraud Rate by Time Bin (Per Product Category)", font=dict(size=16)),
                xaxis_title="Time Bin (4-hour periods)",
                yaxis_title="Fraud Rate",
                yaxis_tickformat=".1%",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=450,
                margin=dict(l=60, r=30, t=60, b=50),
                hovermode="x unified",
            )

            st.plotly_chart(fig, use_container_width=True)

        # ---------------------------------------------------------------
        # Spike detail table
        # ---------------------------------------------------------------
        spikes_summary = trends_data.get("spikes_summary", [])

        if spikes_summary:
            st.markdown(
                '<p class="section-header">Detected Spikes</p>',
                unsafe_allow_html=True,
            )

            spike_rows = []
            for spike in sorted(spikes_summary, key=lambda s: -(s.get("z_score") or 0)):
                fr = spike["fraud_rate"]
                bl = spike.get("baseline_mean")
                zs = spike.get("z_score")
                spike_rows.append({
                    "Time Bin": spike["hour_bin"],
                    "Fraud Rate": f"{fr:.2%}" if fr is not None else "N/A",
                    "Baseline": f"{bl:.2%}" if bl is not None else "N/A",
                    "Z-Score": round(zs, 2) if zs is not None else None,
                    "Transactions": spike["n_transactions"],
                    "Fraud Count": spike["n_fraud"],
                })

            spikes_df = pd.DataFrame(spike_rows)

            def _highlight_severity(val, col):
                """Apply red/amber text to Fraud Rate and Z-Score cells."""
                if col == "Z-Score" and val is not None:
                    color = "#ef4444" if val > 3 else "#f59e0b"
                    return f"color: {color}; font-weight: 600"
                if col == "Fraud Rate" and val != "N/A":
                    return "color: #ef4444; font-weight: 600"
                return ""

            styled = spikes_df.style.apply(
                lambda s: [_highlight_severity(v, s.name) for v in s], axis=0
            )

            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # ---------------------------------------------------------------
            # Export underlying transactions for a spike
            # ---------------------------------------------------------------
            st.markdown(
                '<p class="section-header">View Underlying Transactions</p>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Select a spike bin to inspect the actual test-set transactions "
                "behind the aggregated fraud rate — verify the numbers directly."
            )

            sorted_spikes_for_export = sorted(
                spikes_summary, key=lambda s: -(s.get("z_score") or 0)
            )
            export_options = [
                f"Bin {s['hour_bin']} — {s['fraud_rate']:.2%} fraud rate "
                f"({s['n_fraud']} fraud / {s['n_transactions']} total)"
                for s in sorted_spikes_for_export
            ]

            selected_export_label = st.selectbox(
                "Select a spike bin to inspect",
                options=export_options,
                key="spike_export_select",
            )
            selected_export_idx = export_options.index(selected_export_label)
            selected_export_spike = sorted_spikes_for_export[selected_export_idx]
            selected_bin = selected_export_spike["hour_bin"]

            if st.button("📋 Load transactions for this bin", key="load_bin_tx_btn"):
                with st.spinner("Loading transactions…"):
                    from backend.fraud_trends import get_transactions_for_bin

                    df_test = pd.read_parquet(
                        os.path.join(ARTIFACTS_DIR, "test.parquet")
                    )
                    window_hours = trends_data.get("window_hours", 4)
                    bin_df = get_transactions_for_bin(df_test, selected_bin, window_hours)

                    st.session_state["spike_bin_df"] = bin_df
                    st.session_state["spike_bin_id"] = selected_bin

            # Show previously loaded transactions (persists across reruns)
            if "spike_bin_df" in st.session_state and st.session_state.get("spike_bin_id") == selected_bin:
                bin_df = st.session_state["spike_bin_df"]

                n_fraud = int(bin_df["isFraud"].sum())
                n_total = len(bin_df)
                fr = n_fraud / n_total if n_total > 0 else 0

                col_a, col_b, col_c = st.columns(3)
                with col_a:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Transactions in Bin</div>
                        <div class="value">{n_total}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_b:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Fraud Count</div>
                        <div class="value" style="color: #ef4444;">{n_fraud}</div>
                    </div>
                    """, unsafe_allow_html=True)
                with col_c:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div class="label">Fraud Rate</div>
                        <div class="value" style="color: #ef4444;">{fr:.2%}</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Show a readable subset of columns first, full data in expander
                display_cols = [
                    c for c in ["isFraud", "TransactionDT", "TransactionAmt",
                                "ProductCD", "card1", "card4", "card6",
                                "P_emaildomain", "addr1", "addr2", "dist1"]
                    if c in bin_df.columns
                ]
                st.dataframe(
                    bin_df[display_cols],
                    use_container_width=True,
                    hide_index=True,
                )

                with st.expander("Show all columns"):
                    st.dataframe(bin_df, use_container_width=True, hide_index=True)

                csv_bytes = bin_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"⬇ Download transactions for bin {selected_bin} as CSV",
                    data=csv_bytes,
                    file_name=f"transactions_bin_{selected_bin}.csv",
                    mime="text/csv",
                    key="download_bin_csv",
                )

            # ---------------------------------------------------------------
            # "Explain this spike" — LLM-powered explanation
            # ---------------------------------------------------------------
            st.markdown(
                '<p class="section-header">AI Spike Explanation</p>',
                unsafe_allow_html=True,
            )

            spike_options = [
                f"Bin {s['hour_bin']} \u2014 {s['fraud_rate']:.2%} fraud rate (z={s.get('z_score', 0):.1f})"
                for s in sorted(spikes_summary, key=lambda s: -(s.get("z_score") or 0))
            ]
            sorted_spikes = sorted(spikes_summary, key=lambda s: -(s.get("z_score") or 0))

            selected_spike_label = st.selectbox(
                "Select a spike to explain",
                options=spike_options,
                key="spike_explain_select",
            )
            selected_spike_idx = spike_options.index(selected_spike_label)
            selected_spike = sorted_spikes[selected_spike_idx]

            if st.button("Explain this spike", key="explain_spike_btn"):
                with st.spinner("Generating explanation ..."):
                    try:
                        from dotenv import load_dotenv
                        from groq import Groq

                        load_dotenv()
                        groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

                        spike_prompt = (
                            f"You are a fraud-risk analyst. Summarize this fraud-rate spike in "
                            f"1-2 factual sentences. Do not invent numbers not given to you. "
                            f"Do not suggest how to exploit the pattern.\n\n"
                            f"Data: In time bin {selected_spike['hour_bin']} (a 4-hour period), "
                            f"the fraud rate was {selected_spike['fraud_rate']:.2%} "
                            f"({selected_spike['n_fraud']} fraud out of "
                            f"{selected_spike['n_transactions']} transactions). "
                            f"The trailing 24-bin rolling baseline was "
                            f"{selected_spike.get('baseline_mean', 0):.2%} "
                            f"(z-score: {selected_spike.get('z_score', 0):.2f}). "
                            f"The overall test-set fraud rate is ~3.4%."
                        )

                        response = groq_client.chat.completions.create(
                            model="qwen/qwen3.8-27b",
                            max_tokens=150,
                            messages=[
                                {"role": "system", "content": (
                                    "You are a fraud-risk analyst. Give a brief, factual "
                                    "summary of a detected fraud-rate spike. Use only the "
                                    "numbers provided. Never suggest how to exploit the "
                                    "pattern or evade detection."
                                )},
                                {"role": "user", "content": spike_prompt},
                            ],
                        )
                        explanation = response.choices[0].message.content

                        st.markdown(f"""
                        <div class="narration-line">
                            <span class="narration-icon">🤖</span> {explanation}
                        </div>
                        """, unsafe_allow_html=True)

                    except Exception as e:
                        st.error(f"Could not generate explanation: {str(e)}")

        else:
            st.markdown("""
            <div class="no-action-card">
                <p>No fraud-rate spikes detected in the test set.</p>
            </div>
            """, unsafe_allow_html=True)

