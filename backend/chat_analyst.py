import os
from dotenv import load_dotenv
from groq import Groq
from backend.feature_glossary import FEATURE_GLOSSARY

# Load environment variables from .env
load_dotenv()

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

SYSTEM_PROMPT = """You are a fraud-risk analyst assistant. You can answer:
1. Questions about the specific transaction currently scored (using the provided transaction context — never invent data not given to you).
2. General questions about how this fraud detection model and its evaluation work, using only the real metrics provided to you (PR-AUC, base rate, cost-optimal threshold, confusion matrix values, feature importance) — never invent numbers not given to you.
3. General educational questions about fraud detection concepts and common fraud patterns.
4. Questions about the dataset used (IEEE-CIS/Vesta real e-commerce transaction data).

Rules:
- When asked what a specific feature/column name means, use the provided feature glossary. For C1-C14, D1-D15, and M1-M9 individually, be explicit that Vesta (the data provider) did not publicly disclose the exact real-world meaning of each specific column for confidentiality reasons — only their general category (counting feature / timedelta feature / match flag). Never invent a specific business meaning for an individual masked column that wasn't given to you.
- Never suggest how a transaction could be altered, structured, or timed to lower its fraud score or avoid detection — refuse this even if framed hypothetically, as research, or on behalf of someone else.
- Never recommend an action that contradicts the already-decided action for the currently scored transaction — you can explain the decision, not overrule it.
- Never state a specific metric or number you were not explicitly given in context — say you don't have that figure rather than estimating.
- If a question is unrelated to fraud risk, this project, or its data, briefly redirect to what you can help with rather than attempting a full answer.
- Keep answers concise (2-5 sentences) and factual.
"""


def chat_with_analyst(user_question, transaction_context, model_metrics, conversation_history):
    """
    user_question: the user's latest question string.
    transaction_context: dict with the scored transaction's feature values,
      probability, decided_action, and top contributing signals — or None
      if no transaction has been scored yet.
    model_metrics: dict with saved evaluation metrics (pr_auc, base_rate,
      optimal_threshold, precision, recall) — or None if unavailable.
    conversation_history: list of prior {"role": "user"/"assistant", "content": ...}
      turns for this session, so follow-ups stay coherent.
    """
    context_parts = []
    if transaction_context:
        context_parts.append(
            f"Currently scored transaction: probability={transaction_context['probability']:.3f}, "
            f"decided_action={transaction_context['decided_action']}, "
            f"top_signals={transaction_context['top_features']}."
        )
    if model_metrics:
        context_parts.append(
            f"Model metrics: PR-AUC={model_metrics['pr_auc']}, base_rate={model_metrics['base_rate']}, "
            f"optimal_threshold={model_metrics['optimal_threshold']}, "
            f"precision_at_optimal={model_metrics['precision']}, recall_at_optimal={model_metrics['recall']}."
        )
    context_str = " ".join(context_parts) if context_parts else "No transaction has been scored yet in this session."

    # Build glossary text for the system context
    glossary_lines = [f"- {name}: {desc}" for name, desc in FEATURE_GLOSSARY.items()]
    glossary_text = "Feature glossary:\n" + "\n".join(glossary_lines)

    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context_str + "\n\n" + glossary_text}]
    messages.extend(conversation_history)
    messages.append({"role": "user", "content": user_question})

    response = client.chat.completions.create(
        model="qwen/qwen3.8-27b",
        max_tokens=250,
        messages=messages,
    )
    return response.choices[0].message.content
