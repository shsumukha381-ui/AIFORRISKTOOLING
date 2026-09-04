import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.abspath("backend"))
sys.path.insert(0, PROJECT_ROOT)

from backend.responder import generate_case_note

# Load the cost-optimal threshold from evaluation output (not hardcoded)
ARTIFACTS_DIR = os.path.join(PROJECT_ROOT, "artifacts")
with open(os.path.join(ARTIFACTS_DIR, "eval_summary.json")) as f:
    eval_summary = json.load(f)
optimal_threshold = eval_summary["optimal_threshold"]
print(f"[test] Using optimal_threshold = {optimal_threshold:.4f} (from eval_summary.json)")

# Derive action band boundaries (must match responder.py logic)
above_range = 1.0 - optimal_threshold
hold_ceiling = optimal_threshold + above_range * 0.35
stepup_ceiling = optimal_threshold + above_range * 0.7

test_cases = [
    ("Just below threshold", optimal_threshold - 0.01),
    ("At Hold threshold", optimal_threshold),
    ("Mid-Hold", (optimal_threshold + hold_ceiling) / 2),
    ("At Step-Up threshold", hold_ceiling),
    ("Mid-Step-Up", (hold_ceiling + stepup_ceiling) / 2),
    ("At Decline threshold", stepup_ceiling),
    ("High Decline", 0.90),
]

for name, prob in test_cases:
    print(f"=== {name} (prob: {prob:.3f}) ===")
    res = generate_case_note({}, fraud_prob=prob, optimal_threshold=optimal_threshold)
    print(f"Recommended Action: {res.get('recommended_action')}")
    print(f"Case Note: {res.get('case_note')}")
    print()
