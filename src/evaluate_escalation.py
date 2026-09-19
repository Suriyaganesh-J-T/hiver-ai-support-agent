from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
)

GOLDEN = Path("data/golden/golden_150.csv")


def escalation_rule(message: str) -> str:
    m = str(message).lower()

    high_risk_terms = [
        "fraud",
        "hacked",
        "stolen",
        "unauthorized",
        "scam",
        "police",
        "legal",
        "court",
        "consumer court",
        "charged twice",
        "charged 3",
        "refund",
        "money not refunded",
        "not received",
        "not delivered",
        "missing",
        "wrong item",
        "defective",
        "account",
        "can't log",
        "cannot log",
        "locked",
    ]

    if any(term in m for term in high_risk_terms):
        return "ESCALATE"

    return "AUTO_HANDLE"


def main():
    if not GOLDEN.exists():
        raise FileNotFoundError(f"Golden set not found: {GOLDEN}")

    df = pd.read_csv(GOLDEN)

    required = {"customer_text", "expected_action"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    predictions = df["customer_text"].apply(escalation_rule)

    y_true = df["expected_action"]
    y_pred = predictions

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label="ESCALATE",
        zero_division=0,
    )

    print("=" * 70)
    print("ESCALATION BASELINE")
    print("=" * 70)
    print(f"Examples : {len(df)}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    print()
    print("Predicted distribution:")
    print(predictions.value_counts())

    print()
    print("Actual distribution:")
    print(y_true.value_counts())

    print()
    print("Confusion matrix [AUTO_HANDLE, ESCALATE]:")
    labels = ["AUTO_HANDLE", "ESCALATE"]
    print(confusion_matrix(y_true, y_pred, labels=labels))

    # Save per-example results for analysis.
    output = Path("evaluation/escalation_results.csv")

    result = df.copy()
    result["predicted_action"] = predictions
    result["correct"] = (
        result["expected_action"] == result["predicted_action"]
    )

    result.to_csv(output, index=False)

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
