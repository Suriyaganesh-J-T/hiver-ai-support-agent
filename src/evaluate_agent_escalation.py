from pathlib import Path
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

RESULTS = Path("evaluation/gemini_intent_results.csv")


def decide_escalation(message, intent, confidence):
    high_risk_intents = {"payment_billing", "return_refund"}

    risk_terms = [
        "fraud", "hacked", "stolen", "unauthorized",
        "charged twice", "scam", "police", "legal"
    ]

    text = str(message).lower()

    if intent in high_risk_intents:
        return "ESCALATE"

    if any(term in text for term in risk_terms):
        return "ESCALATE"

    if confidence < 0.70:
        return "ESCALATE"

    return "AUTO_HANDLE"


def main():
    df = pd.read_csv(RESULTS)

    df["predicted_action"] = df.apply(
        lambda r: decide_escalation(
            r["customer_text"],
            r["predicted_intent"],
            float(r["confidence"])
        ),
        axis=1
    )

    y_true = df["expected_action"]
    y_pred = df["predicted_action"]

    accuracy = accuracy_score(y_true, y_pred)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        pos_label="ESCALATE",
        zero_division=0
    )

    print("=" * 70)
    print("PRODUCTION AGENT ESCALATION EVALUATION")
    print("=" * 70)
    print(f"Examples : {len(df)}")
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    print()
    print("Predicted distribution:")
    print(y_pred.value_counts())

    print()
    print("Confusion matrix [AUTO_HANDLE, ESCALATE]:")
    print(
        confusion_matrix(
            y_true,
            y_pred,
            labels=["AUTO_HANDLE", "ESCALATE"]
        )
    )

    output = Path("evaluation/agent_escalation_results.csv")
    df.to_csv(output, index=False)

    print()
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
