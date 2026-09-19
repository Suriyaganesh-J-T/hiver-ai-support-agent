from pathlib import Path
import json
import re
import time

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report
from google import genai

GOLDEN = Path("data/golden/golden_150.csv")
OUTPUT = Path("evaluation/gemini_intent_results.csv")
MODEL = "gemini-3.5-flash-lite"

INTENTS = [
    "delivery_tracking",
    "order_product_issue",
    "return_refund",
    "payment_billing",
    "account_access",
    "prime_subscription",
    "technical_device_digital",
    "product_info_availability",
    "complaint_service",
    "other",
]

client = genai.Client()


def classify_batch(rows, retries=3):
    examples = []

    for row in rows:
        examples.append(
            f'ID: {row["eval_id"]}\n'
            f'Customer message: {row["customer_text"]}'
        )

    prompt = f"""
You are an Amazon customer-support intent classifier.

Choose exactly ONE intent for every customer message.

Allowed intents:
{json.dumps(INTENTS, indent=2)}

Return ONLY valid JSON as an array.

Each item must contain:
{{
  "eval_id": "the exact ID provided",
  "intent": "one allowed intent",
  "confidence": 0.0
}}

Rules:
- Do not omit any ID.
- Keep eval_id exactly unchanged.
- intent must be one of the allowed intents.
- confidence must be between 0 and 1.
- No markdown.
- No explanations outside the JSON array.

Customer messages:

{"\n\n".join(examples)}
"""

    for attempt in range(retries):
        try:
            response = client.interactions.create(
                model=MODEL,
                input=prompt,
                store=False,
            )

            text = response.output_text.strip()
            text = re.sub(r"^```(?:json)?", "", text)
            text = re.sub(r"```$", "", text).strip()

            return json.loads(text)

        except Exception as e:
            error_text = str(e)

            if "429" not in error_text and "quota" not in error_text.lower():
                raise

            if attempt == retries - 1:
                raise

            # Gemini often tells us how long to wait.
            match = re.search(r"retry in ([0-9.]+)s", error_text)

            if match:
                wait_time = float(match.group(1)) + 2
            else:
                wait_time = 20

            print(f"  Rate limit. Waiting {wait_time:.1f}s...")
            time.sleep(wait_time)


def main():
    if not GOLDEN.exists():
        raise FileNotFoundError(f"Golden set not found: {GOLDEN}")

    df = pd.read_csv(GOLDEN)

    required = {"eval_id", "customer_text", "intent"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["eval_id"] = df["eval_id"].astype(str)

    batch_size = 50
    all_predictions = []

    print("=" * 70)
    print("GEMINI BATCH INTENT EVALUATION")
    print("=" * 70)
    print(f"Examples: {len(df)}")
    print(f"API calls required: {(len(df) + batch_size - 1) // batch_size}")
    print()

    for start in range(0, len(df), batch_size):

        batch = df.iloc[start:start + batch_size]

        print(
            f"Evaluating examples "
            f"{start + 1}-{start + len(batch)}..."
        )

        try:
            predictions = classify_batch(
                batch.to_dict("records")
            )

            expected_ids = set(batch["eval_id"])
            returned_ids = {
                str(item["eval_id"])
                for item in predictions
            }

            if expected_ids != returned_ids:
                missing_ids = expected_ids - returned_ids
                extra_ids = returned_ids - expected_ids

                raise ValueError(
                    f"ID mismatch. Missing={missing_ids}, Extra={extra_ids}"
                )

            for item in predictions:
                intent = item.get("intent", "other")

                if intent not in INTENTS:
                    intent = "other"

                try:
                    confidence = float(
                        item.get("confidence", 0.0)
                    )
                except (TypeError, ValueError):
                    confidence = 0.0

                confidence = max(0.0, min(1.0, confidence))

                all_predictions.append({
                    "eval_id": str(item["eval_id"]),
                    "predicted_intent": intent,
                    "confidence": confidence,
                })

            print("  Success")

        except Exception as e:
            print(f"  ERROR: {e}")

            # Do NOT invent predictions.
            for _, row in batch.iterrows():
                all_predictions.append({
                    "eval_id": str(row["eval_id"]),
                    "predicted_intent": None,
                    "confidence": None,
                })

    pred_df = pd.DataFrame(all_predictions)

    result_df = df.merge(
        pred_df,
        on="eval_id",
        how="left",
    )

    result_df.to_csv(OUTPUT, index=False)

    valid = result_df.dropna(
        subset=["predicted_intent"]
    ).copy()

    print()
    print("=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"Total examples : {len(result_df)}")
    print(f"Evaluated      : {len(valid)}")
    print(f"Failed         : {len(result_df) - len(valid)}")

    if len(valid) > 0:

        accuracy = accuracy_score(
            valid["intent"],
            valid["predicted_intent"],
        )

        macro_f1 = f1_score(
            valid["intent"],
            valid["predicted_intent"],
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            valid["intent"],
            valid["predicted_intent"],
            average="weighted",
            zero_division=0,
        )

        print(f"Intent accuracy    : {accuracy:.4f}")
        print(f"Intent macro-F1    : {macro_f1:.4f}")
        print(f"Intent weighted-F1 : {weighted_f1:.4f}")

        print()
        print("=" * 70)
        print("PER-INTENT REPORT")
        print("=" * 70)

        print(
            classification_report(
                valid["intent"],
                valid["predicted_intent"],
                zero_division=0,
            )
        )

    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
