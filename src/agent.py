import json
import os
import pickle
from pathlib import Path

from google import genai
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# Configuration
# ============================================================

INDEX = Path(
    "data/processed/tfidf_index.pkl"
)

MODEL = "gemini-3.6-flash"

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


# ============================================================
# Gemini Client
# ============================================================

if not os.getenv("GEMINI_API_KEY"):
    raise RuntimeError(
        "GEMINI_API_KEY is not set.\n"
        "Run:\n"
        'export GEMINI_API_KEY="YOUR_API_KEY"'
    )

client = genai.Client()


# ============================================================
# Gemini helper
# ============================================================

def call_gemini(prompt: str) -> str:
    """
    Send a stateless request to Gemini using the
    current Interactions API.
    """

    response = client.interactions.create(
        model=MODEL,
        input=prompt,
        store=False,
    )

    return response.output_text.strip()


# ============================================================
# Intent Classification
# ============================================================

def classify_intent(message: str) -> dict:
    """
    Classify an incoming customer message into one
    of the predefined AmazonHelp intents.
    """

    prompt = f"""
You are an Amazon customer-support intent classifier.

Choose exactly ONE intent from the following list:

{json.dumps(INTENTS, indent=2)}

Return ONLY valid JSON in this exact format:

{{
  "intent": "one_intent_from_list",
  "confidence": 0.0
}}

Rules:
- intent must be exactly one item from the list.
- confidence must be a number between 0 and 1.
- Do not include markdown.
- Do not include explanations.

Customer message:
{message}
"""

    text = call_gemini(prompt)

    # Remove accidental markdown fences.
    text = text.replace("```json", "")
    text = text.replace("```", "")
    text = text.strip()

    try:
        result = json.loads(text)

        intent = result.get("intent", "other")
        confidence = float(
            result.get("confidence", 0.0)
        )

        if intent not in INTENTS:
            intent = "other"

        confidence = max(
            0.0,
            min(1.0, confidence)
        )

        return {
            "intent": intent,
            "confidence": confidence,
        }

    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return {
            "intent": "other",
            "confidence": 0.0,
        }


# ============================================================
# Historical Retrieval
# ============================================================

def retrieve(
    message: str,
    k: int = 3,
) -> list[dict]:
    """
    Retrieve the top-k historically similar AmazonHelp cases
    using the TF-IDF index.
    """

    if not INDEX.exists():
        raise FileNotFoundError(
            f"Retrieval index not found: {INDEX}"
        )

    with open(INDEX, "rb") as f:
        index = pickle.load(f)

    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    df = index["df"]

    query_vector = vectorizer.transform(
        [message]
    )

    scores = cosine_similarity(
        query_vector,
        matrix,
    )[0]

    top_indices = scores.argsort()[
        ::-1
    ][:k]

    results = []

    for idx in top_indices:

        row = df.iloc[idx]

        results.append(
            {
                "case_id": str(row["case_id"]),
                "customer_message": str(
                    row["customer_text"]
                ),
                "historical_response": str(
                    row["support_text"]
                ),
                "similarity": round(
                    float(scores[idx]),
                    4,
                ),
            }
        )

    return results


# ============================================================
# Escalation Decision
# ============================================================

def decide_escalation(
    message: str,
    intent: str,
    confidence: float,
) -> tuple[str, str]:
    """
    Apply a simple risk-based escalation policy.
    """

    high_risk_intents = {
        "payment_billing",
        "return_refund",
    }

    risk_terms = [
        "fraud",
        "hacked",
        "stolen",
        "unauthorized",
        "charged twice",
        "scam",
        "police",
        "legal",
    ]

    message_lower = message.lower()

    # Financial/account-specific issues.
    if intent in high_risk_intents:
        return (
            "ESCALATE",
            "The issue may require account or transaction "
            "verification by a human agent.",
        )

    # Explicit high-risk language.
    if any(
        term in message_lower
        for term in risk_terms
    ):
        return (
            "ESCALATE",
            "The message contains a potentially high-risk "
            "issue that should be reviewed by a human.",
        )

    # Uncertain classification.
    if confidence < 0.70:
        return (
            "ESCALATE",
            "Intent classification confidence is low.",
        )

    return (
        "AUTO_HANDLE",
        "Low-risk request with sufficient classification confidence.",
    )


# ============================================================
# Reply Generation
# ============================================================

def generate_reply(
    message: str,
    intent: str,
    evidence: list[dict],
) -> str:
    """
    Generate a historically grounded AmazonHelp response.
    """

    evidence_text = "\n\n".join(
        [
            f"""
Historical case {i + 1}

Customer:
{item["customer_message"]}

AmazonHelp response:
{item["historical_response"]}

Similarity:
{item["similarity"]}
"""
            for i, item in enumerate(evidence)
        ]
    )

    prompt = f"""
You are a customer-support drafting assistant for AmazonHelp.

Current customer message:
{message}

Predicted intent:
{intent}

Historical AmazonHelp examples:
{evidence_text}

Draft ONE concise customer-support reply.

Grounding requirements:
- Use the historical examples as evidence.
- Do not invent Amazon policies.
- Do not invent refunds, credits, replacements, or delivery dates.
- Do not claim an action has already been completed.
- Do not expose private customer information.
- Do not repeat order numbers, phone numbers, emails, or other
  sensitive information from historical examples.
- If the historical evidence is insufficient, recommend further
  assistance instead of fabricating an answer.
- Keep the response professional and concise.
"""

    return call_gemini(prompt)


# ============================================================
# Complete Agent Pipeline
# ============================================================

def run_agent(message: str) -> dict:

    # 1. Classify intent.
    classification = classify_intent(
        message
    )

    # 2. Retrieve historical cases.
    evidence = retrieve(
        message,
        k=3,
    )

    # 3. Decide whether to auto-handle or escalate.
    action, reason = decide_escalation(
        message,
        classification["intent"],
        classification["confidence"],
    )

    # 4. Generate grounded response.
    reply = generate_reply(
        message,
        classification["intent"],
        evidence,
    )

    return {
        "intent": classification["intent"],
        "confidence": classification["confidence"],
        "evidence": evidence,
        "reply": reply,
        "action": action,
        "reason": reason,
    }


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("AMAZONHELP AI SUPPORT AGENT")
    print("=" * 70)

    message = input(
        "\nCustomer message: "
    ).strip()

    if not message:
        raise SystemExit(
            "Customer message cannot be empty."
        )

    result = run_agent(message)

    print("\n" + "=" * 70)
    print("AGENT RESULT")
    print("=" * 70)

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )