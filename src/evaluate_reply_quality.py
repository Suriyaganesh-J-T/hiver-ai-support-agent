from pathlib import Path
import json
import pandas as pd
from google import genai
from sklearn.metrics.pairwise import cosine_similarity
import pickle

GOLDEN = Path("data/golden/golden_150.csv")
INDEX = Path("data/processed/tfidf_index.pkl")
OUTPUT = Path("evaluation/reply_quality.csv")

MODEL = "gemini-3.5-flash-lite"

client = genai.Client()

def parse_json_array(text):
    text = str(text).strip()

    # Remove markdown code fences if present.
    text = text.replace("```json", "").replace("```", "").strip()

    # Find the JSON array even if the model added a short explanation.
    start = text.find("[")
    end = text.rfind("]")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "Judge did not return a JSON array. Raw output:\\n" + text
        )

    return json.loads(text[start:end + 1])



def retrieve(message, k=3):
    with open(INDEX, "rb") as f:
        index = pickle.load(f)

    vectorizer = index["vectorizer"]
    matrix = index["matrix"]
    cases = index["df"]

    q = vectorizer.transform([message])
    scores = cosine_similarity(q, matrix)[0]
    top = scores.argsort()[::-1][:k]

    results = []

    for idx in top:
        row = cases.iloc[idx]
        results.append({
            "customer": str(row["customer_text"]),
            "response": str(row["support_text"]),
            "similarity": float(scores[idx])
        })

    return results


def generate_replies(rows):
    blocks = []

    for row in rows:
        evidence = retrieve(row["customer_text"])

        evidence_text = "\n\n".join(
            f"Historical example {i+1}:\n"
            f"Customer: {x['customer']}\n"
            f"AmazonHelp: {x['response']}"
            for i, x in enumerate(evidence)
        )

        blocks.append(
            f"""
ID: {row['eval_id']}

Customer:
{row['customer_text']}

Historical evidence:
{evidence_text}
"""
        )

    prompt = f"""
You are evaluating a customer-support reply drafting system for AmazonHelp.

For every customer case below, draft ONE concise reply.

Grounding rules:
- Use the historical examples as evidence.
- Do not invent policies.
- Do not invent refunds, credits, replacements, dates, or completed actions.
- Do not expose private information.
- When evidence is insufficient, say that further assistance is needed.
- Be professional and empathetic.

Return ONLY valid JSON:

[
  {{
    "eval_id": "exact ID",
    "reply": "draft reply"
  }}
]

Cases:

{"\n\n".join(blocks)}
"""

    response = client.interactions.create(
        model=MODEL,
        input=prompt,
        store=False
    )

    return parse_json_array(response.output_text)


def judge_replies(rows):
    blocks = []

    for row in rows:
        blocks.append(
            f"""
ID: {row['eval_id']}

Customer:
{row['customer_text']}

Draft reply:
{row['reply']}

Historical evidence:
{row['evidence']}
"""
        )

    prompt = f"""
You are an expert evaluator of customer-support AI replies.

Score each reply from 1 to 5 on:

1. Correctness
2. Groundedness in historical evidence
3. Relevance
4. Empathy/professionalism
5. Non-fabrication/safety

Overall score is the average of these five dimensions.

Rubric:
1 = unacceptable
2 = poor
3 = acceptable
4 = good
5 = excellent

Return ONLY valid JSON:

[
  {{
    "eval_id": "exact ID",
    "correctness": 1,
    "groundedness": 1,
    "relevance": 1,
    "empathy": 1,
    "safety": 1,
    "overall": 1.0,
    "reason": "brief reason"
  }}
]

Cases:

{"\n\n".join(blocks)}
"""

    response = client.interactions.create(
        model=MODEL,
        input=prompt,
        store=False
    )

    return parse_json_array(response.output_text)


def main():
    df = pd.read_csv(GOLDEN)

    # Fixed deterministic sample for reproducibility.
    sample = df.sample(
        n=min(10, len(df)),
        random_state=42
    )

    replies = generate_replies(
        sample.to_dict("records")
    )

    generated = {
        str(x["eval_id"]): x["reply"]
        for x in replies
    }

    records = []

    for _, row in sample.iterrows():
        evidence = retrieve(row["customer_text"])

        evidence_text = "\n\n".join(
            f"Customer: {x['customer']}\n"
            f"Historical response: {x['response']}"
            for x in evidence
        )

        records.append({
            "eval_id": str(row["eval_id"]),
            "customer_text": row["customer_text"],
            "reply": generated.get(
                str(row["eval_id"]),
                ""
            ),
            "evidence": evidence_text
        })

    judges = judge_replies(records)

    judge_map = {
        str(x["eval_id"]): x
        for x in judges
    }

    for record in records:
        j = judge_map.get(record["eval_id"], {})

        for field in [
            "correctness",
            "groundedness",
            "relevance",
            "empathy",
            "safety",
            "overall",
            "reason"
        ]:
            record[field] = j.get(field)

    out = pd.DataFrame(records)
    out.to_csv(OUTPUT, index=False)

    print("=" * 70)
    print("REPLY QUALITY EVALUATION")
    print("=" * 70)
    print(f"Examples: {len(out)}")

    for col in [
        "correctness",
        "groundedness",
        "relevance",
        "empathy",
        "safety",
        "overall"
    ]:
        print(f"{col:15s}: {out[col].mean():.2f}")

    print()
    print(f"Saved: {OUTPUT}")


if __name__ == "__main__":
    main()
