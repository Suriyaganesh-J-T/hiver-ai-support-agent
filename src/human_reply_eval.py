from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score

INPUT = Path("evaluation/reply_quality.csv")
OUTPUT = Path("evaluation/reply_quality_human.csv")

df = pd.read_csv(INPUT)

print("=" * 80)
print("HUMAN REPLY-QUALITY EVALUATION")
print("=" * 80)
print("Rate each reply from 1 to 5:")
print("1 = unacceptable")
print("2 = poor")
print("3 = acceptable")
print("4 = good")
print("5 = excellent")
print()

ratings = []

for i, row in df.iterrows():
    print("=" * 80)
    print(f"Example {i + 1}/10")
    print(f"ID: {row['eval_id']}")
    print("-" * 80)
    print("CUSTOMER:")
    print(str(row["customer_text"]))
    print()
    print("AI REPLY:")
    print(str(row["reply"]))
    print()
    print(f"LLM JUDGE OVERALL: {row['overall']}")
    print()

    while True:
        value = input("Your human score (1-5): ").strip()
        try:
            score = int(value)
            if 1 <= score <= 5:
                break
        except ValueError:
            pass
        print("Please enter an integer from 1 to 5.")

    ratings.append(score)
    print()

df["human_overall"] = ratings

# Round judge score to nearest integer for ordinal agreement analysis.
df["judge_overall_rounded"] = df["overall"].round().astype(int)

judge = df["judge_overall_rounded"]
human = df["human_overall"]

mae = (judge - human).abs().mean()
weighted_kappa = cohen_kappa_score(
    judge,
    human,
    weights="quadratic"
)
exact_agreement = (judge == human).mean()

# Pearson correlation of the ordinal ratings.
correlation = judge.corr(human, method="spearman")

df.to_csv(OUTPUT, index=False)

print("=" * 80)
print("HUMAN-JUDGE AGREEMENT")
print("=" * 80)
print(f"Examples                  : {len(df)}")
print(f"Exact agreement            : {exact_agreement:.2%}")
print(f"Mean absolute difference   : {mae:.2f}")
print(f"Weighted Cohen's kappa     : {weighted_kappa:.4f}")
print(f"Spearman correlation       : {correlation:.4f}")
print()
print(f"Saved: {OUTPUT}")
