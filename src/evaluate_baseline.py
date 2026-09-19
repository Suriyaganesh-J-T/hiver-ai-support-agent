from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity

GOLDEN = Path("data/golden/golden_150.csv")


def majority_baseline(df):
    majority_intent = df["intent"].value_counts().idxmax()
    intent_acc = (df["intent"] == majority_intent).mean()

    majority_action = df["expected_action"].value_counts().idxmax()
    action_acc = (df["expected_action"] == majority_action).mean()

    print("=" * 70)
    print("BASELINE 1: MAJORITY CLASS")
    print("=" * 70)
    print(f"Majority intent: {majority_intent}")
    print(f"Intent accuracy: {intent_acc:.4f}")
    print(f"Intent macro-F1: {f1_score(df['intent'], [majority_intent] * len(df), average='macro', zero_division=0):.4f}")
    print()
    print(f"Majority action: {majority_action}")
    print(f"Action accuracy: {action_acc:.4f}")


def tfidf_knn_baseline(df):
    texts = df["customer_text"].fillna("").astype(str).values
    labels = df["intent"].values

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

    y_true = []
    y_pred = []

    for train_idx, test_idx in skf.split(texts, labels):
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=20000,
        )

        X_train = vectorizer.fit_transform(texts[train_idx])
        X_test = vectorizer.transform(texts[test_idx])

        similarity = cosine_similarity(X_test, X_train)
        nearest = similarity.argmax(axis=1)

        predictions = labels[train_idx][nearest]

        y_true.extend(labels[test_idx])
        y_pred.extend(predictions)

    print()
    print("=" * 70)
    print("BASELINE 2: TF-IDF + 1-NEAREST-NEIGHBOR")
    print("=" * 70)
    print(f"Intent accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Intent macro-F1: {f1_score(y_true, y_pred, average='macro', zero_division=0):.4f}")


def main():
    if not GOLDEN.exists():
        raise FileNotFoundError(f"Golden set not found: {GOLDEN}")

    df = pd.read_csv(GOLDEN)

    required = {"customer_text", "intent", "expected_action"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if len(df) != 150:
        print(f"Warning: expected 150 rows, found {len(df)}")

    majority_baseline(df)
    tfidf_knn_baseline(df)


if __name__ == "__main__":
    main()
