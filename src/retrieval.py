import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


DATA = Path(
    "data/processed/amazon_clean_cases.parquet"
)

INDEX = Path(
    "data/processed/tfidf_index.pkl"
)


def main():
    df = pd.read_parquet(DATA)

    # Keep the retrieval corpus manageable.
    df = df.sample(
        min(20000, len(df)),
        random_state=42
    ).reset_index(drop=True)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        max_features=50000,
    )

    matrix = vectorizer.fit_transform(
        df["customer_text"]
    )

    with open(INDEX, "wb") as f:
        pickle.dump(
            {
                "df": df,
                "vectorizer": vectorizer,
                "matrix": matrix,
            },
            f
        )

    print("TF-IDF retrieval index created.")
    print("Cases:", len(df))
    print("Vocabulary:", len(vectorizer.vocabulary_))
    print("Saved:", INDEX)


if __name__ == "__main__":
    main()