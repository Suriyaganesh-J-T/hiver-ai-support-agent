from pathlib import Path
import duckdb
import pandas as pd


DATASET_PATH = Path(
    "/Users/suriyaganeshjt/.cache/kagglehub/datasets/"
    "thoughtvector/customer-support-on-twitter/versions/10/"
    "twcs/twcs.csv"
)

OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "brand_statistics.csv"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    print("=" * 70)
    print("HIVER SUPPORT AGENT - BRAND ANALYSIS")
    print("=" * 70)

    print(f"\nDataset:")
    print(DATASET_PATH)

    print("\nChecking dataset...")

    con = duckdb.connect()

    # Read the CSV with IDs explicitly treated as strings.
    con.execute(
        f"""
        CREATE OR REPLACE VIEW tweets AS
        SELECT *
        FROM read_csv(
            '{DATASET_PATH}',
            header = true,
            columns = {{
                'tweet_id': 'VARCHAR',
                'author_id': 'VARCHAR',
                'inbound': 'BOOLEAN',
                'created_at': 'VARCHAR',
                'text': 'VARCHAR',
                'response_tweet_id': 'VARCHAR',
                'in_response_to_tweet_id': 'VARCHAR'
            }}
        );
        """
    )

    # ------------------------------------------------------------
    # 1. Basic dataset statistics
    # ------------------------------------------------------------

    basic_stats = con.execute(
        """
        SELECT
            COUNT(*) AS total_tweets,
            COUNT(DISTINCT author_id) AS unique_authors,
            COUNT(*) FILTER (WHERE inbound = TRUE)
                AS customer_messages,
            COUNT(*) FILTER (WHERE inbound = FALSE)
                AS support_messages
        FROM tweets;
        """
    ).fetchone()

    total_tweets, unique_authors, customer_messages, support_messages = (
        basic_stats
    )

    print("\nGLOBAL DATASET STATISTICS")
    print("-" * 70)
    print(f"Total tweets       : {total_tweets:,}")
    print(f"Unique authors     : {unique_authors:,}")
    print(f"Customer messages  : {customer_messages:,}")
    print(f"Support messages   : {support_messages:,}")

    # ------------------------------------------------------------
    # 2. Identify support accounts
    # ------------------------------------------------------------

    print("\nAnalyzing support accounts...")

    support_stats = con.execute(
        """
        SELECT
            author_id AS support_account,
            COUNT(*) AS support_messages,
            COUNT(DISTINCT tweet_id) AS unique_support_tweets
        FROM tweets
        WHERE inbound = FALSE
        GROUP BY author_id
        ORDER BY support_messages DESC;
        """
    ).df()

    # ------------------------------------------------------------
    # 3. Count customers directly replying to support tweets
    # ------------------------------------------------------------

    print("Analyzing customer → support relationships...")

    customer_to_support = con.execute(
        """
        SELECT
            s.author_id AS support_account,
            COUNT(DISTINCT c.tweet_id) AS customer_replies,
            COUNT(DISTINCT c.author_id) AS unique_customers
        FROM tweets c
        JOIN tweets s
            ON c.in_response_to_tweet_id = s.tweet_id
        WHERE c.inbound = TRUE
          AND s.inbound = FALSE
        GROUP BY s.author_id
        ORDER BY customer_replies DESC;
        """
    ).df()

    # ------------------------------------------------------------
    # 4. Count customer messages that have a support response
    # ------------------------------------------------------------

    print("Analyzing support responses to customer messages...")

    customer_to_support_response = con.execute(
        """
        WITH customer_support_links AS (
            SELECT
                c.tweet_id AS customer_tweet_id,
                s.author_id AS support_account
            FROM tweets c
            CROSS JOIN UNNEST(
                string_split(
                    regexp_replace(
                        COALESCE(c.response_tweet_id, ''),
                        '\\s+',
                        '',
                        'g'
                    ),
                    ','
                )
            ) AS response_id
            JOIN tweets s
                ON s.tweet_id = response_id.unnest
            WHERE c.inbound = TRUE
              AND s.inbound = FALSE
              AND COALESCE(c.response_tweet_id, '') != ''
        )
        SELECT
            support_account,
            COUNT(DISTINCT customer_tweet_id)
                AS customer_messages_with_support_response
        FROM customer_support_links
        GROUP BY support_account
        ORDER BY customer_messages_with_support_response DESC;
        """
    ).df()

    # ------------------------------------------------------------
    # 5. Merge all brand statistics
    # ------------------------------------------------------------

    results = support_stats.merge(
        customer_to_support,
        on="support_account",
        how="left",
    )

    results = results.merge(
        customer_to_support_response,
        on="support_account",
        how="left",
    )

    numeric_columns = [
        "customer_replies",
        "unique_customers",
        "customer_messages_with_support_response",
    ]

    for column in numeric_columns:
        if column in results.columns:
            results[column] = results[column].fillna(0).astype(int)

    # A simple exploratory measure of how much two-sided interaction
    # each support account has.
    results["interaction_volume"] = (
        results["customer_replies"]
        + results["customer_messages_with_support_response"]
    )

    results = results.sort_values(
        [
            "interaction_volume",
            "support_messages",
        ],
        ascending=False,
    )

    # ------------------------------------------------------------
    # 6. Save results
    # ------------------------------------------------------------

    results.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("TOP SUPPORT ACCOUNTS")
    print("=" * 70)

    print(
        results.head(30).to_string(index=False)
    )

    print("\nSaved results to:")
    print(OUTPUT_FILE)

    con.close()


if __name__ == "__main__":
    main()