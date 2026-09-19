from pathlib import Path
import duckdb
import pandas as pd


DATASET_PATH = Path(
    "/Users/suriyaganeshjt/.cache/kagglehub/datasets/"
    "thoughtvector/customer-support-on-twitter/versions/10/"
    "twcs/twcs.csv"
)

OUTPUT_FILE = Path(
    "data/processed/conversation_statistics.csv"
)

CANDIDATES = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "AmericanAir",
    "VirginTrains",
]


def main() -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 70)
    print("HIVER SUPPORT AGENT - CONVERSATION QUALITY ANALYSIS")
    print("=" * 70)

    con = duckdb.connect()

    # ------------------------------------------------------------
    # Create a typed view over the CSV.
    # IDs are strings because response_tweet_id can contain
    # multiple comma-separated tweet IDs.
    # ------------------------------------------------------------

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
    # Create a table containing only the candidate brands.
    # ------------------------------------------------------------

    candidate_list = ", ".join(
        f"'{brand}'" for brand in CANDIDATES
    )

    print("\nExtracting candidate support messages...")

    candidate_messages = con.execute(
        f"""
        SELECT
            tweet_id,
            author_id,
            inbound,
            created_at,
            text,
            response_tweet_id,
            in_response_to_tweet_id
        FROM tweets
        WHERE author_id IN ({candidate_list})
        """
    ).df()

    print(
        f"Support messages extracted: "
        f"{len(candidate_messages):,}"
    )

    # ------------------------------------------------------------
    # Find direct customer -> support relationships.
    #
    # A customer message has an in_response_to_tweet_id pointing
    # to a support tweet.
    # ------------------------------------------------------------

    print("\nAnalyzing direct customer/support relationships...")

    direct_customer_replies = con.execute(
        f"""
        SELECT
            s.author_id AS support_account,
            COUNT(DISTINCT c.tweet_id)
                AS customer_messages_replying_to_support,
            COUNT(DISTINCT c.author_id)
                AS unique_customers
        FROM tweets c
        JOIN tweets s
            ON c.in_response_to_tweet_id = s.tweet_id
        WHERE c.inbound = TRUE
          AND s.inbound = FALSE
          AND s.author_id IN ({candidate_list})
        GROUP BY s.author_id
        """
    ).df()

    # ------------------------------------------------------------
    # Find direct support responses to customer messages.
    # ------------------------------------------------------------

    print("Analyzing direct support responses...")

    direct_support_replies = con.execute(
        f"""
        SELECT
            s.author_id AS support_account,
            COUNT(DISTINCT s.tweet_id)
                AS support_replies_to_customers,
            COUNT(DISTINCT c.tweet_id)
                AS customer_messages_with_support_reply
        FROM tweets s
        JOIN tweets c
            ON s.in_response_to_tweet_id = c.tweet_id
        WHERE s.inbound = FALSE
          AND c.inbound = TRUE
          AND s.author_id IN ({candidate_list})
        GROUP BY s.author_id
        """
    ).df()

    # ------------------------------------------------------------
    # Merge the statistics.
    # ------------------------------------------------------------

    results = pd.DataFrame({
        "support_account": CANDIDATES
    })

    results = results.merge(
        direct_customer_replies,
        on="support_account",
        how="left"
    )

    results = results.merge(
        direct_support_replies,
        on="support_account",
        how="left"
    )

    results = results.fillna(0)

    integer_columns = [
        "customer_messages_replying_to_support",
        "unique_customers",
        "support_replies_to_customers",
        "customer_messages_with_support_reply",
    ]

    for column in integer_columns:
        results[column] = results[column].astype(int)

    # ------------------------------------------------------------
    # A useful ratio for comparing candidates.
    # ------------------------------------------------------------

    results["support_replies_per_customer_message"] = (
        results["support_replies_to_customers"]
        / results["customer_messages_with_support_reply"].replace(
            0,
            pd.NA
        )
    ).round(3)

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("CONVERSATION QUALITY SUMMARY")
    print("=" * 70)

    print(
        results.to_string(index=False)
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    con.close()


if __name__ == "__main__":
    main()