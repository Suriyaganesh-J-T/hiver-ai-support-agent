from pathlib import Path

import duckdb


RELATIONSHIP_FILE = Path(
    "data/processed/tweet_relationships.parquet"
)

CONVERSATION_INDEX_FILE = Path(
    "data/processed/conversation_index.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/candidate_conversation_quality.csv"
)

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "AmericanAir",
    "VirginTrains",
]


def main() -> None:
    if not RELATIONSHIP_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {RELATIONSHIP_FILE}"
        )

    if not CONVERSATION_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {CONVERSATION_INDEX_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    con = duckdb.connect()

    print("=" * 70)
    print("CANDIDATE BRAND - CONVERSATION QUALITY ANALYSIS")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load relationship data.
    # ------------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE VIEW rel AS
        SELECT
            tweet_id,
            author_id,
            inbound
        FROM read_parquet(
            '{RELATIONSHIP_FILE}'
        );
        """
    )

    # ------------------------------------------------------------
    # Load conversation index.
    # ------------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE VIEW conv AS
        SELECT
            tweet_id,
            conversation_id,
            depth
        FROM read_parquet(
            '{CONVERSATION_INDEX_FILE}'
        );
        """
    )

    # ------------------------------------------------------------
    # First calculate statistics for every conversation.
    # ------------------------------------------------------------

    print("\nCalculating conversation-level statistics...")

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW conversation_stats AS

        SELECT
            c.conversation_id,

            COUNT(*) AS message_count,

            COUNT(*) FILTER (
                WHERE r.inbound = TRUE
            ) AS customer_messages,

            COUNT(*) FILTER (
                WHERE r.inbound = FALSE
            ) AS support_messages,

            COUNT(*) FILTER (
                WHERE r.inbound = TRUE
            ) > 0
            AND
            COUNT(*) FILTER (
                WHERE r.inbound = FALSE
            ) > 0
            AS has_customer_and_support

        FROM conv c

        JOIN rel r
            ON c.tweet_id = r.tweet_id

        GROUP BY
            c.conversation_id
        """
    )

    # ------------------------------------------------------------
    # Calculate brand-specific conversation quality.
    # ------------------------------------------------------------

    brand_values = ", ".join(
        f"'{brand}'" for brand in BRANDS
    )

    query = f"""
        WITH brand_conversations AS (

            SELECT DISTINCT
                r.author_id AS support_account,
                c.conversation_id
            FROM rel r
            JOIN conv c
                ON r.tweet_id = c.tweet_id
            WHERE
                r.author_id IN ({brand_values})
                AND r.inbound = FALSE
        )

        SELECT
            bc.support_account,

            COUNT(*) AS conversations,

            ROUND(
                AVG(cs.message_count),
                2
            ) AS avg_messages_per_conversation,

            ROUND(
                MEDIAN(cs.message_count),
                2
            ) AS median_messages_per_conversation,

            ROUND(
                AVG(cs.customer_messages),
                2
            ) AS avg_customer_messages,

            ROUND(
                AVG(cs.support_messages),
                2
            ) AS avg_support_messages,

            COUNT(*) FILTER (
                WHERE cs.message_count >= 3
            ) AS multi_turn_conversations,

            ROUND(
                100.0 *
                COUNT(*) FILTER (
                    WHERE cs.message_count >= 3
                )
                / COUNT(*),
                2
            ) AS multi_turn_percentage,

            COUNT(*) FILTER (
                WHERE cs.has_customer_and_support
            ) AS two_sided_conversations,

            ROUND(
                100.0 *
                COUNT(*) FILTER (
                    WHERE cs.has_customer_and_support
                )
                / COUNT(*),
                2
            ) AS two_sided_percentage

        FROM brand_conversations bc

        JOIN conversation_stats cs
            ON bc.conversation_id = cs.conversation_id

        GROUP BY
            bc.support_account

        ORDER BY
            conversations DESC
    """

    results = con.execute(query).df()

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(
        results.to_string(index=False)
    )

    print("\nSaved to:")
    print(OUTPUT_FILE)

    con.close()


if __name__ == "__main__":
    main()