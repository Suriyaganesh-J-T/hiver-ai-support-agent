from pathlib import Path
import duckdb


CONVERSATION_INDEX_FILE = Path(
    "data/processed/conversation_index.parquet"
)

METADATA_FILE = Path(
    "data/processed/tweet_metadata.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/resolution_patterns.csv"
)

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "AmericanAir",
    "VirginTrains",
]


def main() -> None:
    if not CONVERSATION_INDEX_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {CONVERSATION_INDEX_FILE}"
        )

    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {METADATA_FILE}"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    con = duckdb.connect()

    print("=" * 70)
    print("RESOLUTION PATTERN ANALYSIS")
    print("=" * 70)

    # ------------------------------------------------------------
    # Conversation index
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
    # Tweet metadata
    # ------------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE VIEW tweets AS
        SELECT
            tweet_id,
            author_id,
            inbound,
            created_at,
            text
        FROM read_parquet(
            '{METADATA_FILE}'
        );
        """
    )

    # ------------------------------------------------------------
    # Conversation-level information.
    # ------------------------------------------------------------

    print("\nBuilding conversation-level statistics...")

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW conversation_info AS

        WITH ordered AS (
            SELECT
                c.conversation_id,
                c.tweet_id,
                c.depth,
                t.author_id,
                t.inbound,
                t.created_at,

                ROW_NUMBER() OVER (
                    PARTITION BY c.conversation_id
                    ORDER BY t.created_at DESC
                ) AS reverse_position

            FROM conv c

            JOIN tweets t
                ON c.tweet_id = t.tweet_id
        )

        SELECT
            conversation_id,

            COUNT(*) AS message_count,

            MAX(depth) AS max_depth,

            COUNT(*) FILTER (
                WHERE inbound = TRUE
            ) AS customer_messages,

            COUNT(*) FILTER (
                WHERE inbound = FALSE
            ) AS support_messages,

            MAX(
                CASE
                    WHEN reverse_position = 1
                    THEN inbound
                END
            ) AS final_message_is_customer,

            MAX(
                CASE
                    WHEN reverse_position = 1
                    THEN author_id
                END
            ) AS final_author,

            MAX(
                CASE
                    WHEN reverse_position = 1
                    THEN created_at
                END
            ) AS final_timestamp

        FROM ordered

        GROUP BY conversation_id
        """
    )

    # ------------------------------------------------------------
    # Find conversations associated with each candidate brand.
    # ------------------------------------------------------------

    candidate_list = ", ".join(
        f"'{brand}'" for brand in BRANDS
    )

    query = f"""
        WITH brand_conversations AS (

            SELECT DISTINCT
                t.author_id AS support_account,
                c.conversation_id

            FROM conv c

            JOIN tweets t
                ON c.tweet_id = t.tweet_id

            WHERE
                t.author_id IN ({candidate_list})
                AND t.inbound = FALSE
        )

        SELECT
            bc.support_account,

            COUNT(*) AS conversations,

            ROUND(
                AVG(ci.message_count),
                2
            ) AS avg_messages,

            ROUND(
                MEDIAN(ci.message_count),
                2
            ) AS median_messages,

            COUNT(*) FILTER (
                WHERE ci.message_count >= 3
            ) AS conversations_3plus,

            ROUND(
                100.0 *
                COUNT(*) FILTER (
                    WHERE ci.message_count >= 3
                ) / COUNT(*),
                2
            ) AS pct_3plus,

            COUNT(*) FILTER (
                WHERE ci.message_count >= 4
            ) AS conversations_4plus,

            ROUND(
                100.0 *
                COUNT(*) FILTER (
                    WHERE ci.message_count >= 4
                ) / COUNT(*),
                2
            ) AS pct_4plus,

            COUNT(*) FILTER (
                WHERE ci.message_count >= 6
            ) AS conversations_6plus,

            ROUND(
                100.0 *
                COUNT(*) FILTER (
                    WHERE ci.message_count >= 6
                ) / COUNT(*),
                2
            ) AS pct_6plus,

            COUNT(*) FILTER (
                WHERE ci.final_message_is_customer = FALSE
            ) AS ending_with_support,

            ROUND(
                100.0 *
                COUNT(*) FILTER (
                    WHERE ci.final_message_is_customer = FALSE
                ) / COUNT(*),
                2
            ) AS pct_ending_with_support,

            COUNT(*) FILTER (
                WHERE ci.final_message_is_customer = TRUE
            ) AS ending_with_customer,

            ROUND(
                100.0 *
                COUNT(*) FILTER (
                    WHERE ci.final_message_is_customer = TRUE
                ) / COUNT(*),
                2
            ) AS pct_ending_with_customer

        FROM brand_conversations bc

        JOIN conversation_info ci
            ON bc.conversation_id = ci.conversation_id

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