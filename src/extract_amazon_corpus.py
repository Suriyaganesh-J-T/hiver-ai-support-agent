from pathlib import Path

import duckdb


CONVERSATION_INDEX_FILE = Path(
    "data/processed/conversation_index.parquet"
)

METADATA_FILE = Path(
    "data/processed/tweet_metadata.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/amazon_conversations.parquet"
)

BRAND = "AmazonHelp"


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

    print("=" * 70)
    print("EXTRACTING AMAZONHELP CONVERSATION CORPUS")
    print("=" * 70)

    con = duckdb.connect()

    print("\nFinding AmazonHelp conversations...")

    query = f"""
        COPY (

            WITH amazon_conversations AS (

                SELECT DISTINCT
                    c.conversation_id

                FROM read_parquet(
                    '{CONVERSATION_INDEX_FILE}'
                ) c

                JOIN read_parquet(
                    '{METADATA_FILE}'
                ) t
                    ON c.tweet_id = t.tweet_id

                WHERE
                    t.author_id = '{BRAND}'
                    AND t.inbound = FALSE
            )

            SELECT
                c.conversation_id,
                c.depth,
                t.tweet_id,
                t.author_id,
                CASE
                    WHEN t.inbound = TRUE
                    THEN 'customer'
                    ELSE 'support'
                END AS role,
                t.created_at,
                t.text

            FROM read_parquet(
                '{CONVERSATION_INDEX_FILE}'
            ) c

            JOIN read_parquet(
                '{METADATA_FILE}'
            ) t
                ON c.tweet_id = t.tweet_id

            JOIN amazon_conversations ac
                ON c.conversation_id = ac.conversation_id

            ORDER BY
                c.conversation_id,
                t.created_at

        )

        TO '{OUTPUT_FILE}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
    """

    print("Extracting messages...")
    print("Please wait...\n")

    con.execute(query)

    # ------------------------------------------------------------
    # Verify output
    # ------------------------------------------------------------

    message_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{OUTPUT_FILE}')
        """
    ).fetchone()[0]

    conversation_count = con.execute(
        f"""
        SELECT COUNT(DISTINCT conversation_id)
        FROM read_parquet('{OUTPUT_FILE}')
        """
    ).fetchone()[0]

    customer_messages = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{OUTPUT_FILE}')
        WHERE role = 'customer'
        """
    ).fetchone()[0]

    support_messages = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{OUTPUT_FILE}')
        WHERE role = 'support'
        """
    ).fetchone()[0]

    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"AmazonHelp conversations : {conversation_count:,}")
    print(f"Total messages           : {message_count:,}")
    print(f"Customer messages        : {customer_messages:,}")
    print(f"Support messages         : {support_messages:,}")

    print(f"\nSaved to:")
    print(OUTPUT_FILE)

    con.close()


if __name__ == "__main__":
    main()