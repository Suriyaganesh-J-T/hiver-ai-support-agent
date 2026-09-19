from pathlib import Path
import duckdb


RELATIONSHIP_FILE = Path(
    "data/processed/tweet_relationships.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/conversation_index.parquet"
)


MAX_DEPTH = 50


def main() -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not RELATIONSHIP_FILE.exists():
        raise FileNotFoundError(
            f"Missing: {RELATIONSHIP_FILE}"
        )

    print("=" * 70)
    print("BUILDING CONVERSATION INDEX")
    print("=" * 70)

    con = duckdb.connect()

    print("\nReading relationship index...")

    con.execute(
        f"""
        CREATE OR REPLACE VIEW rel AS
        SELECT
            tweet_id,
            author_id,
            inbound,
            in_response_to_tweet_id
        FROM read_parquet(
            '{RELATIONSHIP_FILE}'
        );
        """
    )

    print("Finding conversation roots...")

    query = f"""
        COPY (

            WITH RECURSIVE

            roots AS (
                SELECT
                    r.tweet_id,
                    r.tweet_id AS conversation_id,
                    0 AS depth
                FROM rel r
                LEFT JOIN rel p
                    ON r.in_response_to_tweet_id = p.tweet_id
                WHERE
                    r.in_response_to_tweet_id IS NULL
                    OR p.tweet_id IS NULL
            ),

            conversation_tree(
                tweet_id,
                conversation_id,
                depth
            ) AS (

                SELECT
                    tweet_id,
                    conversation_id,
                    depth
                FROM roots

                UNION ALL

                SELECT
                    child.tweet_id,
                    tree.conversation_id,
                    tree.depth + 1
                FROM rel child
                JOIN conversation_tree tree
                    ON child.in_response_to_tweet_id =
                       tree.tweet_id
                WHERE
                    tree.depth < {MAX_DEPTH}
            )

            SELECT DISTINCT
                tweet_id,
                conversation_id,
                depth
            FROM conversation_tree

        )

        TO '{OUTPUT_FILE}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
    """

    print("\nBuilding conversation mapping...")
    print("This may take a little while.\n")

    con.execute(query)

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{OUTPUT_FILE}'
        );
        """
    ).fetchone()[0]

    conversation_count = con.execute(
        f"""
        SELECT COUNT(DISTINCT conversation_id)
        FROM read_parquet(
            '{OUTPUT_FILE}'
        );
        """
    ).fetchone()[0]

    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Indexed tweets       : {count:,}")
    print(f"Conversations        : {conversation_count:,}")
    print(f"Output               : {OUTPUT_FILE}")

    con.close()


if __name__ == "__main__":
    main()