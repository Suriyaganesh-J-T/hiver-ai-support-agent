from pathlib import Path

import duckdb


RELATIONSHIP_FILE = Path(
    "data/processed/tweet_relationships.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/support_case_index.parquet"
)


MAX_DEPTH = 100


def main() -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if not RELATIONSHIP_FILE.exists():
        raise FileNotFoundError(
            f"Missing relationship file: {RELATIONSHIP_FILE}"
        )

    print("=" * 70)
    print("BUILDING SUPPORT CASE INDEX")
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

    print("Reconstructing customer-specific support cases...")
    print("This may take a little while.\n")

    query = f"""
        COPY (

            WITH RECURSIVE walk(
                tweet_id,
                author_id,
                inbound,
                parent_tweet_id,
                case_id,
                customer_id,
                case_depth,
                path
            ) AS (

                -- ------------------------------------------------
                -- ROOT TWEETS
                -- ------------------------------------------------

                SELECT
                    r.tweet_id,
                    r.author_id,
                    r.inbound,
                    r.in_response_to_tweet_id,

                    CASE
                        WHEN r.inbound = TRUE
                        THEN r.tweet_id
                        ELSE CAST(NULL AS VARCHAR)
                    END AS case_id,

                    CASE
                        WHEN r.inbound = TRUE
                        THEN r.author_id
                        ELSE CAST(NULL AS VARCHAR)
                    END AS customer_id,

                    0 AS case_depth,

                    [r.tweet_id] AS path

                FROM rel r

                LEFT JOIN rel parent
                    ON r.in_response_to_tweet_id =
                       parent.tweet_id

                WHERE
                    r.in_response_to_tweet_id IS NULL
                    OR parent.tweet_id IS NULL


                UNION ALL


                -- ------------------------------------------------
                -- CHILD TWEETS
                -- ------------------------------------------------

                SELECT
                    child.tweet_id,
                    child.author_id,
                    child.inbound,
                    child.in_response_to_tweet_id,

                    CASE

                        -- A customer starting a new branch
                        -- becomes a new support case.
                        WHEN child.inbound = TRUE
                             AND (
                                 walk.customer_id IS NULL
                                 OR child.author_id !=
                                    walk.customer_id
                             )
                        THEN child.tweet_id

                        -- Same customer continues the existing case.
                        WHEN child.inbound = TRUE
                             AND child.author_id =
                                 walk.customer_id
                        THEN walk.case_id

                        -- Support message inherits the case.
                        ELSE walk.case_id

                    END AS case_id,


                    CASE

                        WHEN child.inbound = TRUE
                             AND (
                                 walk.customer_id IS NULL
                                 OR child.author_id !=
                                    walk.customer_id
                             )
                        THEN child.author_id

                        WHEN child.inbound = TRUE
                             AND child.author_id =
                                 walk.customer_id
                        THEN walk.customer_id

                        ELSE walk.customer_id

                    END AS customer_id,


                    CASE

                        WHEN child.inbound = TRUE
                             AND (
                                 walk.customer_id IS NULL
                                 OR child.author_id !=
                                    walk.customer_id
                             )
                        THEN 0

                        ELSE walk.case_depth + 1

                    END AS case_depth,


                    list_append(
                        walk.path,
                        child.tweet_id
                    ) AS path

                FROM walk

                JOIN rel child
                    ON child.in_response_to_tweet_id =
                       walk.tweet_id

                WHERE
                    walk.case_depth < {MAX_DEPTH}

                    AND NOT list_contains(
                        walk.path,
                        child.tweet_id
                    )
            )


            SELECT
                tweet_id,
                case_id,
                customer_id,
                case_depth

            FROM walk

            WHERE
                case_id IS NOT NULL

        )

        TO '{OUTPUT_FILE}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
    """

    con.execute(query)

    # ------------------------------------------------------------
    # Verification
    # ------------------------------------------------------------

    total_rows = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{OUTPUT_FILE}'
        );
        """
    ).fetchone()[0]

    total_cases = con.execute(
        f"""
        SELECT COUNT(DISTINCT case_id)
        FROM read_parquet(
            '{OUTPUT_FILE}'
        );
        """
    ).fetchone()[0]

    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Indexed tweets : {total_rows:,}")
    print(f"Support cases  : {total_cases:,}")

    print("\nOutput:")
    print(OUTPUT_FILE)

    con.close()


if __name__ == "__main__":
    main()