from pathlib import Path

import duckdb


CASE_INDEX = Path(
    "data/processed/support_case_index.parquet"
)

METADATA_FILE = Path(
    "data/processed/tweet_metadata.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/corrected_brand_case_statistics.csv"
)

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "AmericanAir",
    "VirginTrains",
]


def main() -> None:
    if not CASE_INDEX.exists():
        raise FileNotFoundError(CASE_INDEX)

    if not METADATA_FILE.exists():
        raise FileNotFoundError(METADATA_FILE)

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = duckdb.connect()

    print("=" * 70)
    print("CORRECTED BRAND CASE ANALYSIS")
    print("=" * 70)

    # ------------------------------------------------------------
    # Load the corrected case index.
    # ------------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE VIEW cases AS
        SELECT
            tweet_id,
            case_id,
            customer_id,
            case_depth
        FROM read_parquet(
            '{CASE_INDEX}'
        );
        """
    )

    # ------------------------------------------------------------
    # Load tweet metadata.
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
    # Attach metadata to every indexed message.
    # ------------------------------------------------------------

    print("\nJoining case index with tweet metadata...")

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW case_messages AS

        SELECT
            c.case_id,
            c.customer_id,
            c.case_depth,
            t.tweet_id,
            t.author_id,
            t.inbound,
            t.created_at,
            t.text

        FROM cases c

        JOIN tweets t
            ON c.tweet_id = t.tweet_id;
        """
    )

    # ------------------------------------------------------------
    # Build case-level statistics.
    #
    # A case can contain customer + one or more support accounts.
    # We specifically identify the cases containing each candidate.
    # ------------------------------------------------------------

    print("Building case-level statistics...")

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW case_stats AS

        SELECT
            case_id,

            MAX(customer_id) AS customer_id,

            COUNT(*) AS message_count,

            COUNT(*) FILTER (
                WHERE inbound = TRUE
            ) AS customer_messages,

            COUNT(*) FILTER (
                WHERE inbound = FALSE
            ) AS support_messages,

            COUNT(DISTINCT author_id) FILTER (
                WHERE inbound = FALSE
            ) AS support_account_count,

            COUNT(DISTINCT author_id) FILTER (
                WHERE inbound = FALSE
            ) > 1 AS mixed_support_case

        FROM case_messages

        GROUP BY case_id;
        """
    )

    brand_list = ", ".join(
        f"'{brand}'" for brand in BRANDS
    )

    # ------------------------------------------------------------
    # Candidate-specific case statistics.
    # ------------------------------------------------------------

    print("Calculating candidate brand statistics...")

    query = f"""
        WITH brand_cases AS (

            SELECT DISTINCT
                cm.author_id AS support_account,
                cm.case_id

            FROM case_messages cm

            WHERE
                cm.inbound = FALSE
                AND cm.author_id IN ({brand_list})
        ),

        enriched AS (

            SELECT
                bc.support_account,
                bc.case_id,

                cs.customer_id,
                cs.message_count,
                cs.customer_messages,
                cs.support_messages,
                cs.support_account_count,
                cs.mixed_support_case

            FROM brand_cases bc

            JOIN case_stats cs
                ON bc.case_id = cs.case_id
        )

        SELECT
            support_account,

            COUNT(*) AS total_cases,

            COUNT(DISTINCT customer_id)
                AS unique_customers,

            ROUND(
                AVG(message_count),
                2
            ) AS avg_messages_per_case,

            ROUND(
                MEDIAN(message_count),
                2
            ) AS median_messages_per_case,

            ROUND(
                AVG(customer_messages),
                2
            ) AS avg_customer_messages,

            ROUND(
                AVG(support_messages),
                2
            ) AS avg_support_messages,

            COUNT(*) FILTER (
                WHERE message_count >= 3
            ) AS cases_3plus,

            ROUND(
                100.0 *
                COUNT(*) FILTER (
                    WHERE message_count >= 3
                )
                / COUNT(*),
                2
            ) AS pct_3plus,

            COUNT(*) FILTER (
                WHERE message_count >= 5
            ) AS cases_5plus,

            ROUND(
                100.0 *
                COUNT(*) FILTER (
                    WHERE message_count >= 5
                )
                / COUNT(*),
                2
            ) AS pct_5plus,

            COUNT(*) FILTER (
                WHERE message_count >= 7
            ) AS cases_7plus,

            ROUND(
                100.0 *
                COUNT(*) FILTER (
                    WHERE message_count >= 7
                )
                / COUNT(*),
                2
            ) AS pct_7plus,

            COUNT(*) FILTER (
                WHERE mixed_support_case = TRUE
            ) AS mixed_support_cases,

            ROUND(
                100.0 *
                COUNT(*) FILTER (
                    WHERE mixed_support_case = TRUE
                )
                / COUNT(*),
                2
            ) AS pct_mixed_support_cases,

            COUNT(*) FILTER (
                WHERE
                    mixed_support_case = FALSE
            ) AS clean_cases

        FROM enriched

        GROUP BY support_account

        ORDER BY total_cases DESC;
    """

    results = con.execute(query).df()

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print("\n" + "=" * 70)
    print("CORRECTED RESULTS")
    print("=" * 70)

    print(results.to_string(index=False))

    print("\nSaved to:")
    print(OUTPUT_FILE)

    con.close()


if __name__ == "__main__":
    main()