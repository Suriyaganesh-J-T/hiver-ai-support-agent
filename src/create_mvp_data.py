from pathlib import Path
import duckdb
import pandas as pd

CASE_INDEX = Path("data/processed/support_case_index.parquet")
METADATA = Path("data/processed/tweet_metadata.parquet")

CASE_OUTPUT = Path("data/processed/amazon_clean_cases.parquet")
GOLD_OUTPUT = Path("data/golden/golden_150.csv")


def main():
    CASE_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    GOLD_OUTPUT.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect()

    print("Building clean AmazonHelp case corpus...")

    query = f"""
    WITH case_messages AS (

        SELECT
            c.case_id,
            c.customer_id,
            t.tweet_id,
            t.author_id,
            t.inbound,
            t.created_at,
            t.text

        FROM read_parquet('{CASE_INDEX}') c

        JOIN read_parquet('{METADATA}') t
            ON c.tweet_id = t.tweet_id
    ),

    valid_cases AS (

        SELECT
            case_id

        FROM case_messages

        GROUP BY case_id

        HAVING
            COUNT(*) FILTER (
                WHERE inbound = FALSE
                  AND author_id = 'AmazonHelp'
            ) > 0

            AND COUNT(DISTINCT author_id) FILTER (
                WHERE inbound = FALSE
            ) = 1

            AND COUNT(*) FILTER (
                WHERE inbound = TRUE
            ) > 0
    )

    SELECT
        cm.case_id,
        MAX(cm.customer_id) AS customer_id,

        string_agg(
            CASE
                WHEN cm.inbound = TRUE
                THEN cm.text
            END,
            '\\n'
            ORDER BY cm.created_at
        ) AS customer_text,

        string_agg(
            CASE
                WHEN cm.inbound = FALSE
                THEN cm.text
            END,
            '\\n'
            ORDER BY cm.created_at
        ) AS support_text,

        COUNT(*) AS message_count

    FROM case_messages cm

    JOIN valid_cases vc
        ON cm.case_id = vc.case_id

    GROUP BY
        cm.case_id

    HAVING
        customer_text IS NOT NULL
        AND support_text IS NOT NULL
    """

    cases = con.execute(query).df()

    # Remove extremely small/noisy records.
    cases["customer_text"] = cases["customer_text"].fillna("")
    cases["support_text"] = cases["support_text"].fillna("")

    cases = cases[
        (cases["customer_text"].str.len() >= 15)
        & (cases["support_text"].str.len() >= 15)
    ].copy()

    print(f"Clean Amazon cases: {len(cases):,}")

    cases.to_parquet(
        CASE_OUTPUT,
        index=False
    )

    print(f"Saved: {CASE_OUTPUT}")

    # ---------------------------------------------------------
    # Create a diverse 150-example golden candidate set.
    # We sample from several message-length buckets so that
    # the evaluation set is not dominated by short tweets.
    # ---------------------------------------------------------

    cases["length_bucket"] = pd.qcut(
        cases["customer_text"].str.len(),
        q=10,
        labels=False,
        duplicates="drop"
    )

    golden = (
        cases.groupby(
            "length_bucket",
            group_keys=False
        )
        .apply(
            lambda x: x.sample(
                min(len(x), 15),
                random_state=42
            ),
            include_groups=False
        )
        .reset_index(drop=True)
    )

    golden = golden.sample(
        min(150, len(golden)),
        random_state=42
    ).reset_index(drop=True)

    golden["eval_id"] = [
        f"eval_{i:03d}"
        for i in range(1, len(golden) + 1)
    ]

    golden["intent"] = ""
    golden["expected_action"] = ""
    golden["difficulty"] = ""

    golden = golden[
        [
            "eval_id",
            "case_id",
            "customer_text",
            "support_text",
            "message_count",
            "intent",
            "expected_action",
            "difficulty",
        ]
    ]

    golden.to_csv(
        GOLD_OUTPUT,
        index=False
    )

    print(f"\nGolden examples created: {len(golden)}")
    print(f"Saved: {GOLD_OUTPUT}")

    con.close()


if __name__ == "__main__":
    main()