from pathlib import Path
import duckdb
from datetime import datetime


DATASET_PATH = Path(
    "/Users/suriyaganeshjt/.cache/kagglehub/datasets/"
    "thoughtvector/customer-support-on-twitter/versions/10/"
    "twcs/twcs.csv"
)

RELATIONSHIP_FILE = Path(
    "data/processed/tweet_relationships.parquet"
)

OUTPUT_FILE = Path(
    "data/processed/sample_conversations.txt"
)

BRANDS = [
    "AmazonHelp",
    "AppleSupport",
    "Uber_Support",
    "AmericanAir",
    "VirginTrains",
]

SAMPLES_PER_BRAND = 10
MAX_DEPTH = 6


def main() -> None:
    if not RELATIONSHIP_FILE.exists():
        raise FileNotFoundError(
            f"Relationship index not found: {RELATIONSHIP_FILE}\n"
            "Run:\n"
            "python src/prepare_relationships.py"
        )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    con = duckdb.connect()

    print("=" * 70)
    print("SAMPLING REAL SUPPORT CONVERSATIONS")
    print("=" * 70)

    print("\nUsing:")
    print(RELATIONSHIP_FILE)

    # ------------------------------------------------------------
    # Create a view over the relationship index.
    # ------------------------------------------------------------

    con.execute(
        f"""
        CREATE OR REPLACE VIEW rel AS
        SELECT *
        FROM read_parquet('{RELATIONSHIP_FILE}');
        """
    )

    # ------------------------------------------------------------
    # Build bidirectional edges.
    #
    # If:
    #
    # customer tweet B
    #       ↓
    # in_response_to_tweet_id = A
    #
    # then A and B belong to the same conversation chain.
    # ------------------------------------------------------------

    con.execute(
        """
        CREATE OR REPLACE TEMP VIEW edges AS

        SELECT
            tweet_id AS src,
            in_response_to_tweet_id AS dst
        FROM rel
        WHERE in_response_to_tweet_id IS NOT NULL

        UNION

        SELECT
            in_response_to_tweet_id AS src,
            tweet_id AS dst
        FROM rel
        WHERE in_response_to_tweet_id IS NOT NULL;
        """
    )

    # ------------------------------------------------------------
    # Select deterministic sample seeds.
    #
    # We use a hash instead of random sampling so the experiment
    # is reproducible.
    # ------------------------------------------------------------

    seed_rows = []

    for brand in BRANDS:
        rows = con.execute(
            f"""
            SELECT tweet_id
            FROM rel
            WHERE author_id = '{brand}'
              AND inbound = FALSE
            ORDER BY hash(tweet_id)
            LIMIT {SAMPLES_PER_BRAND};
            """
        ).fetchall()

        print(
            f"{brand}: selected {len(rows)} support tweets"
        )

        for (tweet_id,) in rows:
            seed_rows.append(
                (brand, tweet_id)
            )

    # ------------------------------------------------------------
    # Traverse each seed's local conversation graph.
    # ------------------------------------------------------------

    conversations = []
    all_tweet_ids = set()

    for brand, seed_id in seed_rows:

        query = f"""
            WITH RECURSIVE walk(
                tweet_id,
                depth,
                path
            ) AS (

                SELECT
                    '{seed_id}'::VARCHAR,
                    0,
                    ['{seed_id}']

                UNION ALL

                SELECT
                    e.dst,
                    w.depth + 1,
                    list_append(
                        w.path,
                        e.dst
                    )
                FROM walk w
                JOIN edges e
                    ON e.src = w.tweet_id
                WHERE
                    w.depth < {MAX_DEPTH}
                    AND NOT list_contains(
                        w.path,
                        e.dst
                    )
            )

            SELECT DISTINCT tweet_id
            FROM walk
        """

        rows = con.execute(query).fetchall()

        tweet_ids = [
            row[0]
            for row in rows
        ]

        all_tweet_ids.update(tweet_ids)

        conversations.append(
            {
                "brand": brand,
                "seed_id": seed_id,
                "tweet_ids": tweet_ids,
            }
        )

    print(
        f"\nUnique tweets collected: "
        f"{len(all_tweet_ids):,}"
    )

    # ------------------------------------------------------------
    # Fetch text and metadata from the original CSV ONCE.
    #
    # We only retrieve the tweets that belong to our sampled
    # conversations.
    # ------------------------------------------------------------

    print("\nFetching conversation text...")

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE sample_ids (
            tweet_id VARCHAR
        );
        """
    )

    con.executemany(
        "INSERT INTO sample_ids VALUES (?)",
        [(tweet_id,) for tweet_id in all_tweet_ids]
    )

    sampled_data = con.execute(
        f"""
        SELECT
            t.tweet_id,
            t.author_id,
            t.inbound,
            t.created_at,
            t.text,
            t.in_response_to_tweet_id
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
        ) t
        INNER JOIN sample_ids s
            ON t.tweet_id = s.tweet_id
        """
    ).df()

    # ------------------------------------------------------------
    # Create quick lookup.
    # ------------------------------------------------------------

    tweet_lookup = {
        str(row["tweet_id"]): row
        for _, row in sampled_data.iterrows()
    }

    # ------------------------------------------------------------
    # Write human-readable conversations.
    # ------------------------------------------------------------

    output = []

    for item in conversations:

        brand = item["brand"]
        seed_id = item["seed_id"]
        tweet_ids = item["tweet_ids"]

        conversation_rows = [
            tweet_lookup[tweet_id]
            for tweet_id in tweet_ids
            if tweet_id in tweet_lookup
        ]

        conversation_rows.sort(
            key=lambda row: datetime.strptime(
                row["created_at"],
                "%a %b %d %H:%M:%S %z %Y"
            )
        )

        if not conversation_rows:
            continue

        output.append("=" * 80)
        output.append(f"BRAND: {brand}")
        output.append(f"SEED TWEET: {seed_id}")
        output.append("=" * 80)

        for row in conversation_rows:

            if bool(row["inbound"]):
                speaker = "CUSTOMER"
            else:
                speaker = brand

            output.append(
                f"\n[{row['created_at']}] {speaker}:"
            )

            output.append(
                f"  {str(row['text']).replace(chr(10), ' ')}"
            )

        output.append("\n")

    OUTPUT_FILE.write_text(
        "\n".join(output),
        encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Saved conversations to:")
    print(OUTPUT_FILE)

    con.close()


if __name__ == "__main__":
    main()