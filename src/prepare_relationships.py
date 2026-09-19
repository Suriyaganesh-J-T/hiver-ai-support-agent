from pathlib import Path
import duckdb


DATASET_PATH = Path(
    "/Users/suriyaganeshjt/.cache/kagglehub/datasets/"
    "thoughtvector/customer-support-on-twitter/versions/10/"
    "twcs/twcs.csv"
)

OUTPUT_FILE = Path(
    "data/processed/tweet_relationships.parquet"
)


def main() -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if OUTPUT_FILE.exists():
        print(f"Already exists: {OUTPUT_FILE}")
        return

    print("=" * 70)
    print("BUILDING TWEET RELATIONSHIP INDEX")
    print("=" * 70)

    print("\nSource:")
    print(DATASET_PATH)

    print("\nThis will scan the 493 MB CSV once.")
    print("Please wait...\n")

    con = duckdb.connect()

    query = f"""
        COPY (
            SELECT
                tweet_id,
                author_id,
                inbound,
                in_response_to_tweet_id
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
            )
        )
        TO '{OUTPUT_FILE}'
        (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    con.execute(query)

    print("\nRelationship index created successfully.")
    print(f"Output: {OUTPUT_FILE}")

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{OUTPUT_FILE}')
        """
    ).fetchone()[0]

    print(f"Rows indexed: {count:,}")

    con.close()


if __name__ == "__main__":
    main()