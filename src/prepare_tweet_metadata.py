from pathlib import Path
import duckdb


DATASET_PATH = Path(
    "/Users/suriyaganeshjt/.cache/kagglehub/datasets/"
    "thoughtvector/customer-support-on-twitter/versions/10/"
    "twcs/twcs.csv"
)

OUTPUT_FILE = Path(
    "data/processed/tweet_metadata.parquet"
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
    print("BUILDING TWEET METADATA INDEX")
    print("=" * 70)

    print("\nScanning the CSV once to create a reusable metadata index...")
    print("Please wait...\n")

    con = duckdb.connect()

    query = f"""
        COPY (
            SELECT
                tweet_id,
                author_id,
                inbound,
                strptime(
                    created_at,
                    '%a %b %d %H:%M:%S %z %Y'
                ) AS created_at,
                text
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
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
    """

    con.execute(query)

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{OUTPUT_FILE}')
        """
    ).fetchone()[0]

    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print(f"Rows indexed: {count:,}")
    print(f"Output: {OUTPUT_FILE}")

    con.close()


if __name__ == "__main__":
    main()