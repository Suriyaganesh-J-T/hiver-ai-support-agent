from pathlib import Path
import duckdb
import pandas as pd


DATASET_PATH = Path(
    "/Users/suriyaganeshjt/.cache/kagglehub/datasets/"
    "thoughtvector/customer-support-on-twitter/versions/10/"
    "twcs/twcs.csv"
)

BRAND = "AppleSupport"
OUTPUT_FILE = Path(
    "data/processed/apple_support_messages.csv"
)


def main() -> None:
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    con = duckdb.connect()

    print(f"Extracting messages for {BRAND}...")

    query = f"""
        SELECT
            tweet_id,
            author_id,
            inbound,
            created_at,
            text,
            response_tweet_id,
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
        WHERE author_id = '{BRAND}'
        OR tweet_id IN (
            SELECT DISTINCT in_response_to_tweet_id
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
            WHERE author_id = '{BRAND}'
              AND in_response_to_tweet_id IS NOT NULL
        )
        OR EXISTS (
            SELECT 1
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
            ) parent
            WHERE parent.author_id = '{BRAND}'
              AND (
                  ',' || parent.response_tweet_id || ','
              ) LIKE ('%,' || tweet_id || ',%')
        )
    """

    df = con.execute(query).df()

    print(f"Extracted {len(df):,} messages.")

    df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved to: {OUTPUT_FILE}")

    con.close()


if __name__ == "__main__":
    main()