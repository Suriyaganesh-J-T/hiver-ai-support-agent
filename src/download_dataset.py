from pathlib import Path
import kagglehub


DATASET = "thoughtvector/customer-support-on-twitter"
OUTPUT_DIR = Path("data/raw")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading Customer Support on Twitter dataset...")
    dataset_path = kagglehub.dataset_download(DATASET)

    print(f"\nKaggleHub downloaded the dataset to:")
    print(dataset_path)

    print("\nFiles found:")
    for path in Path(dataset_path).rglob("*"):
        if path.is_file():
            print(f" - {path}")


if __name__ == "__main__":
    main()