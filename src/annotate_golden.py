from pathlib import Path

import pandas as pd


FILE = Path("data/golden/golden_150.csv")

INTENTS = {
    "1": "delivery_tracking",
    "2": "order_product_issue",
    "3": "return_refund",
    "4": "payment_billing",
    "5": "account_access",
    "6": "prime_subscription",
    "7": "technical_device_digital",
    "8": "product_info_availability",
    "9": "complaint_service",
    "10": "other",
}


def main():
    df = pd.read_csv(FILE)

    df["intent"] = df["intent"].fillna("")
    df["expected_action"] = df["expected_action"].fillna("")
    df["difficulty"] = df["difficulty"].fillna("")

    # Resume from the first unlabeled example.
    remaining = df.index[
        df["intent"].str.strip() == ""
    ].tolist()

    print("=" * 70)
    print("GOLDEN SET ANNOTATION")
    print("=" * 70)

    print("\nIntent codes:")

    for key, value in INTENTS.items():
        print(f"{key:>2} = {value}")

    print("\nAction:")
    print("A = AUTO_HANDLE")
    print("E = ESCALATE")

    print("\nDifficulty:")
    print("1 = easy")
    print("2 = medium")
    print("3 = hard")

    print(
        f"\nRemaining examples: {len(remaining)}"
    )

    for position, idx in enumerate(remaining, start=1):

        print("\n" + "=" * 70)
        print(
            f"Example {position}/{len(remaining)}"
        )
        print("=" * 70)

        print("\nCUSTOMER MESSAGE:\n")
        print(df.loc[idx, "customer_text"])

        while True:
            intent_code = input(
                "\nIntent (1-10): "
            ).strip()

            if intent_code in INTENTS:
                break

            print("Invalid intent code.")

        while True:
            action = input(
                "Action (A/E): "
            ).strip().upper()

            if action in {"A", "E"}:
                break

            print("Enter A or E.")

        while True:
            difficulty = input(
                "Difficulty (1/2/3): "
            ).strip()

            if difficulty in {"1", "2", "3"}:
                break

            print("Enter 1, 2, or 3.")

        action_value = (
            "AUTO_HANDLE"
            if action == "A"
            else "ESCALATE"
        )

        difficulty_value = {
            "1": "easy",
            "2": "medium",
            "3": "hard",
        }[difficulty]

        df.loc[idx, "intent"] = INTENTS[
            intent_code
        ]

        df.loc[idx, "expected_action"] = action_value

        df.loc[idx, "difficulty"] = difficulty_value

        # Save after EVERY example.
        df.to_csv(FILE, index=False)

        print("\nSaved.")

    print("\n" + "=" * 70)
    print("ANNOTATION COMPLETE")
    print("=" * 70)

    print(
        df["intent"].value_counts()
    )

    print(
        "\nAction distribution:"
    )

    print(
        df["expected_action"].value_counts()
    )


if __name__ == "__main__":
    main()