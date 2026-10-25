import pandas as pd
import glob
import os


RAW_FOLDER = "data/raw"
OUTPUT_FILE = "data/training_dataset.csv"


def main():

    files = glob.glob(
        os.path.join(RAW_FOLDER, "*.csv")
    )

    if not files:
        print("❌ No CSV files found in data/raw/")
        return

    print(f"Found {len(files)} CSV files.")

    dataframes = []

    for file in files:

        print(f"Reading: {os.path.basename(file)}")

        try:
            df = pd.read_csv(
                file,
                encoding="utf-8",
                low_memory=False
            )

            # Remove spaces from column names
            df.columns = (
                df.columns
                .str.strip()
            )

            dataframes.append(df)

            print(
                f"   Rows loaded: {len(df)}"
            )

        except Exception as e:

            print(
                f"   ⚠️ Could not read file: {e}"
            )

    if not dataframes:
        print("❌ No datasets could be loaded.")
        return

    print("\nCombining datasets...")

    combined = pd.concat(
        dataframes,
        ignore_index=True
    )

    print(
        f"Total rows: {len(combined)}"
    )

    print(
        f"Total columns: {len(combined.columns)}"
    )

    # Remove completely empty columns
    combined = combined.dropna(
        axis=1,
        how="all"
    )

    # Remove duplicate rows
    combined = combined.drop_duplicates()

    # Save
    combined.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(
        f"\n✅ Dataset saved to: {OUTPUT_FILE}"
    )

    print(
        f"Final rows: {len(combined)}"
    )

    print(
        f"Final columns: {len(combined.columns)}"
    )


if __name__ == "__main__":
    main()