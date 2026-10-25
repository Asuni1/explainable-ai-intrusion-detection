import pandas as pd
import numpy as np

INPUT_FILE = "data/training_dataset.csv"
OUTPUT_FILE = "data/cleaned_dataset.csv"

print("Loading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Original rows: {len(df)}")
print(f"Original columns: {len(df.columns)}")

# --------------------------------------------------
# 1. Clean column names
# --------------------------------------------------

df.columns = df.columns.str.strip()

# --------------------------------------------------
# 2. Find the label column
# --------------------------------------------------

label_column = None

for column in df.columns:
    if column.strip().lower() == "label":
        label_column = column
        break

if label_column is None:
    print("❌ Label column not found.")
    print(df.columns.tolist())
    exit()

print(f"\nLabel column found: {label_column}")

# --------------------------------------------------
# 3. Clean label values
# --------------------------------------------------

df[label_column] = (
    df[label_column]
    .astype(str)
    .str.strip()
)

print("\nOriginal labels:")
print(df[label_column].value_counts())

# --------------------------------------------------
# 4. Convert labels to binary
# --------------------------------------------------

df["target"] = np.where(
    df[label_column].str.upper() == "BENIGN",
    0,
    1
)

print("\nBinary labels:")
print("0 = NORMAL")
print("1 = ATTACK")

print(df["target"].value_counts())

# --------------------------------------------------
# 5. Replace infinity values
# --------------------------------------------------

df = df.replace([np.inf, -np.inf], np.nan)

# --------------------------------------------------
# 6. Remove completely empty columns
# --------------------------------------------------

df = df.dropna(
    axis=1,
    how="all"
)

# --------------------------------------------------
# 7. Remove duplicate rows
# --------------------------------------------------

before_duplicates = len(df)

df = df.drop_duplicates()

removed_duplicates = (
    before_duplicates - len(df)
)

print(
    f"\nDuplicate rows removed: "
    f"{removed_duplicates}"
)

# --------------------------------------------------
# 8. Save cleaned dataset
# --------------------------------------------------

df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n--------------------------------")
print("Cleaning completed successfully!")
print("--------------------------------")

print(f"Final rows: {len(df)}")
print(f"Final columns: {len(df.columns)}")

print(f"\nSaved to:")
print(OUTPUT_FILE)