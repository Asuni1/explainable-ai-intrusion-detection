import pandas as pd

INPUT_FILE = "data/training_dataset.csv"

print("Loading dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("\nDataset loaded successfully!")
print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumn names:")
print(df.columns.tolist())

print("\nChecking Label column...")

if "Label" in df.columns:
    print("\nLabel distribution:")
    print(df["Label"].value_counts())
else:
    print("\n❌ Label column not found!")

print("\nMissing values:")
print(df.isnull().sum().sort_values(ascending=False).head(15))