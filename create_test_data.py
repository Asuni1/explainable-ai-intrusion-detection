
import pandas as pd
from pathlib import Path

# Project paths
project_dir = Path(__file__).resolve().parent
dataset_path = project_dir / "data" / "cleaned_dataset.csv"
output_path = project_dir / "data" / "test_ml_input.csv"

# Load the existing cleaned dataset
print("Loading cleaned dataset...")
df = pd.read_csv(dataset_path, nrows=100)

# Remove the target column to prevent label leakage
columns_to_remove = [
    col for col in df.columns
    if col.strip().lower() in ["target", "label", "attack_type"]
]

df = df.drop(columns=columns_to_remove, errors="ignore")

# Keep a small sample for testing
df.head(20).to_csv(output_path, index=False)

print("Test dataset created:", output_path)
print("Rows:", min(20, len(df)))
print("Columns:", len(df.columns))
print("Column names:")
print(df.columns.tolist())
