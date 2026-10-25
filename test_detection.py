import pandas as pd
from src.detect import detect_intrusion

# Load the 20-row test dataset
df = pd.read_csv("data/test_ml_input.csv")

print("=" * 60)
print("RANDOM FOREST INTRUSION DETECTION TEST")
print("=" * 60)

# Run intrusion detection
results = detect_intrusion(df)

print("\nDetection results:")

# Display the returned results
if isinstance(results, pd.DataFrame):
    columns_to_show = [
        col for col in [
            "Label",
            "prediction",
            "predicted_label",
            "confidence",
            "detection_method",
            "attack_type",
        ]
        if col in results.columns
    ]

    if columns_to_show:
        print(results[columns_to_show].to_string(index=False))
    else:
        print(results.head(20).to_string(index=False))

elif isinstance(results, tuple):
    for item in results:
        print(item)

else:
    print(results)

print("\nTest completed.")