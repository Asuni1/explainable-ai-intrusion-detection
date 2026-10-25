import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

INPUT_FILE = "data/cleaned_dataset.csv"
MODEL_FILE = "models/intrusion_model.pkl"

print("Loading cleaned dataset...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print("Rows:", len(df))
print("Columns:", len(df.columns))

# --------------------------------------------------
# 1. Separate target
# --------------------------------------------------

if "target" not in df.columns:
    print("❌ Target column not found!")
    exit()

y = df["target"]

# --------------------------------------------------
# 2. Select numerical features
# --------------------------------------------------

X = df.select_dtypes(
    include=["number"]
).copy()

# Remove target from features
if "target" in X.columns:
    X = X.drop(columns=["target"])

print("\nNumerical features selected:", len(X.columns))

# --------------------------------------------------
# 3. Replace infinity values
# --------------------------------------------------

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

# Replace missing numerical values
X = X.fillna(0)

# --------------------------------------------------
# 4. Make sure feature names are strings
# --------------------------------------------------

X.columns = X.columns.astype(str)

# --------------------------------------------------
# 5. Train/Test split
# --------------------------------------------------

print("\nSplitting dataset...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))

# --------------------------------------------------
# 6. Train Random Forest
# --------------------------------------------------

print("\nTraining Random Forest...")

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

model.fit(
    X_train,
    y_train
)

print("✅ Model training completed!")

# --------------------------------------------------
# 7. Predictions
# --------------------------------------------------

print("\nMaking predictions...")

y_pred = model.predict(X_test)

# --------------------------------------------------
# 8. Evaluation
# --------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

print("\n================================")
print("MODEL PERFORMANCE")
print("================================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["NORMAL", "ATTACK"],
        zero_division=0
    )
)

# --------------------------------------------------
# 9. Save model + feature names
# --------------------------------------------------

model_package = {
    "model": model,
    "features": list(X.columns)
}

joblib.dump(
    model_package,
    MODEL_FILE
)

print("\n================================")
print("MODEL SAVED")
print("================================")

print(
    f"Saved model to: {MODEL_FILE}"
)