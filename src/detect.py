import os
import joblib
import pandas as pd

from .feature_adapter import (
    canonical_name,
    convert_to_model_features,
)

MODEL_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "models", "intrusion_model.pkl")
)


def load_model():
    """Load either a packaged model {'model': ..., 'features': [...]} or a raw model."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Trained model not found at: {MODEL_PATH}")

    package = joblib.load(MODEL_PATH)

    if isinstance(package, dict):
        model = package.get("model")
        features = package.get("features") or list(
            getattr(model, "feature_names_in_", [])
        )
    else:
        model = package
        features = list(getattr(model, "feature_names_in_", []))

    if model is None:
        raise ValueError("The saved model package does not contain a model.")
    if not features:
        raise ValueError(
            "The model's feature names are unavailable. "
            "Check the package created by src/train_model.py."
        )
    if not hasattr(model, "predict"):
        raise TypeError("The loaded object does not provide predict().")

    return model, list(features)


def is_ml_compatible(df, model, features):
    """
    Only route records to ML when every trained feature is present and numeric.
    This avoids silently filling missing network features with zeros.
    """
    if df is None or df.empty or not features:
        return False

    column_lookup = {}
    for col in df.columns:
        column_lookup.setdefault(canonical_name(col), col)

    matched = []
    missing = []
    non_numeric = []

    for feature in features:
        source_col = column_lookup.get(canonical_name(feature))
        if source_col is None:
            missing.append(feature)
            continue

        values = pd.to_numeric(df[source_col], errors="coerce")
        # Treat a column as usable if it has at least one numeric value.
        if values.notna().any():
            matched.append(feature)
        else:
            non_numeric.append(feature)

    print("=" * 45)
    print("ML COMPATIBILITY CHECK")
    print("Expected features:", len(features))
    print("Usable numeric features:", len(matched))
    print("Missing features:", len(missing))
    print("Non-numeric features:", len(non_numeric))
    print("ML compatible:", not missing and not non_numeric)
    print("=" * 45)

    # Require all trained features. Do not infer missing network-flow values.
    return not missing and not non_numeric


def detect_with_ml(df, model, features):
    result = df.copy()
    X = convert_to_model_features(result, model, features, strict=True)

    predictions = model.predict(X)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)
        classes = list(getattr(model, "classes_", []))
        attack_class = 1 if 1 in classes else (classes[-1] if classes else None)

        confidence = probabilities.max(axis=1) * 100.0
        if attack_class is not None and attack_class in classes:
            attack_idx = classes.index(attack_class)
            attack_probabilities = probabilities[:, attack_idx]
        else:
            attack_probabilities = confidence / 100.0
    else:
        classes = list(getattr(model, "classes_", []))
        attack_class = 1 if 1 in classes else (classes[-1] if classes else None)
        confidence = [100.0] * len(predictions)
        attack_probabilities = [None] * len(predictions)

    readable = [
        "ATTACK" if (prediction == attack_class) else "NORMAL"
        for prediction in predictions
    ]

    result["prediction"] = readable
    result["confidence"] = pd.Series(confidence, index=result.index).round(2)
    if "attack_type" not in result.columns:
        result["attack_type"] = "None"

    # A binary classifier generally cannot identify the specific attack family.
    attack_mask = result["prediction"].eq("ATTACK")
    empty_type = result["attack_type"].astype(str).str.strip().str.lower().isin(
        ["", "none", "nan", "null"]
    )
    result.loc[attack_mask & empty_type, "attack_type"] = "Unknown Attack"
    result.loc[~attack_mask, "attack_type"] = "None"
    result["detection_method"] = "Random Forest ML"

    print("Random Forest detection completed.")
    print("Records:", len(result))
    print("Normal:", int(result["prediction"].eq("NORMAL").sum()))
    print("Attack:", int(result["prediction"].eq("ATTACK").sum()))
    return result


def detect_custom_logs(df):
    """Rule-based fallback for ordinary logs; this is not an ML prediction."""
    result = df.copy()
    predictions, attack_types, confidences = [], [], []

    # Prioritize specific patterns before generic 'attack' matches.
    rules = [
        (r"\b(port[\s_-]?scan|port scanning)\b", "Port Scan", 98.0),
        (r"\b(brute[\s_-]?force)\b", "Brute Force", 98.0),
        (r"\b(unauthorized[\s_-]?(access|login)?|access denied)\b",
         "Unauthorized Access", 98.0),
        (r"\bddos\b", "DDoS", 99.0),
        (r"\bdos\b", "DoS", 97.0),
        (r"\b(sql[\s_-]?injection)\b", "SQL Injection", 98.0),
        (r"\bxss\b", "XSS", 98.0),
        (r"\b(suspicious|malicious|attack|intrusion)\b", "Unknown Attack", 95.0),
    ]

    for _, row in result.iterrows():
        parts = []
        for value in row.tolist():
            if value is None:
                continue
            try:
                if pd.isna(value):
                    continue
            except (TypeError, ValueError):
                pass
            if isinstance(value, (list, tuple, dict)):
                parts.append(str(value))
            else:
                parts.append(str(value))

        text = " ".join(parts).lower()
        found = next(
            ((name, conf) for pattern, name, conf in rules
             if __import__("re").search(pattern, text)),
            None,
        )

        if found:
            attack_type, conf = found
            prediction = "ATTACK"
        else:
            prediction, attack_type, conf = "NORMAL", "None", 95.0

        predictions.append(prediction)
        attack_types.append(attack_type)
        confidences.append(conf)

    result["prediction"] = predictions
    result["attack_type"] = attack_types
    result["confidence"] = confidences
    result["detection_method"] = "Custom Security Log Detection"
    return result


def detect_intrusion(df):
    if df is None:
        return pd.DataFrame()
    if df.empty:
        return df.copy()

    try:
        model, features = load_model()
        if is_ml_compatible(df, model, features):
            print("Using Random Forest ML detection.")
            return detect_with_ml(df, model, features)

        print(
            "Input does not contain all numeric model features. "
            "Using custom security-log detection instead."
        )
    except Exception as exc:
        print(f"ML detection unavailable: {exc}")
        print("Using custom security-log detection instead.")

    return detect_custom_logs(df)
