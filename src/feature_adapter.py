import re
import numpy as np
import pandas as pd


def canonical_name(name):
    """Normalize names so spaces, underscores and punctuation do not matter."""
    return re.sub(r"[^a-z0-9]", "", str(name).strip().lower())


def convert_to_model_features(df, model, features=None, strict=False):
    """
    Align a dataframe to the exact model feature names and order.

    strict=True refuses to fabricate missing features. Use strict mode for
    predictions and explanations. Non-strict mode is retained only for
    backwards compatibility; it fills absent values with zero.
    """
    if df is None or df.empty:
        raise ValueError("No log data available.")

    if features is None:
        features = list(getattr(model, "feature_names_in_", []))
        if not features:
            raise ValueError("Model feature information is unavailable.")

    expected = list(features)
    lookup = {}
    for column in df.columns:
        lookup.setdefault(canonical_name(column), column)

    result = pd.DataFrame(index=df.index)
    missing = []
    unusable = []

    for feature in expected:
        source = feature if feature in df.columns else lookup.get(canonical_name(feature))
        if source is None:
            missing.append(feature)
            continue

        values = pd.to_numeric(df[source], errors="coerce")
        if values.notna().sum() == 0:
            unusable.append(feature)
            continue
        result[feature] = values

    if strict and (missing or unusable):
        details = []
        if missing:
            details.append(f"missing features: {missing[:10]}")
        if unusable:
            details.append(f"non-numeric features: {unusable[:10]}")
        raise ValueError(
            "Input cannot be safely converted to the trained model's features; "
            + "; ".join(details)
        )

    for feature in expected:
        if feature not in result.columns:
            result[feature] = 0.0

    result = result[expected].replace([np.inf, -np.inf], np.nan)

    if strict and result.isna().any().any():
        bad = result.columns[result.isna().any()].tolist()
        raise ValueError(
            "Numeric model features contain missing/invalid values. "
            f"Clean or impute them using the training pipeline first. "
            f"Examples: {bad[:10]}"
        )

    # Backwards-compatible non-strict behaviour.
    if not strict:
        result = result.fillna(0.0)

    expected_count = getattr(model, "n_features_in_", len(expected))
    if result.shape[1] != expected_count:
        raise ValueError(
            f"Feature-count mismatch: adapter produced {result.shape[1]} "
            f"features, model expects {expected_count}."
        )

    print("========== MODEL FEATURE ADAPTER ==========")
    print("Expected features:", len(expected))
    print("Final feature count:", result.shape[1])
    print("Strict mode:", strict)
    print("===========================================")
    return result
