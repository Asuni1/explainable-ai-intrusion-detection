import numpy as np
import pandas as pd


# ============================================================
# RULE-BASED EXPLANATION
# ============================================================

def generate_explanation(row):
    """Generate a human-readable explanation for the detection."""
    prediction = str(row.get("prediction", "NORMAL")).upper()
    attack_type = str(row.get("attack_type", "None"))

    try:
        confidence = float(row.get("confidence", 0))
    except (TypeError, ValueError):
        confidence = 0.0

    if prediction == "NORMAL":
        return (
            f"The activity was classified as NORMAL with {confidence:.2f}% confidence. "
            "No significant malicious behavior was detected."
        )

    explanations = {
        "Port Scan": "Multiple scanning or probing activities indicate possible reconnaissance against the target system.",
        "Brute Force": "Repeated authentication attempts indicate possible credential brute-force activity.",
        "Unauthorized Access": "The activity indicates an attempt to access a protected resource without authorization.",
        "DoS": "The activity indicates a possible denial-of-service attempt intended to disrupt resource availability.",
        "DDoS": "The activity indicates a possible distributed denial-of-service attack.",
        "SQL Injection": "The activity contains characteristics associated with SQL injection against an application or database.",
        "XSS": "The activity contains characteristics associated with cross-site scripting.",
        "Attack": "The Random Forest model classified the activity as potentially malicious.",
        "Unknown Attack": "The activity was classified as malicious, but the specific attack type could not be determined.",
    }

    reason = explanations.get(attack_type, explanations["Attack"])
    return f"{reason} The detection confidence is {confidence:.2f}%."


# ============================================================
# SHAP EXPLANATION
# ============================================================

def generate_shap_explanation(df, model, row_index=0, top_n=10):
    """Generate SHAP feature contributions for one prediction."""
    if isinstance(model, dict):
        if "model" not in model:
            return [{
                "feature": "SHAP Error",
                "value": 0,
                "contribution": 0,
                "direction": "Saved model package does not contain a 'model' key."
            }]
        model = model["model"]

    try:
        import shap
    except ImportError:
        return [{
            "feature": "SHAP",
            "value": 0,
            "contribution": 0,
            "direction": "Install SHAP using: pip install shap"
        }]

    if df is None or df.empty:
        return [{
            "feature": "SHAP",
            "value": 0,
            "contribution": 0,
            "direction": "No data available for SHAP explanation."
        }]

    if not hasattr(model, "n_features_in_"):
        return [{
            "feature": "SHAP",
            "value": 0,
            "contribution": 0,
            "direction": "Model feature information is unavailable."
        }]

    expected_count = int(model.n_features_in_)
    X = df.copy()

    if hasattr(model, "feature_names_in_"):
        expected_features = list(model.feature_names_in_)
        for feature in expected_features:
            if feature not in X.columns:
                X[feature] = 0.0
        X = X[expected_features].copy()
    else:
        numeric_columns = []
        for column in X.columns:
            converted = pd.to_numeric(X[column], errors="coerce")
            if converted.notna().any():
                numeric_columns.append(column)
                X[column] = converted

        if len(numeric_columns) < expected_count:
            return [{
                "feature": "SHAP",
                "value": 0,
                "contribution": 0,
                "direction": (
                    f"Model expects {expected_count} numerical features, but uploaded log contains only {len(numeric_columns)} compatible features."
                )
            }]

        expected_features = numeric_columns[:expected_count]
        X = X[expected_features].copy()

    for feature in expected_features:
        X[feature] = pd.to_numeric(X[feature], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if X.shape[1] != expected_count:
        return [{
            "feature": "SHAP Error",
            "value": 0,
            "contribution": 0,
            "direction": (
                f"Feature mismatch: model expects {expected_count} features, but {X.shape[1]} were prepared."
            )
        }]

    if len(X) == 0:
        return [{
            "feature": "SHAP",
            "value": 0,
            "contribution": 0,
            "direction": "No data available for SHAP explanation."
        }]

    if row_index < 0 or row_index >= len(X):
        row_index = 0

    X_row = X.iloc[[row_index]]

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_row)
    except Exception as exc:
        return [{
            "feature": "SHAP Error",
            "value": 0,
            "contribution": 0,
            "direction": str(exc)
        }]

    attack_class_index = 1
    try:
        if hasattr(model, "classes_"):
            classes = list(model.classes_)
            if 1 in classes:
                attack_class_index = classes.index(1)
            else:
                attack_class_index = len(classes) - 1
    except Exception:
        attack_class_index = 1

    try:
        if isinstance(shap_values, list):
            if len(shap_values) > attack_class_index:
                values = shap_values[attack_class_index][0]
            else:
                values = shap_values[0][0]
        else:
            values = np.asarray(shap_values)
            if values.ndim == 3:
                if values.shape[2] > attack_class_index:
                    values = values[0, :, attack_class_index]
                else:
                    values = values[0, :, 0]
            elif values.ndim == 2:
                values = values[0]
            elif values.ndim == 1:
                values = values
            else:
                values = values.flatten()
    except Exception as exc:
        return [{
            "feature": "SHAP Error",
            "value": 0,
            "contribution": 0,
            "direction": f"Unable to process SHAP values: {exc}"
        }]

    if len(values) != len(expected_features):
        return [{
            "feature": "SHAP Error",
            "value": 0,
            "contribution": 0,
            "direction": (
                f"SHAP returned {len(values)} values for {len(expected_features)} features."
            )
        }]

    explanation = []
    for feature, value, contribution in zip(expected_features, X_row.iloc[0].values, values):
        try:
            contribution = float(contribution)
        except (TypeError, ValueError):
            contribution = 0.0

        if contribution > 0:
            direction = "Pushes prediction toward ATTACK"
        elif contribution < 0:
            direction = "Pushes prediction toward NORMAL"
        else:
            direction = "No significant contribution"

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            numeric_value = 0.0

        explanation.append({
            "feature": str(feature),
            "value": round(numeric_value, 4),
            "contribution": round(contribution, 6),
            "direction": direction,
        })

    explanation.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return explanation[:top_n]


# ============================================================
# LIME LOCAL EXPLANATION
# ============================================================

def generate_lime_explanation(background_df, model, row_index=0, top_n=10, class_names=None):
    """Generate a LIME explanation for a compatible ML feature matrix."""
    if isinstance(model, dict):
        model = model.get("model")

    if model is None:
        return [{
            "feature": "LIME",
            "value": 0,
            "weight": 0,
            "direction": "Trained model is unavailable."
        }]

    if background_df is None or background_df.empty:
        return [{
            "feature": "LIME",
            "value": 0,
            "weight": 0,
            "direction": "No compatible model features are available."
        }]

    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        return [{
            "feature": "LIME",
            "value": 0,
            "weight": 0,
            "direction": "LIME is not installed. Run: pip install lime"
        }]

    X = background_df.copy()
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        for feature in expected:
            if feature not in X.columns:
                X[feature] = 0.0
        X = X[expected]

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    if X.empty:
        return [{
            "feature": "LIME",
            "value": 0,
            "weight": 0,
            "direction": "No compatible model features are available."
        }]

    row_index = int(row_index)
    if row_index < 0 or row_index >= len(X):
        row_index = 0

    if not hasattr(model, "predict_proba"):
        return [{
            "feature": "LIME",
            "value": 0,
            "weight": 0,
            "direction": "LIME requires a model with predict_proba()."
        }]

    classes = list(getattr(model, "classes_", [0, 1]))
    if class_names is None:
        class_names = [str(c) for c in classes]

    try:
        explainer = LimeTabularExplainer(
            training_data=X.to_numpy(dtype=float),
            feature_names=list(X.columns),
            class_names=class_names,
            mode="classification",
            discretize_continuous=True,
            random_state=42,
        )
        explanation = explainer.explain_instance(
            X.iloc[row_index].to_numpy(dtype=float),
            model.predict_proba,
            labels=tuple(range(len(classes))),
            num_features=min(int(top_n), X.shape[1]),
            num_samples=1000,
        )
        predicted_class = model.predict(X.iloc[[row_index]])[0]
        try:
            label_index = classes.index(predicted_class)
        except ValueError:
            label_index = len(classes) - 1
        weights = explanation.as_list(label=label_index)

        results = []
        for rule, weight in weights:
            results.append({
                "feature": str(rule),
                "value": "Local condition",
                "weight": round(float(weight), 6),
                "direction": (
                    "Supports predicted class" if weight > 0 else
                    "Opposes predicted class" if weight < 0 else
                    "No measurable local influence"
                )
            })
        return results
    except Exception as exc:
        return [{
            "feature": "LIME Error",
            "value": 0,
            "weight": 0,
            "direction": str(exc)
        }]


# ============================================================
# ADD RULE EXPLANATIONS TO DATAFRAME
# ============================================================

def add_rule_explanations(df):
    """Add human-readable explanation to every row."""
    if df is None or df.empty:
        return df

    result = df.copy()
    result["explanation"] = result.apply(generate_explanation, axis=1)
    return result
