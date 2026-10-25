import pandas as pd
import numpy as np


# ============================================================
# ATTACK SEVERITY
# ============================================================

ATTACK_SEVERITY = {
    "Port Scan": 5,
    "Brute Force": 7,
    "Unauthorized Access": 9,
    "DoS": 8,
    "DDoS": 10,
    "SQL Injection": 9,
    "XSS": 8,
    "Attack": 6,
    "Unknown Attack": 6
}


# ============================================================
# SAFE VALUE -> TEXT CONVERSION
# ============================================================

def _safe_value_to_text(value):
    """
    Convert a DataFrame cell to text safely.

    Handles:
    - strings
    - numbers
    - None / NaN / pd.NA
    - lists
    - tuples
    - NumPy arrays

    This prevents:
    ValueError: The truth value of an array with more than one
    element is ambiguous.
    """

    if value is None:
        return ""

    # Handle lists, tuples, and NumPy arrays recursively.
    if isinstance(value, (list, tuple, np.ndarray)):
        try:
            values = np.asarray(value, dtype=object).reshape(-1)
            return " ".join(
                _safe_value_to_text(item)
                for item in values
            )
        except Exception:
            return str(value)

    # pd.isna() is safe here because value is scalar.
    if pd.api.types.is_scalar(value):
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass

    return str(value)


# ============================================================
# RESOURCE CRITICALITY
# ============================================================

def get_resource_criticality(row):
    """
    Determine how important the affected resource is.
    """

    text = " ".join(
        _safe_value_to_text(value)
        for value in row.values
    ).lower()

    # Highest priority resources first
    if "database" in text:
        return 10

    if "admin" in text:
        return 9

    if "server" in text:
        return 8

    if "cloud" in text:
        return 8

    if "application" in text:
        return 7

    return 5


# ============================================================
# ATTACK SEVERITY
# ============================================================

def get_attack_severity(attack_type):
    """
    Return attack severity on a 1-10 scale.
    """

    attack_type = str(
        attack_type
    ).strip()

    return ATTACK_SEVERITY.get(
        attack_type,
        6
    )


# ============================================================
# RISK CALCULATION
# ============================================================

def calculate_risk(row):
    """
    Risk formula:

    R =
    Prediction Confidence
    × Attack Severity
    × Resource Criticality

    All three components are normalized to 0-1,
    then converted to a 0-100 score.
    """

    prediction = str(
        row.get(
            "prediction",
            "NORMAL"
        )
    ).upper()

    # Normal activity has no security risk
    if prediction != "ATTACK":
        return 0.0, "Low"

    # --------------------------------------------------------
    # Prediction confidence
    # --------------------------------------------------------

    try:
        confidence = float(
            row.get(
                "confidence",
                0
            )
        )
    except (ValueError, TypeError):
        confidence = 0

    prediction_confidence = (
        max(
            0,
            min(
                confidence,
                100
            )
        )
        / 100
    )

    # --------------------------------------------------------
    # Attack severity
    # --------------------------------------------------------

    attack_type = str(
        row.get(
            "attack_type",
            "Attack"
        )
    )

    severity = get_attack_severity(
        attack_type
    )

    attack_severity = (
        severity / 10
    )

    # --------------------------------------------------------
    # Resource criticality
    # --------------------------------------------------------

    criticality = (
        get_resource_criticality(
            row
        )
    )

    resource_criticality = (
        criticality / 10
    )

    # --------------------------------------------------------
    # Paper-based risk formula
    # --------------------------------------------------------

    risk_score = (
        prediction_confidence
        * attack_severity
        * resource_criticality
        * 100
    )

    risk_score = round(
        risk_score,
        2
    )

    # --------------------------------------------------------
    # Risk level
    # --------------------------------------------------------

    if risk_score >= 80:
        risk_level = "Critical"

    elif risk_score >= 60:
        risk_level = "High"

    elif risk_score >= 30:
        risk_level = "Medium"

    else:
        risk_level = "Low"

    return (
        risk_score,
        risk_level
    )


# ============================================================
# APPLY RISK ASSESSMENT
# ============================================================

def assess_risk(df):
    """
    Calculate risk for every detected log.
    """

    if df is None or df.empty:
        return df

    result = df.copy()

    risk_scores = []
    risk_levels = []
    severities = []
    criticalities = []

    for _, row in result.iterrows():

        score, level = calculate_risk(
            row
        )

        attack_type = str(
            row.get(
                "attack_type",
                "Attack"
            )
        )

        severity = get_attack_severity(
            attack_type
        )

        criticality = get_resource_criticality(
            row
        )

        risk_scores.append(
            score
        )

        risk_levels.append(
            level
        )

        severities.append(
            severity
        )

        criticalities.append(
            criticality
        )

    result["attack_severity"] = (
        severities
    )

    result["resource_criticality"] = (
        criticalities
    )

    result["risk_score"] = (
        risk_scores
    )

    result["risk_level"] = (
        risk_levels
    )

    return result
