
# =========================================================
# AUTOMATED RESPONSE RECOMMENDATION ENGINE
# Recommends actions; does not execute cloud changes.
# =========================================================

import pandas as pd


def _clean_text(value, default=""):
    if value is None:
        return default

    try:
        if pd.isna(value):
            return default
    except (TypeError, ValueError):
        pass

    return str(value).strip()


def get_response(row):
    prediction = _clean_text(
        row.get("prediction", "NORMAL")
    ).upper()

    attack_type = _clean_text(
        row.get("attack_type", "Unknown Attack"),
        "Unknown Attack"
    ).lower()

    risk_level = _clean_text(
        row.get("risk_level", "Low"),
        "Low"
    ).capitalize()

    # NORMAL ACTIVITY
    if prediction != "ATTACK":
        return "No Action Required"

    # CRITICAL RISK
    if risk_level == "Critical":
        return "Block Source and Isolate Resource (Recommended)"

    # HIGH RISK
    if risk_level == "High":
        responses = {
            "port scan": "Block Source IP and Review Network Rules",
            "brute force": "Block Source and Review Account Security",
            "sql injection": "Block Malicious Request and Review Application",
            "xss": "Block Malicious Request and Review Application",
            "dos": "Rate Limit Traffic and Investigate Source",
            "ddos": "Enable DDoS Mitigation and Rate Limit Traffic",
            "unauthorized access": "Revoke Unauthorized Access and Review Permissions",
            "malware": "Isolate Affected Resource and Investigate",
            "unknown attack": "Restrict Suspicious Activity and Investigate",
        }

        return responses.get(
            attack_type,
            "Restrict Suspicious Activity and Investigate"
        )

    # MEDIUM RISK
    if risk_level == "Medium":
        return "Generate Alert and Monitor Source"

    # LOW RISK
    return "Generate Security Alert and Review Event"


# =========================================================
# APPLY RECOMMENDATIONS TO DATAFRAME
# =========================================================

def apply_response(df):
    if df is None or df.empty:
        return df

    result = df.copy()

    result["automated_response"] = result.apply(
        get_response,
        axis=1
    )

    result["response_status"] = result[
        "automated_response"
    ].apply(
        lambda action: (
            "No Action Required"
            if action == "No Action Required"
            else "Recommendation Only - Not Executed"
        )
    )

    return result
