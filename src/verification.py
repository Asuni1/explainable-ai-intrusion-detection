
import pandas as pd


def verify_response(row):
    prediction = str(
        row.get("prediction", "NORMAL")
    ).strip().upper()

    response = str(
        row.get("automated_response", "No Action Required")
    ).strip()

    response_status = str(
        row.get("response_status", "")
    ).strip()

    # Normal activity requires no response.
    if prediction != "ATTACK":
        return "Not Required"

    # The system has only recommended a response.
    if (
        response_status == "Recommendation Only - Not Executed"
        or response == "No Action Required"
    ):
        return "Recommendation Only - Not Executed"

    # No evidence of actual execution and verification exists.
    return "Not Verified"


def apply_verification(df):
    if df is None or df.empty:
        return df

    result = df.copy()

    result["verification_status"] = result.apply(
        verify_response,
        axis=1
    )

    return result
