import streamlit as st
import pandas as pd
import os
import sys
import joblib


# =========================================================
# PROJECT PATH
# =========================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# =========================================================
# IMPORT PROJECT MODULES
# =========================================================

from src.log_parser import parse_multiple_files
from src.detect import detect_intrusion
from src.risk import assess_risk
from src.response import apply_response
from src.verification import apply_verification
from src.explain import (
    generate_explanation,
    generate_shap_explanation,
    generate_lime_explanation,
)
from src.feature_adapter import convert_to_model_features


# =========================================================
# MODEL PATH
# =========================================================

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "intrusion_model.pkl"
)


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Intrusion Detection System",
    page_icon="🛡️",
    layout="wide"
)


# =========================================================
# HEADER
# =========================================================

st.title(
    "🛡️ Explainable AI-Based Intrusion Detection System"
)

st.markdown(
    """
    **Multi-Cloud Security Monitoring | AI Detection | Explainable AI | Risk Assessment | Automated Response**
    """
)

st.divider()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🛡️ Security System")

st.sidebar.subheader("System Status")

if os.path.exists(MODEL_PATH):

    st.sidebar.success(
        "🟢 Random Forest Model Loaded"
    )

else:

    st.sidebar.error(
        "🔴 Random Forest Model Not Found"
    )


st.sidebar.markdown("---")

st.sidebar.subheader("Supported Logs")

st.sidebar.write("📄 CSV")
st.sidebar.write("📄 TXT")
st.sidebar.write("📄 JSON")
st.sidebar.write("📄 LOG")

st.sidebar.markdown("---")

st.sidebar.info(
    "Upload multiple security log files "
    "for centralized analysis."
)


# =========================================================
# FILE UPLOAD
# =========================================================

st.header("📂 Security Log Upload")

uploaded_files = st.file_uploader(
    "Select one or more security log files",
    type=[
        "csv",
        "txt",
        "json",
        "log"
    ],
    accept_multiple_files=True
)


# =========================================================
# NO FILE
# =========================================================

if not uploaded_files:

    st.info(
        "👆 Upload one or more security logs to begin analysis."
    )

    st.markdown(
        """
        ### System Workflow

        ```
        Log Upload
             ↓
        Log Parsing
             ↓
        Normalization
             ↓
        AI Intrusion Detection
             ↓
        SHAP + LIME Explanation
             ↓
        Risk Assessment
             ↓
        Automated Response
             ↓
        Response Verification
             ↓
        Security Audit
        ```
        """
    )

    st.stop()


# =========================================================
# FILE INFORMATION
# =========================================================

st.success(
    f"✅ {len(uploaded_files)} file(s) uploaded."
)

file_data = []

for file in uploaded_files:

    file_data.append(
        {
            "File Name": file.name,
            "File Type": file.type,
            "Size (KB)": round(
                file.size / 1024,
                2
            )
        }
    )

file_df = pd.DataFrame(file_data)

st.subheader("📁 Uploaded Files")

st.dataframe(
    file_df,
    width="stretch",
    hide_index=True
)


# =========================================================
# PARSE LOGS
# =========================================================

with st.spinner(
    "🔄 Parsing and normalizing log files..."
):

    df = parse_multiple_files(
        uploaded_files
    )


if df.empty:

    st.error(
        "❌ No valid log records could be extracted."
    )

    st.stop()


st.success(
    f"✅ {len(df)} log records extracted."
)


# =========================================================
# RAW LOG DATA
# =========================================================

with st.expander(
    "📄 View Parsed Log Data"
):

    st.dataframe(
        df,
        width="stretch",
        hide_index=True
    )


# =========================================================
# DETECTION
# =========================================================

with st.spinner(
    "🔍 Running intrusion detection..."
):

    detected_df = detect_intrusion(
        df
    )


# =========================================================
# RISK
# =========================================================

with st.spinner(
    "⚠️ Calculating risk scores..."
):

    risk_df = assess_risk(
        detected_df
    )


# =========================================================
# RESPONSE
# =========================================================

with st.spinner(
    "🤖 Selecting automated response..."
):

    response_df = apply_response(
        risk_df
    )

# =========================================================
# VERIFICATION
# =========================================================

with st.spinner(
    "✅ Verifying security responses..."
):

    final_df = apply_verification(
        response_df
    )


# =========================================================
# SUMMARY
# =========================================================

st.divider()

st.header("📊 Security Overview")


total_logs = len(
    final_df
)

attack_count = int(
    (
        final_df["prediction"]
        == "ATTACK"
    ).sum()
)

normal_count = int(
    (
        final_df["prediction"]
        == "NORMAL"
    ).sum()
)

critical_count = int(
    (
        final_df["risk_level"]
        == "Critical"
    ).sum()
)

high_count = int(
    (
        final_df["risk_level"]
        == "High"
    ).sum()
)

medium_count = int(
    (
        final_df["risk_level"]
        == "Medium"
    ).sum()
)


col1, col2, col3, col4, col5, col6 = st.columns(6)

col1.metric(
    "Total Logs",
    total_logs
)

col2.metric(
    "🟢 Normal",
    normal_count
)

col3.metric(
    "🔴 Attacks",
    attack_count
)

col4.metric(
    "🚨 Critical",
    critical_count
)

col5.metric(
    "🟠 High",
    high_count
)

col6.metric(
    "🟡 Medium",
    medium_count
)


# =========================================================
# ATTACK RATE
# =========================================================

if total_logs > 0:

    attack_rate = (
        attack_count
        / total_logs
        * 100
    )

    st.progress(
        attack_rate / 100
    )

    st.caption(
        f"Share of uploaded records classified as attacks (not model accuracy): "
        f"{attack_rate:.2f}%"
    )


# =========================================================
# DETECTION RESULTS
# =========================================================

st.divider()

st.header("🔍 Detection Results")


detection_columns = [
    column
    for column in [
        "filename",
        "source_ip",
        "destination",
        "prediction",
        "attack_type",
        "confidence",
        "detection_method"
    ]
    if column in final_df.columns
]


st.dataframe(
    final_df[detection_columns],
    width="stretch",
    hide_index=True
)


# =========================================================
# ATTACKS
# =========================================================

st.divider()

st.header("🚨 Detected Attacks")


attack_df = final_df[
    final_df["prediction"]
    == "ATTACK"
]


if attack_df.empty:

    st.success(
        "🟢 No malicious activity detected."
    )

else:

    st.error(
        f"🚨 {len(attack_df)} attack(s) detected."
    )

    attack_columns = [
        column
        for column in [
            "filename",
            "source_ip",
            "destination",
            "attack_type",
            "confidence",
            "risk_score",
            "risk_level"
        ]
        if column in attack_df.columns
    ]

    st.dataframe(
        attack_df[
            attack_columns
        ],
        width="stretch",
        hide_index=True
    )


# =========================================================
# ATTACK TYPE ANALYSIS
# =========================================================

st.divider()

st.header("📌 Attack Type Analysis")


attack_type_counts = (
    final_df[
        final_df["prediction"]
        == "ATTACK"
    ]["attack_type"]
    .value_counts()
)


if attack_type_counts.empty:

    st.info(
        "No attack types detected."
    )

else:

    chart_df = (
        attack_type_counts
        .rename("Count")
        .to_frame()
    )

    st.bar_chart(
        chart_df
    )


# =========================================================
# EXPLANATION
# =========================================================

st.divider()

st.header("🧠 Explainable AI")


detected_methods = final_df.get(
    "detection_method",
    pd.Series("", index=final_df.index),
).fillna("").astype(str).str.strip()
ml_positions = [
    position
    for position, method in enumerate(detected_methods)
    if method == "Random Forest ML"
]
default_position = ml_positions[0] if ml_positions else 0

selected_index = st.selectbox(
    "Select a log record for detailed explanation",
    options=range(len(final_df)),
    index=default_position,
    format_func=lambda position: (
        f"Record {position} | {detected_methods.iloc[position]} | "
        f"{final_df.iloc[position].get('filename', 'uploaded file')}"
    ),
    key="explanation_record_index",
)


selected_row = final_df.iloc[
    selected_index
]

st.caption(
    f"ML explanation available for {len(ml_positions)} of {len(final_df)} records."
)

if not ml_positions:
    st.warning(
        "No uploaded records contain all 78 numeric features required by the "
        "trained model. SHAP and LIME cannot explain custom-rule log records. "
        "Upload a compatible network-flow CSV to use these explanations."
    )

    sample_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "test_ml_input.csv",
    )
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as sample_file:
            st.download_button(
                "Download ML sample CSV",
                data=sample_file.read(),
                file_name="test_ml_input.csv",
                mime="text/csv",
                key="download_ml_sample",
            )


# =========================================================
# DETECTION EXPLANATION
# =========================================================

st.subheader(
    "Detection Explanation"
)


explanation = generate_explanation(
    selected_row
)


st.info(
    explanation
)


# =========================================================
# SHAP FEATURE EXPLANATION
# =========================================================

st.subheader(
    "🔬 SHAP Feature Explanation"
)


if os.path.exists(MODEL_PATH):

    try:

        # ----------------------------------------------------
        # LOAD SAVED MODEL PACKAGE
        # ----------------------------------------------------

        model_package = joblib.load(
            MODEL_PATH
        )

        # ----------------------------------------------------
        # IMPORTANT:
        # The pkl contains:
        #
        # {
        #     "model": RandomForestClassifier,
        #     "features": [...]
        # }
        #
        # Extract the actual Random Forest.
        # ----------------------------------------------------

        if isinstance(
            model_package,
            dict
        ):

            model = model_package.get(
                "model"
            )

            model_features = model_package.get(
                "features",
                []
            ) or list(getattr(model_package.get("model"), "feature_names_in_", []))

        else:

            model = model_package

            model_features = list(
                getattr(
                    model,
                    "feature_names_in_",
                    []
                )
            )

        # ----------------------------------------------------
        # CHECK MODEL
        # ----------------------------------------------------

        if model is None:

            st.warning(
                "Random Forest model could not be extracted."
            )

        else:

            # ------------------------------------------------
            # CHECK HOW THIS RECORD WAS DETECTED
            # ------------------------------------------------

            detection_method = str(
                selected_row.get(
                    "detection_method",
                    ""
                )
            )

            # ------------------------------------------------
            # SHAP IS MEANINGFUL ONLY FOR ML DETECTION
            # ------------------------------------------------

            if detection_method == "Random Forest ML":

                # --------------------------------------------
                # Convert selected record into model features
                # --------------------------------------------

                selected_data = pd.DataFrame(
                    [selected_row]
                )

                ml_features = convert_to_model_features(
                    selected_data,
                    model,
                    model_features
                )

                # --------------------------------------------
                # Generate real SHAP explanation
                # --------------------------------------------

                shap_results = generate_shap_explanation(
                    ml_features,
                    model,
                    row_index=0,
                    top_n=10
                )

                shap_df = pd.DataFrame(
                    shap_results
                )

                # --------------------------------------------
                # Display SHAP results
                # --------------------------------------------

                if not shap_df.empty:

                    st.dataframe(
                        shap_df,
                        width="stretch",
                        hide_index=True
                    )

                    st.caption(
                        "Positive SHAP values push the prediction "
                        "toward ATTACK. Negative values push it "
                        "toward NORMAL."
                    )

            else:

                # ------------------------------------------------
                # Custom logs don't have the 78 ML features.
                # Do NOT display fake SHAP values.
                # ------------------------------------------------

                st.info(
                    "This record was classified by custom security-log rules. "
                    "Select a record marked 'Random Forest ML' above to view "
                    "its SHAP feature contributions."
                )

    except Exception as e:

        st.warning(
            f"SHAP explanation unavailable: {e}"
        )

else:

    st.warning(
        "Random Forest model is not available."
    )


# =========================================================
# LIME LOCAL EXPLANATION
# =========================================================

st.subheader("🧩 LIME Local Explanation")

selected_method = str(selected_row.get("detection_method", "")).strip()

if selected_method != "Random Forest ML":
    st.info(
        "This record was classified by custom security-log rules. "
        "Select a record marked 'Random Forest ML' above to view its LIME explanation."
    )
elif not os.path.exists(MODEL_PATH):
    st.warning("Trained model is unavailable; LIME cannot be generated.")
else:
    try:
        lime_package = joblib.load(MODEL_PATH)

        if isinstance(lime_package, dict):
            lime_model = lime_package.get("model")
            lime_features = lime_package.get("features", []) or list(
                getattr(lime_package.get("model"), "feature_names_in_", [])
            )
        else:
            lime_model = lime_package
            lime_features = list(
                getattr(lime_model, "feature_names_in_", [])
            )

        if lime_model is None:
            st.warning("Could not load the classifier from the saved model package.")
        elif not hasattr(lime_model, "predict_proba"):
            st.warning(
                "LIME requires a model with predict_proba(). "
                "The loaded classifier does not provide it."
            )
        else:
            ml_rows = final_df[
                final_df.get(
                    "detection_method",
                    pd.Series(index=final_df.index, dtype=str)
                ).astype(str).str.strip() == "Random Forest ML"
            ].copy()

            if ml_rows.empty:
                st.info("No ML-classified records are available for a LIME explanation.")
            else:
                # Match the selected row by its DataFrame index label, then
                # convert that match into a position within ml_rows.
                selected_label = final_df.index[int(selected_index)]
                matching_positions = [
                    position
                    for position, label in enumerate(ml_rows.index)
                    if label == selected_label
                ]

                if not matching_positions:
                    st.warning(
                        "The selected record was not found among the ML-classified "
                        "records. Select a record detected by Random Forest ML."
                    )
                else:
                    selected_position = matching_positions[0]

                    lime_background = convert_to_model_features(
                        ml_rows,
                        lime_model,
                        lime_features
                    )

                    expected_count = getattr(
                        lime_model,
                        "n_features_in_",
                        lime_background.shape[1]
                    )

                    if lime_background.shape[1] != expected_count:
                        st.warning(
                            "LIME feature mismatch: the converted data contains "
                            f"{lime_background.shape[1]} features, but the model expects "
                            f"{expected_count}. Check src/feature_adapter.py and the "
                            "training feature order."
                        )
                    elif lime_background.empty:
                        st.warning("No compatible numeric ML features are available for LIME.")
                    else:
                        lime_results = generate_lime_explanation(
                            lime_background,
                            lime_model,
                            row_index=selected_position,
                            top_n=min(10, lime_background.shape[1]),
                        )

                        lime_df = pd.DataFrame(lime_results)

                        if lime_df.empty:
                            st.info(
                                "LIME did not return feature weights for this record."
                            )
                        else:
                            st.dataframe(
                                lime_df,
                                width="stretch",
                                hide_index=True
                            )

                            if len(lime_background) < 10:
                                st.caption(
                                    "Caution: fewer than 10 compatible background "
                                    "records are available; interpret local LIME "
                                    "weights cautiously."
                                )

                            st.caption(
                                "LIME is a local approximation around the selected "
                                "instance, not a guarantee of causal behavior."
                            )

    except Exception as exc:
        st.warning(f"LIME explanation unavailable: {exc}")


# =========================================================
# RISK ASSESSMENT
# =========================================================

st.divider()

st.header("⚠️ Risk Assessment")


risk_columns = [
    column
    for column in [
        "filename",
        "prediction",
        "attack_type",
        "confidence",
        "attack_severity",
        "resource_criticality",
        "risk_score",
        "risk_level"
    ]
    if column in final_df.columns
]


st.dataframe(
    final_df[
        risk_columns
    ],
    width="stretch",
    hide_index=True
)


selected_risk = selected_row.get(
    "risk_score",
    0
)

selected_level = selected_row.get(
    "risk_level",
    "Low"
)


st.metric(
    "Selected Log Risk Score",
    f"{float(selected_risk):.2f}"
)


st.write(
    f"Risk Level: **{selected_level}**"
)


# =========================================================
# AUTOMATED RESPONSE
# =========================================================

st.divider()

st.header("🤖 Automated Response")


response_columns = [
    column
    for column in [
        "filename",
        "prediction",
        "attack_type",
        "risk_level",
        "automated_response"
    ]
    if column in final_df.columns
]


st.dataframe(
    final_df[
        response_columns
    ],
    width="stretch",
    hide_index=True
)


selected_response = selected_row.get(
    "automated_response",
    "No Action Required"
)


st.info(
    f"Recommended Response: **{selected_response}**"
)


# =========================================================
# RESPONSE VERIFICATION
# =========================================================

st.divider()

st.header("✅ Response Verification")


verification_columns = [
    column
    for column in [
        "filename",
        "prediction",
        "risk_level",
        "automated_response",
        "response_status",
        "execution_mode",
        "verification_status"
    ]
    if column in final_df.columns
]


st.dataframe(
    final_df[
        verification_columns
    ],
    width="stretch",
    hide_index=True
)


# =========================================================
# VERIFICATION SUMMARY
# =========================================================

if "verification_status" in final_df.columns:
    verification_counts = final_df["verification_status"].value_counts()
else:
    verification_counts = pd.Series(dtype="int64")


if not verification_counts.empty:

    st.subheader(
        "Verification Summary"
    )

    st.bar_chart(
        verification_counts
        .rename("Count")
        .to_frame()
    )


# =========================================================
# AUDIT LOG
# =========================================================

st.divider()

st.header("📋 Security Audit Log")


audit_columns = [
    column
    for column in [
        "filename",
        "timestamp",
        "source_ip",
        "destination",
        "prediction",
        "attack_type",
        "confidence",
        "risk_score",
        "risk_level",
        "automated_response",
        "response_status",
        "execution_mode",
        "verification_status"
    ]
    if column in final_df.columns
]


st.dataframe(
    final_df[
        audit_columns
    ],
    width="stretch",
    hide_index=True
)


# =========================================================
# EXPORT
# =========================================================

st.divider()
st.header("⬇️ Export Results")


csv_data = final_df.to_csv(
    index=False
).encode(
    "utf-8"
)


st.download_button(
    label="📥 Download Detection Results",
    data=csv_data,
    file_name=(
        "intrusion_detection_results.csv"
    ),
    mime="text/csv"
)


# =========================================================
# COMPLETE
# =========================================================

st.success(
    "✅ Security analysis completed."
)