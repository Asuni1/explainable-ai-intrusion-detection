import pandas as pd
import numpy as np
import re
from datetime import datetime


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def safe_number(value, default=0.0):
    try:
        number = float(value)
        if np.isfinite(number):
            return number
        return default
    except (ValueError, TypeError):
        return default


def extract_port(value):
    """
    Extract a port number when one is available.
    """

    if pd.isna(value):
        return 0

    text = str(value)

    # Direct number
    match = re.search(r"\b\d{1,5}\b", text)

    if match:
        port = int(match.group())

        if 0 <= port <= 65535:
            return port

    return 0


def hash_to_number(value, minimum=1, maximum=65535):
    """
    Convert text such as an IP address into a
    deterministic numerical value.
    """

    if pd.isna(value):
        return minimum

    text = str(value)

    total = 0

    for character in text:
        total = (
            total * 31
            + ord(character)
        ) % 1000000

    return (
        minimum
        + total % (
            maximum - minimum + 1
        )
    )


def text_contains(text, words):
    text = str(text).lower()

    return any(
        word in text
        for word in words
    )


# =========================================================
# EXTRACT INFORMATION FROM ONE LOG RECORD
# =========================================================

def extract_log_information(row):

    # -----------------------------------------------------
    # Convert complete row into text
    # -----------------------------------------------------

    complete_text = " ".join(
        str(value)
        for value in row.values
        if pd.notna(value)
    ).lower()

    # -----------------------------------------------------
    # Source
    # -----------------------------------------------------

    source = row.get(
        "source_ip",
        row.get(
            "source",
            ""
        )
    )

    # -----------------------------------------------------
    # Destination
    # -----------------------------------------------------

    destination = row.get(
        "destination",
        row.get(
            "destination_ip",
            ""
        )
    )

    # -----------------------------------------------------
    # Action
    # -----------------------------------------------------

    action = str(
        row.get(
            "action",
            ""
        )
    ).lower()

    # -----------------------------------------------------
    # Status
    # -----------------------------------------------------

    status = str(
        row.get(
            "status",
            ""
        )
    ).lower()

    # -----------------------------------------------------
    # Attack type
    # -----------------------------------------------------

    attack_type = str(
        row.get(
            "attack_type",
            ""
        )
    ).lower()

    # -----------------------------------------------------
    # Detect attack indicators
    # -----------------------------------------------------

    is_attack = text_contains(
        complete_text,
        [
            "attack",
            "malicious",
            "suspicious",
            "port_scan",
            "port scan",
            "brute_force",
            "brute force",
            "unauthorized",
            "ddos",
            "dos",
            "sql_injection",
            "sql injection",
            "xss"
        ]
    )

    # -----------------------------------------------------
    # Scan
    # -----------------------------------------------------

    is_scan = text_contains(
        complete_text,
        [
            "scan",
            "port_scan",
            "port scan"
        ]
    )

    # -----------------------------------------------------
    # Authentication
    # -----------------------------------------------------

    is_auth = text_contains(
        complete_text,
        [
            "login",
            "authentication",
            "password",
            "brute"
        ]
    )

    # -----------------------------------------------------
    # Database
    # -----------------------------------------------------

    is_database = text_contains(
        complete_text,
        [
            "database",
            "db",
            "query",
            "sql"
        ]
    )

    # -----------------------------------------------------
    # HTTP
    # -----------------------------------------------------

    is_http = text_contains(
        complete_text,
        [
            "http",
            "https",
            "web",
            "request"
        ]
    )

    # -----------------------------------------------------
    # Suspicious status
    # -----------------------------------------------------

    is_suspicious = text_contains(
        status,
        [
            "suspicious",
            "failed",
            "denied",
            "blocked",
            "error"
        ]
    )

    # -----------------------------------------------------
    # Source numerical representation
    # -----------------------------------------------------

    source_number = hash_to_number(
        source
    )

    # -----------------------------------------------------
    # Destination numerical representation
    # -----------------------------------------------------

    destination_number = hash_to_number(
        destination
    )

    # -----------------------------------------------------
    # Destination port
    # -----------------------------------------------------

    destination_port = extract_port(
        row.get(
            "destination_port",
            row.get(
                "port",
                0
            )
        )
    )

    # If no port exists, derive a
    # deterministic value for the prototype.

    if destination_port == 0:

        if "http" in complete_text:
            destination_port = 80

        elif "https" in complete_text:
            destination_port = 443

        elif "database" in complete_text:
            destination_port = 3306

        elif "ssh" in complete_text:
            destination_port = 22

        elif is_scan:
            destination_port = 0

        else:
            destination_port = (
                int(source_number) % 60000
            )

    # =====================================================
    # BUILD SECURITY FEATURE VALUES
    # =====================================================

    flow_duration = 1000

    if is_attack:
        flow_duration = 5000

    if is_scan:
        flow_duration = 200

    total_fwd_packets = 1

    if is_attack:
        total_fwd_packets = 10

    if is_scan:
        total_fwd_packets = 50

    total_bwd_packets = 1

    if is_attack:
        total_bwd_packets = 5

    total_fwd_length = (
        total_fwd_packets * 60
    )

    total_bwd_length = (
        total_bwd_packets * 60
    )

    packet_max = 60

    packet_min = 20

    packet_mean = 40

    packet_std = 5

    if is_attack:
        packet_max = 1500
        packet_mean = 500
        packet_std = 100

    # =====================================================
    # GENERATE THE 78 FEATURES
    # =====================================================

    features = {}

    # -----------------------------------------------------
    # Core flow features
    # -----------------------------------------------------

    features[
        "Destination Port"
    ] = destination_port

    features[
        "Flow Duration"
    ] = flow_duration

    features[
        "Total Fwd Packets"
    ] = total_fwd_packets

    features[
        "Total Backward Packets"
    ] = total_bwd_packets

    features[
        "Total Length of Fwd Packets"
    ] = total_fwd_length

    features[
        "Total Length of Bwd Packets"
    ] = total_bwd_length

    features[
        "Fwd Packet Length Max"
    ] = packet_max

    features[
        "Fwd Packet Length Min"
    ] = packet_min

    features[
        "Fwd Packet Length Mean"
    ] = packet_mean

    features[
        "Fwd Packet Length Std"
    ] = packet_std

    # -----------------------------------------------------
    # Additional common CIC-style features
    # -----------------------------------------------------

    features[
        "Bwd Packet Length Max"
    ] = packet_max

    features[
        "Bwd Packet Length Min"
    ] = packet_min

    features[
        "Bwd Packet Length Mean"
    ] = packet_mean

    features[
        "Bwd Packet Length Std"
    ] = packet_std

    features[
        "Flow Bytes/s"
    ] = (
        total_fwd_length
        + total_bwd_length
    ) / max(
        flow_duration / 1000000,
        0.001
    )

    features[
        "Flow Packets/s"
    ] = (
        total_fwd_packets
        + total_bwd_packets
    ) / max(
        flow_duration / 1000000,
        0.001
    )

    features[
        "Flow IAT Mean"
    ] = 100

    features[
        "Flow IAT Std"
    ] = 10

    features[
        "Flow IAT Max"
    ] = 200

    features[
        "Flow IAT Min"
    ] = 1

    features[
        "Fwd IAT Total"
    ] = 100

    features[
        "Fwd IAT Mean"
    ] = 50

    features[
        "Fwd IAT Std"
    ] = 5

    features[
        "Fwd IAT Max"
    ] = 100

    features[
        "Fwd IAT Min"
    ] = 1

    features[
        "Bwd IAT Total"
    ] = 100

    features[
        "Bwd IAT Mean"
    ] = 50

    features[
        "Bwd IAT Std"
    ] = 5

    features[
        "Bwd IAT Max"
    ] = 100

    features[
        "Bwd IAT Min"
    ] = 1

    features[
        "Fwd PSH Flags"
    ] = int(
        is_attack
    )

    features[
        "Bwd PSH Flags"
    ] = 0

    features[
        "Fwd URG Flags"
    ] = 0

    features[
        "Bwd URG Flags"
    ] = 0

    features[
        "Fwd Header Length"
    ] = 20

    features[
        "Bwd Header Length"
    ] = 20

    features[
        "Fwd Packets/s"
    ] = total_fwd_packets

    features[
        "Bwd Packets/s"
    ] = total_bwd_packets

    features[
        "Min Packet Length"
    ] = packet_min

    features[
        "Max Packet Length"
    ] = packet_max

    features[
        "Packet Length Mean"
    ] = packet_mean

    features[
        "Packet Length Std"
    ] = packet_std

    features[
        "Packet Length Variance"
    ] = packet_std ** 2

    features[
        "FIN Flag Count"
    ] = 0

    features[
        "SYN Flag Count"
    ] = int(
        is_attack
    )

    features[
        "RST Flag Count"
    ] = int(
        is_suspicious
    )

    features[
        "PSH Flag Count"
    ] = int(
        is_attack
    )

    features[
        "ACK Flag Count"
    ] = 1

    features[
        "URG Flag Count"
    ] = 0

    features[
        "CWE Flag Count"
    ] = 0

    features[
        "ECE Flag Count"
    ] = 0

    features[
        "Down/Up Ratio"
    ] = (
        total_bwd_packets
        / max(
            total_fwd_packets,
            1
        )
    )

    features[
        "Average Packet Size"
    ] = (
        total_fwd_length
        + total_bwd_length
    ) / max(
        total_fwd_packets
        + total_bwd_packets,
        1
    )

    features[
        "Avg Fwd Segment Size"
    ] = packet_mean

    features[
        "Avg Bwd Segment Size"
    ] = packet_mean

    features[
        "Fwd Header Length.1"
    ] = 20

    features[
        "Subflow Fwd Packets"
    ] = total_fwd_packets

    features[
        "Subflow Fwd Bytes"
    ] = total_fwd_length

    features[
        "Subflow Bwd Packets"
    ] = total_bwd_packets

    features[
        "Subflow Bwd Bytes"
    ] = total_bwd_length

    features[
        "Init_Win_bytes_forward"
    ] = 8192

    features[
        "Init_Win_bytes_backward"
    ] = 8192

    features[
        "act_data_pkt_fwd"
    ] = total_fwd_packets

    features[
        "min_seg_size_forward"
    ] = 20

    features[
        "Active Mean"
    ] = 100

    features[
        "Active Std"
    ] = 10

    features[
        "Active Max"
    ] = 200

    features[
        "Active Min"
    ] = 1

    features[
        "Idle Mean"
    ] = 100

    features[
        "Idle Std"
    ] = 10

    features[
        "Idle Max"
    ] = 200

    features[
        "Idle Min"
    ] = 1

    # -----------------------------------------------------
    # Make attack activity stronger
    # for prototype feature representation
    # -----------------------------------------------------

    if is_scan:

        features[
            "Total Fwd Packets"
        ] = 100

        features[
            "Flow Packets/s"
        ] = 500

        features[
            "SYN Flag Count"
        ] = 20

        features[
            "Destination Port"
        ] = 0

    if "brute" in attack_type:

        features[
            "Total Fwd Packets"
        ] = 30

        features[
            "Flow Duration"
        ] = 10000

    if "ddos" in attack_type:

        features[
            "Total Fwd Packets"
        ] = 500

        features[
            "Flow Packets/s"
        ] = 5000

    return features


# =========================================================
# CONVERT COMPLETE DATAFRAME
# =========================================================

def extract_features(df):

    if df is None or df.empty:

        return pd.DataFrame()

    extracted_rows = []

    for _, row in df.iterrows():

        features = extract_log_information(
            row
        )

        extracted_rows.append(
            features
        )

    feature_df = pd.DataFrame(
        extracted_rows,
        index=df.index
    )

    # -----------------------------------------------------
    # Clean values
    # -----------------------------------------------------

    feature_df = feature_df.replace(
        [
            np.inf,
            -np.inf
        ],
        np.nan
    )

    feature_df = feature_df.fillna(
        0
    )

    return feature_df