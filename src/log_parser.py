import json
import pandas as pd


def parse_csv(file):
    file.seek(0)
    return pd.read_csv(file, low_memory=False)


def _extract_records(data):
    """Flatten common AWS CloudTrail, Azure Activity and JSON log wrappers."""
    if isinstance(data, list):
        records = []
        for item in data:
            records.extend(_extract_records(item))
        return records

    if isinstance(data, dict):
        for key in ("Records", "value", "events", "logs", "records", "items"):
            if isinstance(data.get(key), list):
                return _extract_records(data[key])
        return [data]

    return []


def parse_json(file):
    file.seek(0)
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig", errors="replace")
    content = content.strip()

    if not content:
        return pd.DataFrame()

    try:
        data = json.loads(content)
        records = _extract_records(data)
    except json.JSONDecodeError:
        # Newline-delimited JSON (one JSON object per line).
        records = []
        for line_number, line in enumerate(content.splitlines(), start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.extend(_extract_records(json.loads(line)))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_number}: {exc}"
                ) from exc

    return pd.json_normalize(records, sep=".") if records else pd.DataFrame()


def parse_text(file):
    """Parse simple pipe-separated key=value records and ordinary text logs."""
    file.seek(0)
    content = file.read()
    if isinstance(content, bytes):
        content = content.decode("utf-8-sig", errors="replace")

    rows = []
    for line in content.splitlines():
        line = line.strip()
        if not line:
            continue

        row = {"message": line}
        parts = line.split("|")
        for part in parts:
            part = part.strip()
            if "=" in part:
                key, value = part.split("=", 1)
                row[key.strip()] = value.strip()
        rows.append(row)

    return pd.DataFrame(rows)


def normalize_columns(df):
    if df is None or df.empty:
        return df

    df = df.copy()
    aliases = {
        "source": "source_ip",
        "src": "source_ip",
        "src_ip": "source_ip",
        "source_address": "source_ip",
        "sourceip": "source_ip",
        "dst": "destination",
        "dst_ip": "destination",
        "destination_address": "destination",
        "destinationip": "destination",
        "type": "attack_type",
        "attack": "attack_type",
        "attack_name": "attack_type",
        "event": "label",
        "class": "label",
        "category": "label",
    }

    existing = {str(c).strip().lower() for c in df.columns}
    rename_map = {}
    for column in df.columns:
        key = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        target = aliases.get(key)
        if target and target.lower() not in existing:
            rename_map[column] = target
            existing.add(target.lower())

    return df.rename(columns=rename_map) if rename_map else df


def normalize_attack_type(df):
    if df is None or df.empty:
        return df

    result = df.copy()
    if "attack_type" not in result.columns:
        result["attack_type"] = "None"
    result["attack_type"] = (
        result["attack_type"].fillna("None").astype(str).str.strip().replace("", "None")
    )
    return result


def parse_log_file(file):
    filename = file.name.lower()
    if filename.endswith(".csv"):
        df = parse_csv(file)
    elif filename.endswith(".json"):
        df = parse_json(file)
    elif filename.endswith((".txt", ".log")):
        df = parse_text(file)
    else:
        raise ValueError(f"Unsupported file format: {filename}")

    if df is None:
        return pd.DataFrame()
    return normalize_attack_type(normalize_columns(df))


def parse_multiple_files(files):
    all_logs = []
    errors = []

    for file in files:
        try:
            df = parse_log_file(file)
            if df is None or df.empty:
                errors.append(f"{file.name}: no records found")
                continue
            df = df.copy()
            df["filename"] = file.name
            all_logs.append(df)
            print(f"[OK] {file.name}: {len(df)} records")
        except Exception as exc:
            errors.append(f"{file.name}: {exc}")
            print(f"[ERROR] {file.name}: {exc}")

    if not all_logs:
        if errors:
            print("Parser errors:", errors)
        return pd.DataFrame()

    combined = pd.concat(all_logs, ignore_index=True, sort=False)
    print("=" * 50)
    print("LOG PARSER SUMMARY")
    print("Files processed:", len(all_logs))
    print("Total records:", len(combined))
    print("Total columns:", len(combined.columns))
    print("Errors:", errors if errors else "None")
    print("=" * 50)
    return combined
