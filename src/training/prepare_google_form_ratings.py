import argparse
import json
import re
from pathlib import Path

import pandas as pd

MIN_REQUIRED_CANONICAL = [
    "participant_id",
    "sample_id",
    "groove_quality",
    "coherence",
    "variety",
]

OPTIONAL_COLUMNS = ["overall_preference", "comment"]

METRIC_PATTERNS = {
    "groove_quality": ["groove"],
    "coherence": ["coherence", "flow"],
    "variety": ["variety", "non-repetitive", "non repetitive", "nonrepetitive"],
    "overall_preference": ["overall", "preference"],
    "comment": ["comment", "feedback"],
}


def normalize_colname(col: str) -> str:
    col = str(col).strip().lower()
    col = re.sub(r"[^a-z0-9]+", "_", col)
    col = re.sub(r"_+", "_", col).strip("_")
    return col



def standardize_track_value(value: str, sample_prefix: str) -> str:
    text = str(value).strip()
    m = re.search(r"(\d+)", text)
    if not m:
        return text
    return f"{sample_prefix}{int(m.group(1))}"



def infer_participant_column(df: pd.DataFrame):
    lower = {normalize_colname(c): c for c in df.columns}
    for key in [
        "participant_id",
        "participant_id_",
        "participant_id_name",
        "participant_id_email",
        "name",
        "full_name",
        "email_address",
        "email",
        "timestamp",
    ]:
        if key in lower:
            return lower[key]
    return None



def matches_exact_long_format(df: pd.DataFrame) -> bool:
    normalized = {normalize_colname(c): c for c in df.columns}
    required = {"timestamp", "track", "groove_quality", "coherence", "variety"}
    return required.issubset(set(normalized.keys()))



def convert_exact_long_format(df: pd.DataFrame, sample_prefix: str) -> pd.DataFrame:
    normalized = {normalize_colname(c): c for c in df.columns}
    out = pd.DataFrame()
    out["participant_id"] = df[normalized["timestamp"]].astype(str).str.strip().str.replace("/", "-", regex=False).str.replace(" ", "_", regex=False)
    out["sample_id"] = df[normalized["track"]].apply(lambda x: standardize_track_value(x, sample_prefix))
    out["groove_quality"] = pd.to_numeric(df[normalized["groove_quality"]], errors="coerce")
    out["coherence"] = pd.to_numeric(df[normalized["coherence"]], errors="coerce")
    out["variety"] = pd.to_numeric(df[normalized["variety"]], errors="coerce")
    if "overall_preference" in normalized:
        out["overall_preference"] = pd.to_numeric(df[normalized["overall_preference"]], errors="coerce")
    if "comment" in normalized:
        out["comment"] = df[normalized["comment"]].fillna("").astype(str)
    return out



def infer_sample_metric_columns(columns):
    mapping = {}
    unparsed = []
    for col in columns:
        low = str(col).lower().strip()
        match = re.search(r"sample\s*[-_ ]*(\d+)", low)
        if not match:
            continue
        sample_num = int(match.group(1))
        metric_name = None
        for canonical, patterns in METRIC_PATTERNS.items():
            if any(p in low for p in patterns):
                metric_name = canonical
                break
        if metric_name is None:
            unparsed.append(col)
            continue
        mapping.setdefault(sample_num, {})[metric_name] = col
    return mapping, unparsed



def convert_wide_to_long(df: pd.DataFrame, sample_prefix: str, participant_col: str | None):
    mapping, unparsed = infer_sample_metric_columns(df.columns)
    if not mapping:
        raise ValueError(
            "Could not infer the Google Form layout. Supported inputs are either: "
            "(1) exact long-format columns like Timestamp, Track, Groove quality, Coherence, Variety; or "
            "(2) wide-format columns like 'Sample 1 - Groove quality'."
        )

    rows = []
    for row_idx, row in df.iterrows():
        participant_id = None
        if participant_col is not None and pd.notna(row[participant_col]):
            participant_id = str(row[participant_col]).strip()
        if not participant_id:
            participant_id = f"P{row_idx + 1:02d}"
        participant_id = participant_id.replace("/", "-").replace(" ", "_")

        for sample_num in sorted(mapping):
            metric_cols = mapping[sample_num]
            record = {
                "participant_id": participant_id,
                "sample_id": f"{sample_prefix}{sample_num}",
                "groove_quality": None,
                "coherence": None,
                "variety": None,
                "comment": "",
            }
            if "overall_preference" in metric_cols:
                record["overall_preference"] = None

            for metric_name, source_col in metric_cols.items():
                value = row[source_col]
                if metric_name == "comment":
                    record[metric_name] = "" if pd.isna(value) else str(value)
                else:
                    record[metric_name] = None if pd.isna(value) or str(value).strip() == "" else float(value)

            core_metrics = [record["groove_quality"], record["coherence"], record["variety"]]
            if all(v is not None for v in core_metrics):
                rows.append(record)

    out_df = pd.DataFrame(rows)
    if out_df.empty:
        raise ValueError("No complete sample ratings were parsed from the input CSV.")

    return out_df, mapping, unparsed



def main():
    parser = argparse.ArgumentParser(description="Convert your survey CSV into canonical Task 4 human_ratings.csv format.")
    parser.add_argument("--input", default="data/survey_results/google_form_raw.csv", help="Path to the raw Google Form CSV export")
    parser.add_argument("--output", default="data/survey_results/human_ratings.csv", help="Path to the cleaned long-format ratings CSV")
    parser.add_argument("--sample-prefix", default="transformer_sample_", help="Prefix used to build sample IDs")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    participant_col = None
    mapping = {}
    unparsed = []

    if matches_exact_long_format(df):
        out_df = convert_exact_long_format(df, sample_prefix=args.sample_prefix)
    else:
        normalized = {normalize_colname(c): c for c in df.columns}
        if set(MIN_REQUIRED_CANONICAL).issubset(set(normalized.keys())):
            out_df = pd.DataFrame({
                "participant_id": df[normalized["participant_id"]],
                "sample_id": df[normalized["sample_id"]],
                "groove_quality": df[normalized["groove_quality"]],
                "coherence": df[normalized["coherence"]],
                "variety": df[normalized["variety"]],
            })
            if "overall_preference" in normalized:
                out_df["overall_preference"] = df[normalized["overall_preference"]]
            if "comment" in normalized:
                out_df["comment"] = df[normalized["comment"]]
        else:
            participant_col = infer_participant_column(df)
            out_df, mapping, unparsed = convert_wide_to_long(df, sample_prefix=args.sample_prefix, participant_col=participant_col)

    numeric_cols = ["groove_quality", "coherence", "variety"]
    if "overall_preference" in out_df.columns:
        numeric_cols.append("overall_preference")

    for col in numeric_cols:
        out_df[col] = pd.to_numeric(out_df[col], errors="coerce")
    out_df = out_df.dropna(subset=["participant_id", "sample_id", *numeric_cols]).copy()
    out_df[numeric_cols] = out_df[numeric_cols].clip(lower=1, upper=5)
    if "comment" in out_df.columns:
        out_df["comment"] = out_df["comment"].fillna("").astype(str)

    column_order = ["participant_id", "sample_id", "groove_quality", "coherence", "variety"]
    if "overall_preference" in out_df.columns:
        column_order.append("overall_preference")
    if "comment" in out_df.columns:
        column_order.append("comment")
    out_df = out_df[column_order]
    out_df.to_csv(output_path, index=False)

    summary = {
        "input_csv": str(input_path),
        "output_csv": str(output_path),
        "num_rows": int(len(out_df)),
        "num_participants": int(out_df["participant_id"].nunique()),
        "num_samples": int(out_df["sample_id"].nunique()),
        "participant_source_column": participant_col,
        "detected_sample_columns": {str(k): v for k, v in mapping.items()},
        "unparsed_detected_columns": unparsed,
        "used_overall_preference": bool("overall_preference" in out_df.columns),
    }
    with open(output_path.parent / "prepare_google_form_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(out_df.head())
    print(f"Saved cleaned ratings to {output_path}")


if __name__ == "__main__":
    main()
