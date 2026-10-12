"""CSV ingestion and validation for SupportAI Radar."""

from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = {
    "ticket_id",
    "customer_id",
    "timestamp",
    "raw_text",
    "issue_type",
    "department",
    "incident_group",
    "scenario",
    "is_incident_ticket",
}


class IngestionError(ValueError):
    """Raised when ticket data cannot be loaded or validated."""


def load_tickets(csv_path: str | Path) -> pd.DataFrame:
    """Load and validate ticket data without treating labels as predictions."""

    path = Path(csv_path)

    if not path.exists():
        raise FileNotFoundError(f"Ticket file not found: {path}")

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError as exc:
        raise IngestionError("The CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise IngestionError(f"Unable to parse CSV file: {exc}") from exc

    if df.empty:
        raise IngestionError("The CSV file contains no ticket records.")

    # Normalize column names.
    df.columns = df.columns.str.strip()

    missing_columns = REQUIRED_COLUMNS - set(df.columns)

    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise IngestionError(f"Missing required columns: {missing}")

    # Remove records that cannot represent valid tickets.
    df["ticket_id"] = df["ticket_id"].astype("string").str.strip()
    df["customer_id"] = df["customer_id"].astype("string").str.strip()
    df["raw_text"] = df["raw_text"].astype("string").str.strip()

    invalid_mask = (
        df["ticket_id"].isna()
        | df["ticket_id"].eq("")
        | df["customer_id"].isna()
        | df["customer_id"].eq("")
        | df["raw_text"].isna()
        | df["raw_text"].eq("")
    )

    df = df.loc[~invalid_mask].copy()

    if df.empty:
        raise IngestionError(
            "No valid tickets remain after removing invalid records."
        )

    # Preserve original labels as ground truth.
    # These must not be presented as model predictions.
    df["ground_truth_issue_id"] = df["issue_type"]
    df["ground_truth_department_id"] = df["department"]

    # Standardize timestamp handling.
    # Training records may have empty timestamps, so NaT is acceptable.
    df["created_at"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    # Standardize incident flags.
    if pd.api.types.is_bool_dtype(df["is_incident_ticket"]):
        df["is_incident_ticket"] = df["is_incident_ticket"].fillna(False)
    else:
        flag_values = (
            df["is_incident_ticket"]
            .astype("string")
            .str.strip()
            .str.lower()
        )

        valid_flags = {"true", "false", "1", "0"}

        invalid_flags = flag_values.notna() & ~flag_values.isin(valid_flags)

        if invalid_flags.any():
            raise IngestionError(
                "is_incident_ticket must contain true/false or 1/0 values."
            )

        df["is_incident_ticket"] = flag_values.isin({"true", "1"})

    # Standardized text column for downstream processing.
    df["raw_text"] = df["raw_text"].astype(str)

    return df.reset_index(drop=True)