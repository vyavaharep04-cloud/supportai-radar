import pandas as pd
import pytest

from supportai.ingestion import IngestionError, load_tickets


def create_valid_csv(path):
    df = pd.DataFrame(
        {
            "ticket_id": ["T001", "T002"],
            "customer_id": ["C001", "C002"],
            "timestamp": [
                "2026-10-11 10:00:00",
                "",
            ],
            "raw_text": [
                "Payment failed",
                "Order has not arrived",
            ],
            "issue_type": [
                "payment_failed",
                "order_delay",
            ],
            "department": [
                "Payments & Billing",
                "Orders & Delivery",
            ],
            "incident_group": [
                "payment_checkout",
                "orders_delivery",
            ],
            "scenario": [
                "background",
                "background",
            ],
            "is_incident_ticket": [
                False,
                False,
            ],
        }
    )

    df.to_csv(path, index=False)


def test_loads_valid_tickets(tmp_path):
    path = tmp_path / "tickets.csv"
    create_valid_csv(path)

    df = load_tickets(path)

    assert len(df) == 2
    assert "created_at" in df.columns
    assert "ground_truth_issue_id" in df.columns
    assert df.loc[0, "ground_truth_issue_id"] == "payment_failed"


def test_preserves_empty_training_timestamps(tmp_path):
    path = tmp_path / "tickets.csv"
    create_valid_csv(path)

    df = load_tickets(path)

    assert pd.isna(df.loc[1, "created_at"])


def test_missing_columns_raise_error(tmp_path):
    path = tmp_path / "tickets.csv"
    pd.DataFrame({"ticket_id": ["T001"]}).to_csv(path, index=False)

    with pytest.raises(IngestionError, match="Missing required columns"):
        load_tickets(path)


def test_empty_csv_raises_error(tmp_path):
    path = tmp_path / "tickets.csv"
    path.write_text("", encoding="utf-8")

    with pytest.raises(IngestionError, match="empty"):
        load_tickets(path)


def test_removes_invalid_ticket_records(tmp_path):
    path = tmp_path / "tickets.csv"
    create_valid_csv(path)

    df = pd.read_csv(path)
    df.loc[0, "raw_text"] = ""
    df.to_csv(path, index=False)

    loaded = load_tickets(path)

    assert len(loaded) == 1
    assert loaded.iloc[0]["ticket_id"] == "T002"


def test_missing_file_raises_error(tmp_path):
    path = tmp_path / "missing.csv"

    with pytest.raises(FileNotFoundError):
        load_tickets(path)