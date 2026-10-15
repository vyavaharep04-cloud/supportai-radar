import pandas as pd
import pytest

from supportai.incidents import detect_incidents


def make_tickets():
    rows = []
    # Six historical windows with two customers each.
    for window in range(6):
        for customer in range(2):
            rows.append(
                {
                    "ticket_id": f"H-{window}-{customer}",
                    "customer_id": f"C-{window}-{customer}",
                    "created_at": (
                        pd.Timestamp("2026-10-10", tz="UTC")
                        + pd.Timedelta(minutes=30 * window)
                    ),
                    "group_id": "payment_checkout",
                }
            )
    # Incident window: six distinct customers.
    for customer in range(6):
        rows.append(
            {
                "ticket_id": f"I-{customer}",
                "customer_id": f"INC-{customer}",
                "created_at": pd.Timestamp("2026-10-10 03:00", tz="UTC")
                + pd.Timedelta(minutes=customer),
                "group_id": "payment_checkout",
            }
        )
    return pd.DataFrame(rows)


def test_detects_customer_spike():
    incidents = detect_incidents(make_tickets())
    assert len(incidents) == 1
    assert incidents[0].status == "alert"
    assert incidents[0].unique_customers == 6
    assert incidents[0].raw_tickets == 6
    assert len(incidents[0].ticket_ids) == 6
    assert any("potential incident" in line for line in incidents[0].explanation)


def test_repeated_tickets_from_one_customer_do_not_inflate_count():
    tickets = make_tickets()
    incident_rows = tickets["ticket_id"].str.startswith("I-")
    repeated = tickets.loc[incident_rows].iloc[[0]].copy()
    repeated["ticket_id"] = "I-REPEAT"
    repeated["created_at"] = pd.Timestamp("2026-10-10 03:10", tz="UTC")
    tickets = pd.concat([tickets, repeated], ignore_index=True)

    incidents = detect_incidents(tickets)
    assert len(incidents) == 1
    assert incidents[0].unique_customers == 6
    assert incidents[0].raw_tickets == 7


def test_excludes_other_group():
    tickets = make_tickets()
    tickets["group_id"] = "other"
    assert detect_incidents(tickets) == []


def test_empty_input_returns_no_incidents():
    columns = ["ticket_id", "customer_id", "created_at", "group_id"]
    assert detect_incidents(pd.DataFrame(columns=columns)) == []


def test_missing_required_column_raises():
    with pytest.raises(ValueError, match="Missing incident input columns"):
        detect_incidents(pd.DataFrame({"ticket_id": ["T1"]}))


def test_invalid_thresholds_raise():
    with pytest.raises(ValueError):
        detect_incidents(make_tickets(), alert_ratio=1.0, watch_ratio=2.0)
    with pytest.raises(ValueError, match="finite ratios"):
        detect_incidents(make_tickets(), watch_ratio=float("nan"))


def test_mixed_timezone_formats_are_normalized():
    rows = []
    base = pd.Timestamp("2026-10-10T00:00:00Z")
    for window in range(4):
        for customer in range(2):
            timestamp = base + pd.Timedelta(minutes=30 * window)
            # Alternate equivalent timezone representations.
            if (window + customer) % 2:
                timestamp = timestamp.tz_convert("Asia/Kolkata")
            rows.append(
                {
                    "ticket_id": f"T-{window}-{customer}",
                    "customer_id": f"C-{window}-{customer}",
                    "created_at": timestamp.isoformat(),
                    "group_id": "payments",
                }
            )
    for customer in range(6):
        rows.append(
            {
                "ticket_id": f"SPIKE-{customer}",
                "customer_id": f"NEW-{customer}",
                "created_at": (base + pd.Timedelta(minutes=120)).isoformat(),
                "group_id": "payments",
            }
        )
    incidents = detect_incidents(pd.DataFrame(rows))
    assert len(incidents) == 1
    assert incidents[0].status == "alert"
    assert str(incidents[0].window_start.tz) == "UTC"


def test_missing_windows_count_as_zero_in_baseline():
    base = pd.Timestamp("2026-10-10T00:00:00Z")
    tickets = pd.DataFrame(
        [
            {"ticket_id": "A", "customer_id": "A", "created_at": base, "group_id": "payments"},
            # The next 30-minute window is intentionally empty.
            {"ticket_id": "B", "customer_id": "B", "created_at": base + pd.Timedelta(minutes=60), "group_id": "payments"},
            {"ticket_id": "C", "customer_id": "C", "created_at": base + pd.Timedelta(minutes=60), "group_id": "payments"},
        ]
    )
    incidents = detect_incidents(
        tickets,
        min_customers=2,
        min_baseline_windows=2,
        watch_ratio=2,
        alert_ratio=3,
    )
    assert len(incidents) == 1
    assert incidents[0].baseline_customers == 0.5
    assert incidents[0].status == "alert"


def test_invalid_rows_and_empty_groups_are_ignored():
    tickets = pd.concat(
        [
            make_tickets(),
            pd.DataFrame(
                [
                    {"ticket_id": "BAD", "customer_id": "C", "created_at": "not-a-date", "group_id": "payments"},
                    {"ticket_id": "EMPTY", "customer_id": "C", "created_at": "2026-10-10T03:00:00Z", "group_id": " "},
                ]
            ),
        ],
        ignore_index=True,
    )
    incidents = detect_incidents(tickets)
    assert len(incidents) == 1
    assert incidents[0].unique_customers == 6


def test_insufficient_history_does_not_emit_incident():
    tickets = make_tickets().iloc[:4].copy()
    assert detect_incidents(tickets) == []
