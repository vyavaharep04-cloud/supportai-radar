from __future__ import annotations

import pandas as pd

from supportai.incidents import Incident, explain_incident


def test_explanation_preserves_evidence_and_outage_disclaimer():
    incident = Incident(
        incident_id="payment_checkout-20261010T100000",
        group_id="payment_checkout",
        window_start=pd.Timestamp("2026-10-10T10:00:00Z"),
        window_end=pd.Timestamp("2026-10-10T10:30:00Z"),
        unique_customers=8,
        raw_tickets=11,
        baseline_customers=2.0,
        severity="high",
        status="alert",
        ticket_ids=("T-1", "T-2"),
        explanation=("8 distinct customers reported related issues.",),
    )
    lines = explain_incident(incident)
    assert lines[0] == "8 distinct customers reported related issues."
    assert any("not a confirmed outage" in line.lower() for line in lines)


def test_explanation_rejects_wrong_input():
    import pytest
    with pytest.raises(TypeError):
        explain_incident(object())
