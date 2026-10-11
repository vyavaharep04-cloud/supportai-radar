"""Tests for the synthetic ticket generator."""
import pandas as pd
import pytest

from scripts.generate_synthetic_data import (
    RANDOM_SEED,
    generate_replay_tickets,
    generate_training_tickets,
)


def test_training_dataset_has_expected_size_and_labels():
    frame = generate_training_tickets()
    assert len(frame) == 450
    assert frame["issue_type"].nunique() == 11
    assert frame["raw_text"].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
    assert not frame["issue_type"].isna().any()


def test_training_dataset_is_reproducible():
    pd.testing.assert_frame_equal(
        generate_training_tickets(seed=RANDOM_SEED),
        generate_training_tickets(seed=RANDOM_SEED),
    )


def test_training_dataset_labels_match_taxonomy():
    from supportai.taxonomy import load_taxonomy
    assert set(generate_training_tickets()["issue_type"]) == set(load_taxonomy().issue_ids())


def test_replay_dataset_has_600_rows_and_expected_scenarios():
    frame = generate_replay_tickets()
    assert len(frame) == 600
    assert set(frame["scenario"]) == {
        "background", "planted_payment_incident",
        "decoy_order_delay_promo", "decoy_single_customer_repeat",
    }
    assert frame["timestamp"].is_monotonic_increasing


def test_planted_incident_has_80_distinct_customers():
    frame = generate_replay_tickets()
    incident = frame[frame["scenario"] == "planted_payment_incident"]
    assert len(incident) == 80
    assert incident["customer_id"].nunique() == 80
    assert incident["incident_id"].eq("INC-PAY-001").all()
    assert incident["incident_group"].eq("payment_checkout").all()


def test_single_customer_decoy_has_eight_tickets():
    frame = generate_replay_tickets()
    decoy = frame[frame["scenario"] == "decoy_single_customer_repeat"]
    assert len(decoy) == 8
    assert decoy["customer_id"].nunique() == 1
    assert decoy["issue_type"].eq("unauthorized_activity").all()


def test_order_delay_decoy_has_32_tickets():
    frame = generate_replay_tickets()
    decoy = frame[frame["scenario"] == "decoy_order_delay_promo"]
    assert len(decoy) == 32
    assert decoy["issue_type"].eq("order_delay").all()


def test_custom_replay_count_is_rejected():
    with pytest.raises(ValueError, match="exactly 600"):
        generate_replay_tickets(count=100)
