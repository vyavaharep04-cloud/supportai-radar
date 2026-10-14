import pytest

from supportai.urgency import score_urgency


def test_money_deducted_without_order_has_high_priority_floor():
    result = score_urgency(
        "Money deducted but no order was created",
        "money_deducted_no_order",
    )

    assert result.priority in {"P1", "P2"}
    assert result.score >= 50


def test_unauthorized_activity_is_critical():
    result = score_urgency(
        "Someone accessed my account without permission",
        "unauthorized_activity",
    )

    assert result.priority == "P1"
    assert result.score >= 70


def test_high_amount_adds_correct_bonus():
    result = score_urgency(
        "I was charged ₹12,000",
        "payment_failed",
    )

    assert result.amount_bonus == 10


def test_very_high_amount_adds_fifteen_points():
    result = score_urgency(
        "Payment issue for ₹25,000",
        "payment_failed",
    )

    assert result.amount_bonus == 15


@pytest.mark.parametrize(
    "text",
    [
        "Payment failed for order ORD123456",
        "Transaction TXN778899 is pending",
        "Call me on 9876543210",
        "Reference UTR: 123456789012",
        "Order ORD482913; amount charged was ₹1,299.50",
    ],
)
def test_identifiers_and_phone_numbers_are_not_amounts(text):
    result = score_urgency(text, "payment_failed")
    assert result.amount_bonus == (5 if "₹1,299.50" in text else 0)


def test_risk_bonus_is_capped_at_thirty():
    result = score_urgency(
        "Urgent, fraud transaction, money deducted",
        "payment_failed",
    )

    assert result.risk_bonus <= 30


def test_repeat_contact_bonus():
    result = score_urgency(
        "My delivery is late",
        "order_delay",
        repeat_contact=True,
    )

    assert result.repeat_contact_bonus == 10


def test_empty_message_is_rejected():
    with pytest.raises(ValueError, match="non-empty"):
        score_urgency("", "payment_failed")


def test_unknown_issue_type_is_rejected():
    with pytest.raises(KeyError, match="Unknown issue type"):
        score_urgency("Payment failed", "unknown_issue")


def test_negated_unauthorized_phrase_does_not_trigger_security_floor():
    result = score_urgency(
        "I did not notice any unauthorized activity",
        "login_problem",
    )

    assert result.priority_floor_applied is None


def test_priority_boundaries():
    from supportai.urgency import _priority_for_score

    assert _priority_for_score(69) == "P2"
    assert _priority_for_score(70) == "P1"
    assert _priority_for_score(49) == "P3"
    assert _priority_for_score(30) == "P3"
    assert _priority_for_score(29) == "P4"
