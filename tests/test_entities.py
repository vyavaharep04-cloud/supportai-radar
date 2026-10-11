"""Tests for support-ticket entity extraction."""

from supportai.entities import extract_entities


def test_extracts_multiple_entity_types():
    text = (
        "₹2,499 deducted via UPI for order ORD123456. "
        "Transaction UTR987654321. "
        "My headphones are not working."
    )

    result = extract_entities(text)

    assert result["amounts"] == [2499.0]
    assert result["order_ids"] == ["ORD123456"]
    assert result["transaction_ids"] == ["UTR987654321"]
    assert result["payment_methods"] == ["upi"]
    assert result["products"] == ["headphones"]


def test_extracts_amount_formats():
    text = "I paid Rs. 1,299 and another INR 500.50."

    result = extract_entities(text)

    assert result["amounts"] == [1299.0, 500.5]


def test_extracts_multiple_order_ids():
    text = "Compare orders ORD123456 and ORDER-987654."

    result = extract_entities(text)

    assert result["order_ids"] == [
        "ORD123456",
        "ORDER-987654",
    ]


def test_extracts_payment_method_and_product():
    result = extract_entities(
        "Paid using my credit card for a laptop."
    )

    assert result["payment_methods"] == ["credit_card"]
    assert result["products"] == ["laptop"]


def test_duplicate_entities_are_not_repeated():
    result = extract_entities(
        "UPI payment failed. UPI was used again for ₹500 and ₹500."
    )

    assert result["amounts"] == [500.0]
    assert result["payment_methods"] == ["upi"]


def test_empty_or_missing_text_returns_empty_lists():
    expected = {
        "amounts": [],
        "order_ids": [],
        "transaction_ids": [],
        "payment_methods": [],
        "products": [],
    }

    assert extract_entities("") == expected
    assert extract_entities("   ") == expected
    assert extract_entities(None) == expected