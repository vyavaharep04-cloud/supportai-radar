"""Extract structured entities from raw customer support tickets."""

from __future__ import annotations

import re
from typing import Any


# Matches common order identifiers such as ORD123456 or ORDER-123456.
ORDER_ID_PATTERN = re.compile(
    r"\b(?:ORDER|ORD)[\s:#_-]*([A-Z0-9-]{4,})\b",
    re.IGNORECASE,
)

# Matches UTR and transaction/reference identifiers.
TRANSACTION_ID_PATTERN = re.compile(
    r"\b(?:UTR|TXN|TRANSACTION|REF)[\s:#_-]*([A-Z0-9-]{5,})\b",
    re.IGNORECASE,
)

# Matches common Indian currency formats.
AMOUNT_PATTERN = re.compile(
    r"(?<![A-Z0-9])"
    r"(?:₹\s*|Rs\.?\s*|INR\s*)"
    r"([0-9]+(?:,[0-9]{2,3})*(?:\.[0-9]{1,2})?)"
    r"(?![A-Z0-9])",
    re.IGNORECASE,
)

PAYMENT_METHOD_PATTERNS = {
    "upi": re.compile(r"\bUPI\b", re.IGNORECASE),
    "credit_card": re.compile(
        r"\bcredit\s+card\b", re.IGNORECASE
    ),
    "debit_card": re.compile(
        r"\bdebit\s+card\b", re.IGNORECASE
    ),
    "net_banking": re.compile(
        r"\bnet\s+banking\b", re.IGNORECASE
    ),
    "wallet": re.compile(
        r"\b(?:wallet|paytm wallet)\b", re.IGNORECASE
    ),
    "cash_on_delivery": re.compile(
        r"\b(?:cash on delivery|COD)\b", re.IGNORECASE
    ),
}

# Conservative phrase matching; these are examples, not a full catalog.
PRODUCT_PATTERNS = {
    "headphones": re.compile(r"\bheadphones?\b", re.IGNORECASE),
    "mobile_phone": re.compile(
        r"\b(?:mobile phone|smartphone|phone)\b", re.IGNORECASE
    ),
    "laptop": re.compile(r"\blaptops?\b", re.IGNORECASE),
    "charger": re.compile(r"\bchargers?\b", re.IGNORECASE),
    "shoes": re.compile(r"\bshoes?\b", re.IGNORECASE),
    "shirt": re.compile(r"\bshirts?\b", re.IGNORECASE),
}

def _extract_identifiers(
    text: str,
    pattern: re.Pattern[str],
) -> list[str]:
    """Extract identifiers while preserving their ID prefixes."""
    values: list[str] = []

    for match in pattern.finditer(text):
        value = match.group(0).strip()

        # Remove descriptive words but retain the actual ID prefix.
        value = re.sub(
            r"^(?:ORDER|TRANSACTION)\s+",
            "",
            value,
            flags=re.IGNORECASE,
        )

        if value and value not in values:
            values.append(value)

    return values


def extract_entities(text: Any) -> dict[str, Any]:
    """
    Extract useful entities from an original support-ticket message.

    Non-string or missing values return an empty result. This makes the
    function convenient to use with pandas columns containing missing data.
    """
    result: dict[str, Any] = {
        "amounts": [],
        "order_ids": [],
        "transaction_ids": [],
        "payment_methods": [],
        "products": [],
    }

    if not isinstance(text, str) or not text.strip():
        return result

    result["order_ids"] = _extract_identifiers(
        text, ORDER_ID_PATTERN
    )

    result["transaction_ids"] = _extract_identifiers(
        text, TRANSACTION_ID_PATTERN
    )

    amounts: list[float] = []

    for match in AMOUNT_PATTERN.finditer(text):
        raw_amount = match.group(1).replace(",", "")

        try:
            amount = float(raw_amount)
        except ValueError:
            continue

        if amount not in amounts:
            amounts.append(amount)

    result["amounts"] = amounts

    for method, pattern in PAYMENT_METHOD_PATTERNS.items():
        if pattern.search(text):
            result["payment_methods"].append(method)

    for product, pattern in PRODUCT_PATTERNS.items():
        if pattern.search(text):
            result["products"].append(product)

    return result