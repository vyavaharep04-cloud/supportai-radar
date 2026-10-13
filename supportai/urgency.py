"""Explainable urgency scoring for SupportAI Radar."""

from __future__ import annotations

import re
from dataclasses import dataclass

from supportai.taxonomy import load_taxonomy


# Base urgency scores by issue type.
BASE_SCORES = {
    "payment_failed": 30,
    "money_deducted_no_order": 45,
    "double_charge": 40,
    "checkout_error": 20,
    "refund_pending": 20,
    "unauthorized_activity": 60,
    "login_problem": 25,
    "order_delay": 10,
    "wrong_damaged_item": 20,
    "cancel_modify_order": 10,
    "other": 10,
}

# Each pattern represents a risk signal and its scoring bonus.
RISK_PATTERNS = (
    (
        "unauthorized_access",
        re.compile(
            r"\b(?:unauthori[sz]ed|without permission|"
            r"account (?:was )?hacked|account compromised|"
            r"someone accessed my account|unknown login)\b",
            re.IGNORECASE,
        ),
        25,
    ),
    (
        "fraud_or_unknown_transaction",
        re.compile(
            r"\b(?:fraud|scam|unknown transaction|"
            r"transaction i did not make|"
            r"payment i did not authorize)\b",
            re.IGNORECASE,
        ),
        25,
    ),
    (
        "money_deducted",
        re.compile(
            r"\b(?:money|amount|paise|paisa)\s+"
            r"(?:was\s+)?(?:deducted|debited|cut|kat)\b|"
            r"\b(?:paise|paisa)\s+kat\s+gaye\b",
            re.IGNORECASE,
        ),
        15,
    ),
    (
        "urgent_language",
        re.compile(
            r"\b(?:urgent|immediately|right now|"
            r"jaldi|abhi help karo)\b",
            re.IGNORECASE,
        ),
        10,
    ),
)

FRUSTRATION_PATTERNS = (
    (
        "strong_frustration",
        re.compile(
            r"\b(?:furious|unacceptable|worst service|"
            r"fed up|cheated|scammed)\b",
            re.IGNORECASE,
        ),
        10,
    ),
    (
        "frustration",
        re.compile(
            r"\b(?:angry|frustrated|ridiculous|"
            r"disappointed|very upset)\b",
            re.IGNORECASE,
        ),
        5,
    ),
)

# Basic negation handling for short phrases immediately preceding a risk cue.
NEGATION_PATTERN = re.compile(
    r"\b(?:not|no|never|didn't|did not|don't|do not|"
    r"hasn't|has not|haven't|have not|isn't|is not)\b",
    re.IGNORECASE,
)

AMOUNT_PATTERN = re.compile(
    r"(?:₹|rs\.?\s*|inr\s*)?(\d[\d,]*(?:\.\d{1,2})?)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class UrgencyResult:
    """Explainable urgency score and resulting priority."""

    score: int
    priority: str
    reasons: tuple[str, ...]
    base_score: int
    risk_bonus: int
    amount_bonus: int
    frustration_bonus: int
    repeat_contact_bonus: int
    priority_floor_applied: str | None


def _extract_amounts(raw_text: str) -> list[float]:
    """Extract likely monetary amounts from ticket text."""

    amounts: list[float] = []

    for match in AMOUNT_PATTERN.finditer(raw_text):
        token = match.group(1).replace(",", "")

        try:
            amount = float(token)
        except ValueError:
            continue

        # Avoid interpreting ordinary small numbers as currency.
        prefix = raw_text[max(0, match.start() - 5):match.start()].lower()
        has_currency_marker = (
            "₹" in prefix
            or "rs" in prefix
            or "inr" in prefix
        )

        if has_currency_marker or amount >= 1000:
            amounts.append(amount)

    return amounts


def _is_negated(raw_text: str, match_start: int) -> bool:
    """Check for a nearby negation before a risk phrase."""

    preceding_text = raw_text[max(0, match_start - 35):match_start]

    # Only consider the latest clause, reducing false negation across
    # unrelated statements.
    preceding_text = re.split(r"[.!?;,\n]", preceding_text)[-1]

    negations = list(NEGATION_PATTERN.finditer(preceding_text))
    if not negations:
        return False

    # Restrict negation to a short distance before the risk phrase.
    return len(preceding_text) - negations[-1].end() <= 20


def _priority_for_score(score: int) -> str:
    if score >= 70:
        return "P1"
    if score >= 50:
        return "P2"
    if score >= 30:
        return "P3"
    return "P4"


def score_urgency(
    raw_text: str,
    issue_type: str,
    repeat_contact: bool = False,
) -> UrgencyResult:
    """Score ticket urgency using raw text and the predicted issue type."""

    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("raw_text must be a non-empty string.")

    taxonomy = load_taxonomy()

    # Validate the issue type against the configured taxonomy.
    taxonomy.get_issue(issue_type)

    if issue_type not in BASE_SCORES:
        raise ValueError(
            f"No base urgency score configured for issue type: {issue_type}"
        )

    base_score = BASE_SCORES[issue_type]
    reasons: list[str] = [f"Base issue score: {base_score}"]

    # Collect risk signals, excluding phrases with nearby negation.
    risk_bonus = 0
    matched_risks: list[str] = []

    for rule_id, pattern, points in RISK_PATTERNS:
        match = pattern.search(raw_text)

        if match and not _is_negated(raw_text, match.start()):
            matched_risks.append(rule_id)
            risk_bonus += points
            reasons.append(f"{rule_id}: +{points}")

    risk_bonus = min(risk_bonus, 30)

    if risk_bonus == 30 and len(matched_risks) > 1:
        reasons.append("Risk bonus capped at +30")

    # Apply the highest applicable amount band, not cumulative bands.
    amounts = _extract_amounts(raw_text)
    highest_amount = max(amounts, default=0)

    if highest_amount >= 20_000:
        amount_bonus = 15
    elif highest_amount >= 5_000:
        amount_bonus = 10
    elif highest_amount >= 1_000:
        amount_bonus = 5
    else:
        amount_bonus = 0

    if amount_bonus:
        reasons.append(
            f"Amount threshold reached (₹{highest_amount:,.2f}): "
            f"+{amount_bonus}"
        )

    # Use the strongest applicable frustration signal.
    frustration_bonus = 0

    for rule_id, pattern, points in FRUSTRATION_PATTERNS:
        match = pattern.search(raw_text)

        if match and not _is_negated(raw_text, match.start()):
            frustration_bonus = max(frustration_bonus, points)
            reasons.append(f"{rule_id}: +{points}")
            break

    repeat_contact_bonus = 10 if repeat_contact else 0

    if repeat_contact_bonus:
        reasons.append("Repeat contact: +10")

    score = (
        base_score
        + risk_bonus
        + amount_bonus
        + frustration_bonus
        + repeat_contact_bonus
    )

    priority_floor_applied = None

    # Mandatory security floor: clear security-risk language is critical.
    security_risk = any(
        rule_id == "unauthorized_access"
        for rule_id in matched_risks
    )

    fraud_risk = any(
        rule_id == "fraud_or_unknown_transaction"
        for rule_id in matched_risks
    )

    if security_risk or fraud_risk:
        if score < 70:
            reasons.append("Security/fraud priority floor: P1 Critical")
        score = max(score, 70)
        priority_floor_applied = "P1"

    # Money deducted without an order must never fall below P2.
    if issue_type == "money_deducted_no_order" and score < 50:
        reasons.append("Money deducted without order priority floor: P2 High")
        score = 50
        priority_floor_applied = "P2"

    return UrgencyResult(
        score=score,
        priority=_priority_for_score(score),
        reasons=tuple(reasons),
        base_score=base_score,
        risk_bonus=risk_bonus,
        amount_bonus=amount_bonus,
        frustration_bonus=frustration_bonus,
        repeat_contact_bonus=repeat_contact_bonus,
        priority_floor_applied=priority_floor_applied,
    )