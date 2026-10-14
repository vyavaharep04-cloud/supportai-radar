"""Shared, UI-independent ticket analysis pipeline.

The dashboard adapter and batch consumers should use these same decisions.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable

from supportai.classifier import ClassificationResult, TicketClassifier
from supportai.entities import extract_entities
from supportai.normalizer import NormalizationResult, get_default_normalizer
from supportai.taxonomy import Taxonomy, load_taxonomy
from supportai.urgency import UrgencyResult, score_urgency


@dataclass(frozen=True)
class TicketDecision:
    """Backend decision data for one raw ticket."""

    raw_text: str
    normalization: NormalizationResult
    entities: dict
    classification: ClassificationResult
    runner_up: tuple[tuple[str, float], ...]
    needs_human_review: bool
    review_reason: str | None
    department_id: str
    urgency: UrgencyResult


@lru_cache(maxsize=1)
def get_pipeline_classifier() -> TicketClassifier:
    """Load the trained model once per process."""
    return TicketClassifier.load()


@lru_cache(maxsize=1)
def get_pipeline_taxonomy() -> Taxonomy:
    return load_taxonomy()


def _review_reason(
    raw_text: str,
    issue_id: str,
    confidence: float,
    confidence_threshold: float,
) -> str | None:
    """Return the existing conservative human-review decision and reason."""
    if confidence < confidence_threshold:
        return (
            f"Model confidence {confidence:.2f} is below the "
            f"{confidence_threshold:.2f} review threshold."
        )

    if issue_id == "other":
        return "The issue was classified as 'other' and requires human review."

    normalized = get_default_normalizer().normalize(raw_text)
    security_patterns = (
        "unauthorized transaction",
        "unauthorised transaction",
        "unknown transaction",
        "fraud",
        "someone hacked",
        "account hacked",
        "account compromised",
        "without my permission",
        "without my authorization",
        "without my authorisation",
        "i did not make this transaction",
        "i didn't make this transaction",
        "not my transaction",
    )
    has_security_signal = any(phrase in normalized for phrase in security_patterns)
    security_issue_ids = {
        issue.id
        for issue in get_pipeline_taxonomy().issue_types
        if "security" in issue.department
    }
    if has_security_signal and issue_id not in security_issue_ids:
        return (
            "Security-related language conflicts with the predicted issue; "
            "manual verification is required."
        )
    return None


def analyze_text(
    raw_text: str,
    *,
    classifier: TicketClassifier | None = None,
    taxonomy: Taxonomy | None = None,
    repeat_contact: bool = False,
) -> TicketDecision:
    """Run normalization, extraction, classification, review, routing, and urgency."""
    if not isinstance(raw_text, str) or not raw_text.strip():
        raise ValueError("raw_text must be a non-empty string.")

    active_classifier = classifier or get_pipeline_classifier()
    active_taxonomy = taxonomy or get_pipeline_taxonomy()

    normalization = get_default_normalizer().explain(raw_text)
    entities = extract_entities(raw_text)
    classification = active_classifier.predict(raw_text)
    runner_up = tuple(
        sorted(
            (
                (issue_id, probability)
                for issue_id, probability in classification.probabilities.items()
                if issue_id != classification.issue_type
            ),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
    )
    review_reason = _review_reason(
        raw_text,
        classification.issue_type,
        classification.confidence,
        active_classifier.confidence_threshold,
    )
    department_id = (
        "human_review"
        if review_reason is not None
        else active_taxonomy.department_for(classification.issue_type).id
    )
    urgency = score_urgency(
        raw_text=raw_text,
        issue_type=classification.issue_type,
        repeat_contact=repeat_contact,
    )
    return TicketDecision(
        raw_text=raw_text,
        normalization=normalization,
        entities=entities,
        classification=classification,
        runner_up=runner_up,
        needs_human_review=review_reason is not None,
        review_reason=review_reason,
        department_id=department_id,
        urgency=urgency,
    )


def analyze_dataframe(
    dataframe,
    *,
    text_column: str = "raw_text",
    classifier: TicketClassifier | None = None,
    taxonomy: Taxonomy | None = None,
) -> list[TicketDecision]:
    """Analyze each row using the same per-ticket decisions as analyze_text.

    Missing or blank messages fail explicitly rather than producing fabricated
    classifications. Dataset-specific columns are preserved by the caller.
    """
    if text_column not in dataframe.columns:
        raise ValueError(f"Dataframe must contain {text_column!r}.")
    results = []
    for index, value in dataframe[text_column].items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(
                f"Dataframe row {index!r} has an empty or non-string ticket message."
            )
        results.append(
            analyze_text(value, classifier=classifier, taxonomy=taxonomy)
        )
    return results
