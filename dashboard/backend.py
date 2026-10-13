"""The ONLY place where the dashboard talks to backend modules.

Rules
-----
* Pages import from here (or ``data_access``), never from ``supportai.*``
  intelligence modules directly.
* Every function returns real backend output, or ``None`` / an empty field when
  that backend module does not exist yet.  Nothing here invents results.
* Connecting a new module means editing ONE function below so that it returns
  the shapes defined in ``dashboard.contracts``.

Currently connected: the Hinglish normalizer and the taxonomy.
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from dashboard.contracts import (
    CAPABILITY_SPECS,
    ClassificationView,
    EntityView,
    EvaluationReport,
    IncidentView,
    ReplayData,
    TicketAnalysis,
    UrgencyFactor,
    UrgencyView,
)
from supportai.classifier import TicketClassifier
from supportai.normalizer import get_default_normalizer
from supportai.taxonomy import Taxonomy, load_taxonomy
from supportai.urgency import score_urgency


@dataclass(frozen=True)
class Capability:
    key: str
    label: str
    stage: str
    module: str
    interface: str
    connected: bool
    detail: str


def _module_exists(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def capabilities() -> tuple[Capability, ...]:
    """Honest status of every backend capability the dashboard expects."""
    out = []
    for spec in CAPABILITY_SPECS:
        present = _module_exists(spec.module)
        connected = present and spec.adapter_ready
        if connected:
            detail = "Connected"
        elif present:
            detail = f"`{spec.module}` exists, but the dashboard adapter is not written yet"
        else:
            detail = f"`{spec.module}` not found"
        out.append(
            Capability(spec.key, spec.label, spec.stage, spec.module, spec.interface, connected, detail)
        )
    return tuple(out)


def is_connected(key: str) -> bool:
    return any(c.key == key and c.connected for c in capabilities())


@lru_cache(maxsize=1)
def get_taxonomy() -> Taxonomy:
    """The real taxonomy from ``config/taxonomy.yaml`` (raises ConfigError if invalid)."""
    return load_taxonomy()

@lru_cache(maxsize=1)
def get_classifier() -> TicketClassifier:
    """Load the trained classifier once and reuse it across dashboard requests."""
    return TicketClassifier.load()
def _review_reason(
    raw_text: str,
    issue_id: str,
    confidence: float,
) -> str | None:
    """Explain why a ticket must be reviewed by a human."""
    classifier = get_classifier()

    if confidence < classifier.confidence_threshold:
        return (
            f"Model confidence {confidence:.2f} is below the "
            f"{classifier.confidence_threshold:.2f} review threshold."
        )

    if issue_id == "other":
        return "The issue was classified as 'other' and requires human review."

    # Conservative checks for clear security-related language.
    text = get_default_normalizer().normalize(raw_text)

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

    has_security_signal = any(
        phrase in text for phrase in security_patterns
    )

    security_issue_ids = {
        issue.id
        for issue in get_taxonomy().issue_types
        if "security" in issue.department
    }

    if has_security_signal and issue_id not in security_issue_ids:
        return (
            "Security-related language conflicts with the predicted issue; "
            "manual verification is required."
        )

    return None


def _format_amount(amount: float) -> str:
    return f"₹{amount:,.2f}".removesuffix(".00")


def entity_views(raw: dict) -> tuple[EntityView, ...]:
    """Translate ``supportai.entities.extract_entities`` output into display objects."""
    out: list[EntityView] = []
    out += [EntityView("amount", _format_amount(a)) for a in raw.get("amounts", [])]
    out += [EntityView("order id", v) for v in raw.get("order_ids", [])]
    out += [EntityView("transaction id", v) for v in raw.get("transaction_ids", [])]
    out += [EntityView("payment method", v.replace("_", " ")) for v in raw.get("payment_methods", [])]
    out += [EntityView("product", v.replace("_", " ")) for v in raw.get("products", [])]
    return tuple(out)


def extract_entity_views(text: str) -> tuple[EntityView, ...] | None:
    """Real entity extraction on the RAW text, or ``None`` if that module is not connected."""
    if not is_connected("entities"):
        return None
    from supportai.entities import extract_entities

    return entity_views(extract_entities(text))


def analyze_ticket(text: str) -> TicketAnalysis:
    """Analyse a message using the connected backend stages."""
    normalization = get_default_normalizer().explain(text)
    entities = extract_entity_views(text)

    # Keep classification honest if the trained model is unavailable.
    try:
        classifier = get_classifier()
        result = classifier.predict(text)

        runner_up = tuple(
            sorted(
                (
                    (issue_id, probability)
                    for issue_id, probability in result.probabilities.items()
                    if issue_id != result.issue_type
                ),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
        )

        review_reason = _review_reason(
            text,
            result.issue_type,
            result.confidence,
        )

        from dashboard.contracts import ClassificationView

        classification = ClassificationView(
            issue_id=result.issue_type,
            confidence=result.confidence,
            runner_up=runner_up,
            needs_human_review=review_reason is not None,
            review_reason=review_reason,
        )

        # A review-required ticket is routed to the Human Review department.
        # Its predicted issue remains visible in the classification result.
        department_id = (
            "human_review"
            if review_reason is not None
            else get_taxonomy().department_for(result.issue_type).id
        )
# Stage 7: Calculate urgency independently of review routing.
        urgency_result = score_urgency(
            raw_text=text,
            issue_type=result.issue_type,
        )

        urgency = UrgencyView(
            score=urgency_result.score,
            priority=urgency_result.priority,
            factors=(
                UrgencyFactor(
                    name="Base issue score",
                    points=urgency_result.base_score,
                    detail=f"Base score for {result.issue_type}",
                ),
                UrgencyFactor(
                    name="Risk signals",
                    points=urgency_result.risk_bonus,
                    detail="Risk-related phrases detected in the message",
                ),
                UrgencyFactor(
                    name="Amount",
                    points=urgency_result.amount_bonus,
                    detail="Bonus based on the monetary amount",
                ),
                UrgencyFactor(
                    name="Frustration",
                    points=urgency_result.frustration_bonus,
                    detail="Customer frustration signals",
                ),
                UrgencyFactor(
                    name="Repeat contact",
                    points=urgency_result.repeat_contact_bonus,
                    detail="Bonus for repeat contact",
                ),
            ),
            floors_applied=(
                (urgency_result.priority_floor_applied,)
                if urgency_result.priority_floor_applied
                else ()
            ),
        )

    except (FileNotFoundError, RuntimeError):
        classification = None
        department_id = None
        urgency = None

    return TicketAnalysis(
        raw_text=text,
        normalization=normalization,
        entities=entities,
        classification=classification,
        department_id=department_id,
        urgency=urgency if classification is not None else None,
    )


def normalizer_info() -> dict[str, str | int]:
    """Version/fingerprint of the lexicon in use (shown in the decision trace)."""
    normalizer = get_default_normalizer()
    return {"version": normalizer.lexicon_version, "fingerprint": normalizer.lexicon_fingerprint}


# -- Not connected yet.  Each returns None until its backend module exists. ----


def load_tickets() -> pd.DataFrame | None:
    """Ticket table with ``contracts.TICKET_COLUMNS`` (needs ingestion + classifier + urgency)."""
    return None


def load_incidents() -> list[IncidentView] | None:
    """Potential incidents (needs ``supportai.incidents``)."""
    return None


def load_replay() -> ReplayData | None:
    """Simulated replay bundle (needs ``supportai.replay``)."""
    return None


def load_evaluation() -> EvaluationReport | None:
    """Evaluation report (needs ``supportai.evaluation``)."""
    return None
