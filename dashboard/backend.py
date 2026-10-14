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
from supportai.pipeline import analyze_text, get_pipeline_classifier
from supportai.normalizer import get_default_normalizer
from supportai.taxonomy import Taxonomy, load_taxonomy


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

def get_classifier():
    """Compatibility accessor for the shared pipeline's cached classifier."""
    return get_pipeline_classifier()
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
    """Translate the shared pipeline result into dashboard view models."""
    normalization = get_default_normalizer().explain(text)
    entities = extract_entity_views(text)

    try:
        result = analyze_text(text)
        classification = ClassificationView(
            issue_id=result.classification.issue_type,
            confidence=result.classification.confidence,
            runner_up=result.runner_up,
            needs_human_review=result.needs_human_review,
            review_reason=result.review_reason,
        )
        urgency_result = result.urgency
        urgency = UrgencyView(
            score=urgency_result.score,
            priority=urgency_result.priority,
            factors=(
                UrgencyFactor(
                    name="Base issue score",
                    points=urgency_result.base_score,
                    detail="Base score for " + result.classification.issue_type,
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
        department_id = result.department_id
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
