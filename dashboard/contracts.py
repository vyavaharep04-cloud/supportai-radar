"""What the dashboard needs from the backend: view models and table schemas.

This module is the *documented interface* between the Streamlit UI and the
backend modules that are still being written.  Nothing here makes decisions;
it only describes the shapes the pages render.

How to connect a backend module
-------------------------------
Pages never import backend modules directly.  They call ``dashboard.backend``,
whose functions return the objects below (or ``None`` while a module does not
exist yet).  To connect a module, change ONE function in ``dashboard/backend.py``
so that it translates the backend's output into these shapes.  No page changes.

All shapes are PROPOSALS from the UI side.  If the backend prefers different
shapes, change them here and in the adapter together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from supportai.normalizer import NormalizationResult

# --------------------------------------------------------------------------
# Display vocabulary
# --------------------------------------------------------------------------

# ASSUMPTION (confirm with the urgency engine owner): four priority levels.
# The UI only *labels* a priority string supplied by the backend; it never
# computes one.
PRIORITIES: tuple[str, ...] = ("P1", "P2", "P3", "P4")
PRIORITY_LABELS: dict[str, str] = {
    "P1": "Critical",
    "P2": "High",
    "P3": "Medium",
    "P4": "Low",
}

# Detector states and incident severities shown by the incident screens.
DETECTOR_STATES: tuple[str, ...] = ("normal", "watch", "alert")
SEVERITIES: tuple[str, ...] = ("critical", "high", "medium", "low")

# Synthetic-demo labels (ground truth of the generator; real data has none).
SEGMENTS: tuple[str, ...] = ("normal", "decoy", "planted")

# --------------------------------------------------------------------------
# Single-ticket analysis (Ticket Analysis page)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class EntityView:
    kind: str  # "amount", "order id", "transaction id", "payment method", "product"
    value: str  # display text, taken from the RAW message


@dataclass(frozen=True)
class ClassificationView:
    issue_id: str  # a taxonomy issue id
    confidence: float  # 0..1
    runner_up: tuple[tuple[str, float], ...] = ()  # (issue_id, probability)
    needs_human_review: bool = False
    review_reason: str | None = None


@dataclass(frozen=True)
class UrgencyFactor:
    name: str
    points: float
    detail: str


@dataclass(frozen=True)
class UrgencyView:
    score: float
    priority: str  # one of PRIORITIES
    factors: tuple[UrgencyFactor, ...] = ()
    floors_applied: tuple[str, ...] = ()  # hard priority floors / safety rules


@dataclass(frozen=True)
class PlaybookView:
    playbook_id: str
    reply: str
    checklist: tuple[str, ...] = ()


@dataclass(frozen=True)
class TicketAnalysis:
    """Everything the backend knows about one message.

    A stage that is not connected (or produced nothing) is ``None``; the page
    then renders an honest "not connected" card for that stage.
    """

    raw_text: str
    normalization: "NormalizationResult | None" = None  # real, from supportai
    entities: tuple[EntityView, ...] | None = None
    classification: ClassificationView | None = None
    department_id: str | None = None  # decided by the backend (may be human_review)
    urgency: UrgencyView | None = None
    playbook: PlaybookView | None = None


# --------------------------------------------------------------------------
# Table / bundle schemas
# --------------------------------------------------------------------------

# Ticket table used by Overview and Ticket Explorer (one row per ticket).
TICKET_COLUMNS: tuple[str, ...] = (
    "ticket_id",  # str
    "customer_id",  # str, distinct-customer counting key
    "created_at",  # datetime64
    "raw_text",  # str
    "normalized_text",  # str
    "issue_id",  # str, taxonomy id
    "department_id",  # str, taxonomy department id
    "confidence",  # float 0..1
    "priority",  # one of PRIORITIES
    "needs_review",  # bool
    "entities",  # str, short human-readable summary, "" if none
)


@dataclass(frozen=True)
class IncidentView:
    """A potential incident (inferred from ticket volume; never a confirmed outage)."""

    incident_id: str
    group_id: str  # taxonomy incident_group
    window_start: "pd.Timestamp"
    window_end: "pd.Timestamp"
    unique_customers: int
    raw_tickets: int
    baseline_customers: float  # historical baseline for the same window length
    severity: str  # one of SEVERITIES
    status: str  # "alert" | "watch" | "resolved"
    ticket_ids: tuple[str, ...] = ()
    explanation: tuple[str, ...] = ()  # human-readable evidence lines

    @property
    def ratio(self) -> float:
        return self.unique_customers / self.baseline_customers if self.baseline_customers else float("inf")


@dataclass(frozen=True)
class ReplayData:
    """Simulated incident replay.

    ``tickets``: TICKET_COLUMNS + ``group_id`` + ``segment`` (SEGMENTS).
    ``timeline``: one row per (tick, group): ``ts``, ``group_id``,
    ``distinct_customers``, ``raw_tickets``, ``baseline``, ``threshold``,
    ``state`` (DETECTOR_STATES).
    """

    tickets: "pd.DataFrame"
    timeline: "pd.DataFrame"
    window_minutes: int
    description: str = ""

@dataclass(frozen=True)
class EvaluationReport:
    accuracy: float
    macro_f1: float
    per_class: "pd.DataFrame"
    confusion: "pd.DataFrame"
    critical_recall: float | None
    critical_precision: float | None
    thresholds: "pd.DataFrame"
    incident_scenarios: "pd.DataFrame"
    dataset_note: str = ""
    extras: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# Backend capabilities (what the dashboard expects to exist)
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CapabilitySpec:
    key: str
    label: str
    stage: str  # UNDERSTAND | DECIDE | ASSIST | DETECT | ACT | QUALITY
    module: str  # PROPOSED module path; confirm with the backend owner
    adapter_ready: bool  # does dashboard/backend.py already translate it?
    interface: str  # documented signature the adapter will call


CAPABILITY_SPECS: tuple[CapabilitySpec, ...] = (
    CapabilitySpec(
        "ingestion", "CSV ticket input", "UNDERSTAND", "dashboard.backend", True,
        "Set SUPPORTAI_TICKETS_CSV to a CSV with ticket_id, customer_id, timestamp, raw_text; predictions are generated locally.",
    ),
    CapabilitySpec(
        "normalization", "Hinglish normalization", "UNDERSTAND", "supportai.normalizer", True,
        "get_default_normalizer().explain(text) -> NormalizationResult",
    ),
    CapabilitySpec(
        "entities", "Entity extraction", "UNDERSTAND", "supportai.entities", True,
        "extract_entities(raw_text) -> dict[amounts, order_ids, transaction_ids, payment_methods, products]",
    ),
    CapabilitySpec(
    "classifier", "Issue classification + review routing", "DECIDE", "supportai.classifier", True,
    "TicketClassifier.load().predict(raw_text) -> ClassificationResult",
    ),
    CapabilitySpec(
        "taxonomy", "Taxonomy & department mapping", "DECIDE", "supportai.taxonomy", True,
        "load_taxonomy() -> Taxonomy",
    ),
    CapabilitySpec(
    "urgency", "Urgency & priority scoring", "DECIDE", "supportai.urgency", True,
    "score_urgency(raw_text, issue_type, repeat_contact=False) -> UrgencyResult",
    ),
    CapabilitySpec(
        "playbooks", "Playbook replies & checklists", "ASSIST", "supportai.playbooks", True,
        "get_playbook(issue_id) -> Playbook",
    ),
    CapabilitySpec(
        "incidents", "Incident detection & explanation", "DETECT", "supportai.incidents", True,
        "detect_incidents(tickets) -> list[Incident]",
    ),
    CapabilitySpec(
        "incident_explainer", "Incident explanation", "ACT", "supportai.incidents", True,
        "explain_incident(incident) -> list[str]",
    ),
    CapabilitySpec(
        "replay", "Incident replay", "DETECT", "supportai.replay", True,
        "run_replay(scenario) -> ReplayResult",
    ),
    CapabilitySpec(
        "evaluation", "Evaluation & metrics", "QUALITY", "supportai.evaluation", True,
        "evaluate() -> EvaluationResult",
    ),
)

WORKFLOW_STAGES: tuple[str, ...] = ("UNDERSTAND", "DECIDE", "ASSIST", "DETECT", "ACT")
