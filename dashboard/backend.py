"""The ONLY place where the dashboard talks to backend modules.

Rules
-----
* Pages import from here (or ``data_access``), never from ``supportai.*``
  intelligence modules directly.
* Every function returns real backend output, or ``None`` / an empty field when
  that backend module does not exist yet.  Nothing here invents results.
* Connecting a new module means editing ONE function below so that it returns
  the shapes defined in ``dashboard.contracts``.

Currently connected: configured CSV ticket input, normalization, entity extraction, classification, taxonomy, review routing, urgency, playbooks, incident detection/explanation, replay, and evaluation.
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
from supportai.pipeline import analyze_dataframe, analyze_text, get_pipeline_classifier
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

    playbook = None
    if classification is not None and is_connected("playbooks"):
        from supportai.playbooks import get_playbook

        item = get_playbook(classification.issue_id)
        playbook = __import__("dashboard.contracts", fromlist=["PlaybookView"]).PlaybookView(
            playbook_id=item.playbook_id,
            reply=item.reply,
            checklist=item.checklist,
        )

    return TicketAnalysis(
        raw_text=text,
        normalization=normalization,
        entities=entities,
        classification=classification,
        department_id=department_id,
        urgency=urgency if classification is not None else None,
        playbook=playbook,
    )
def normalizer_info() -> dict[str, str | int]:
    """Version/fingerprint of the lexicon in use (shown in the decision trace)."""
    normalizer = get_default_normalizer()
    return {"version": normalizer.lexicon_version, "fingerprint": normalizer.lexicon_fingerprint}


# -- CSV-backed ticket analysis. Other loaders stay disconnected until their
# real detection/replay/evaluation algorithms are implemented. -----------------


def _configured_ticket_path():
    """Resolve an optional inference CSV path from SUPPORTAI_TICKETS_CSV."""
    import os
    from pathlib import Path

    configured = os.getenv("SUPPORTAI_TICKETS_CSV", "").strip()
    if not configured:
        return None
    path = Path(configured).expanduser()
    if not path.is_absolute():
        path = Path(__file__).resolve().parents[1] / path
    return path


@lru_cache(maxsize=8)
def _read_inference_csv(path_text: str, modified_ns: int, size: int) -> pd.DataFrame:
    """Read a minimal ticket-input CSV; labels are optional and never predictions."""
    from pathlib import Path

    path = Path(path_text)
    try:
        # Read identifiers as strings so values such as 000123 keep their zeros.
        frame = pd.read_csv(path, dtype="string")
    except pd.errors.EmptyDataError as exc:
        raise ValueError(f"Configured ticket CSV is empty: {path}") from exc
    except pd.errors.ParserError as exc:
        raise ValueError(f"Unable to parse configured ticket CSV: {exc}") from exc

    frame.columns = frame.columns.astype(str).str.strip()
    required = {"ticket_id", "customer_id", "timestamp", "raw_text"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(
            "Configured ticket CSV is missing required input columns: "
            + ", ".join(missing)
        )
    if frame.empty:
        raise ValueError(f"Configured ticket CSV has no records: {path}")

    for column in ("ticket_id", "customer_id", "raw_text"):
        frame[column] = frame[column].astype("string").str.strip()

    invalid = (
        frame["ticket_id"].isna() | frame["ticket_id"].eq("")
        | frame["customer_id"].isna() | frame["customer_id"].eq("")
        | frame["raw_text"].isna() | frame["raw_text"].eq("")
    )
    frame = frame.loc[~invalid].copy().reset_index(drop=True)
    if frame.empty:
        raise ValueError("Configured ticket CSV has no valid ticket rows.")

    frame["created_at"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    # Optional source labels are preserved only as ground truth for later evaluation.
    if "issue_type" in frame.columns:
        frame["ground_truth_issue_id"] = frame["issue_type"]
    if "department" in frame.columns:
        frame["ground_truth_department_id"] = frame["department"]
    return frame


@lru_cache(maxsize=8)
def _analyze_input_csv(path_text: str, modified_ns: int, size: int) -> pd.DataFrame:
    """Analyze a CSV once per file version so Streamlit reruns stay responsive."""
    source = _read_inference_csv(path_text, modified_ns, size)
    decisions = analyze_dataframe(source, text_column="raw_text")
    taxonomy = get_taxonomy()

    rows = []
    for position, decision in enumerate(decisions):
        source_row = source.iloc[position]
        issue_id = decision.classification.issue_type
        rows.append({
            "ticket_id": str(source_row["ticket_id"]),
            "customer_id": str(source_row["customer_id"]),
            "created_at": source_row["created_at"],
            "raw_text": decision.raw_text,
            "normalized_text": decision.normalization.normalized,
            "issue_id": issue_id,
            "department_id": decision.department_id,
            "confidence": float(decision.classification.confidence),
            "priority": decision.urgency.priority,
            "needs_review": bool(decision.needs_human_review),
            "entities": ", ".join(
                f"{item.kind} {item.value}" for item in entity_views(decision.entities)
            ),
            "group_id": taxonomy.get_issue(issue_id).incident_group,
            "ground_truth_issue_id": source_row.get("ground_truth_issue_id"),
            "ground_truth_department_id": source_row.get("ground_truth_department_id"),
        })

    frame = pd.DataFrame(rows)
    frame["created_at"] = pd.to_datetime(frame["created_at"], errors="coerce")
    missing_times = int(frame["created_at"].isna().sum())
    if missing_times:
        frame.attrs["timestamp_warning"] = (
            f"{missing_times} ticket(s) have missing or invalid timestamps; "
            "they are retained but cannot appear in time-based charts."
        )
    return frame


def load_tickets() -> pd.DataFrame | None:
    """Analyze rows from an explicitly configured input CSV.

    Set SUPPORTAI_TICKETS_CSV to a CSV with ticket_id, customer_id, timestamp,
    and raw_text columns. Labels are optional and are never treated as predictions.
    """
    path = _configured_ticket_path()
    if path is None:
        return None
    if not path.is_file():
        raise FileNotFoundError(f"Configured SUPPORTAI_TICKETS_CSV does not exist: {path}")

    return _analyze_input_csv(str(path), path.stat().st_mtime_ns, path.stat().st_size).copy()


def load_incidents() -> list[IncidentView] | None:
    """Return potential incidents detected from configured ticket data."""
    tickets = load_tickets()

    if tickets is None or tickets.empty:
        return None

    from supportai.incidents import detect_incidents, explain_incident, load_incident_params

    detected = detect_incidents(tickets, **load_incident_params())

    return [
        IncidentView(
            incident_id=item.incident_id,
            group_id=item.group_id,
            window_start=item.window_start,
            window_end=item.window_end,
            unique_customers=item.unique_customers,
            raw_tickets=item.raw_tickets,
            baseline_customers=item.baseline_customers,
            severity=item.severity,
            status=item.status,
            ticket_ids=item.ticket_ids,
            explanation=tuple(explain_incident(item)),
        )
        for item in detected
    ]


def load_replay() -> ReplayData | None:
    """Build a chronological replay from configured tickets, if available."""
    tickets = load_tickets()
    if tickets is None or tickets.empty:
        return None

    from supportai.replay import run_replay

    return run_replay(tickets)


def load_evaluation() -> EvaluationReport | None:
    """Return measured synthetic held-out evaluation metrics."""

    if not is_connected("evaluation"):
        return None

    from supportai.evaluation import evaluate_classifier

    result = evaluate_classifier()

    return EvaluationReport(
        accuracy=result["accuracy"],
        macro_f1=result["macro_f1"],
        per_class=result["per_class"],
        confusion=result["confusion"],
        critical_recall=result["critical_recall"],
        critical_precision=result["critical_precision"],
        thresholds=result["thresholds"],
        incident_scenarios=pd.DataFrame(
            columns=[
                "scenario",
                "kind",
                "detected",
                "delay_minutes",
                "false_alerts",
            ]
        ),
        dataset_note=result["dataset_note"],
        extras={
            "training_samples": result["training_samples"],
            "test_samples": result["test_samples"],
            "issue_classes": result["issue_classes"],
        },
    )
