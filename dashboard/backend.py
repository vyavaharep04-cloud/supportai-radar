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
    EntityView,
    EvaluationReport,
    IncidentView,
    ReplayData,
    TicketAnalysis,
)
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
    """Analyse one message with whatever backend stages exist.

    Connected today: normalization and entity extraction.  Every other stage
    stays ``None`` so the page shows it as "not connected".
    """
    return TicketAnalysis(
        raw_text=text,
        normalization=get_default_normalizer().explain(text),
        entities=extract_entity_views(text),  # reads the RAW text, as designed
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
