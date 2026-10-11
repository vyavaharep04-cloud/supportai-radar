"""Decides, in ONE place, whether a page gets live data, demo data, or nothing.

Rule: live backend data always wins.  Demo data is returned only when the user
switched on "Design preview", and it is always wrapped in ``Sourced(is_demo=True)``
so pages can label it.  Pages never import ``preview_data`` themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar

import pandas as pd
import streamlit as st

from dashboard import backend, preview_data
from dashboard.contracts import TICKET_COLUMNS, EvaluationReport, IncidentView, ReplayData

T = TypeVar("T")
PREVIEW_KEY = "preview_mode"


class ContractError(ValueError):
    """Backend output does not match dashboard.contracts."""


@dataclass(frozen=True)
class Sourced(Generic[T]):
    data: T
    is_demo: bool


def preview_enabled() -> bool:
    return bool(st.session_state.get(PREVIEW_KEY, False))


@st.cache_data(show_spinner=False)
def _demo_bundle() -> dict:
    taxonomy = backend.get_taxonomy()
    tickets = preview_data.tickets(taxonomy)
    replay = preview_data.replay(taxonomy, tickets)
    return {
        "tickets": tickets,
        "replay": replay,
        "incidents": preview_data.incidents(taxonomy, tickets, replay),
        "evaluation": preview_data.evaluation(taxonomy, tickets, replay),
    }


def _validate_tickets(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in TICKET_COLUMNS if c not in df.columns]
    if missing:
        raise ContractError(f"load_tickets() result is missing columns {missing}; see dashboard.contracts.TICKET_COLUMNS")
    return df


def tickets() -> Sourced[pd.DataFrame] | None:
    live = backend.load_tickets()
    if live is not None:
        return Sourced(_validate_tickets(live), False)
    return Sourced(_demo_bundle()["tickets"], True) if preview_enabled() else None


def incidents() -> Sourced[list[IncidentView]] | None:
    live = backend.load_incidents()
    if live is not None:
        return Sourced(live, False)
    return Sourced(_demo_bundle()["incidents"], True) if preview_enabled() else None


def replay() -> Sourced[ReplayData] | None:
    live = backend.load_replay()
    if live is not None:
        return Sourced(live, False)
    return Sourced(_demo_bundle()["replay"], True) if preview_enabled() else None


def evaluation() -> Sourced[EvaluationReport] | None:
    live = backend.load_evaluation()
    if live is not None:
        return Sourced(live, False)
    return Sourced(_demo_bundle()["evaluation"], True) if preview_enabled() else None


def review_threshold() -> float | None:
    """Only demo data carries a known threshold; live threshold comes from the backend later."""
    return preview_data.REVIEW_THRESHOLD if preview_enabled() and backend.load_tickets() is None else None
