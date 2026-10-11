"""B. Ticket Explorer."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import backend, components as ui, data_access
from dashboard.compat import stretch
from dashboard.contracts import PRIORITIES, PRIORITY_LABELS, TICKET_COLUMNS
from dashboard.labels import department_labels, issue_labels
from supportai.config import ConfigError

INTERFACE = (
    "# dashboard.backend.load_tickets() -> pandas.DataFrame\n"
    "# columns: " + ", ".join(TICKET_COLUMNS) + "\n"
    "# One row per ticket. normalized_text should come from the real Hinglish normalizer;\n"
    "# issue_id / department_id must be taxonomy ids; needs_review routes the ticket to people."
)


def _filtered(df: pd.DataFrame, labels: dict[str, str], depts: dict[str, str]) -> pd.DataFrame:
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns([2.2, 1.6, 1.6, 1.3])
        query = c1.text_input("Search", placeholder="Ticket ID, customer, or words in the message", key="ex_query")
        issues = c2.multiselect("Issue type", list(labels), format_func=labels.get, key="ex_issue")
        dept_ids = c3.multiselect("Department", list(depts), format_func=depts.get, key="ex_dept")
        prios = c4.multiselect("Priority", list(PRIORITIES), format_func=lambda p: f"{p} {PRIORITY_LABELS[p]}", key="ex_prio")
        only_review = st.checkbox("Needs human review only", key="ex_review")
    out = df
    if query.strip():
        q = query.strip().lower()
        hay = (out["ticket_id"] + " " + out["customer_id"] + " " + out["raw_text"] + " " + out["normalized_text"]).str.lower()
        out = out[hay.str.contains(q, regex=False)]
    if issues:
        out = out[out["issue_id"].isin(issues)]
    if dept_ids:
        out = out[out["department_id"].isin(dept_ids)]
    if prios:
        out = out[out["priority"].isin(prios)]
    if only_review:
        out = out[out["needs_review"]]
    return out.sort_values("created_at", ascending=False)


def _table(out: pd.DataFrame, labels, depts) -> pd.Series | None:
    view = pd.DataFrame({
        "Ticket": out["ticket_id"],
        "Created": out["created_at"].dt.strftime("%d %b %H:%M"),
        "Customer": out["customer_id"],
        "Issue": out["issue_id"].map(lambda i: labels.get(i, i)),
        "Department": out["department_id"].map(lambda d: depts.get(d, d)),
        "Priority": out["priority"].map(lambda p: f"{p} {PRIORITY_LABELS.get(p, '')}".strip()),
        "Confidence": out["confidence"],
        "Review": out["needs_review"].map({True: "Human review", False: "Auto"}),
        "Message": out["raw_text"],
    })
    event = st.dataframe(
        view, hide_index=True, height=340, on_select="rerun", selection_mode="single-row", key="ex_table",
        column_config={
            "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0, format="%.2f"),
            "Message": st.column_config.TextColumn("Message", width="large"),
        },
        **stretch(),
    )
    rows = event.selection.rows if event and event.selection else []
    return out.iloc[rows[0]] if rows else None


def _detail(row: pd.Series, labels, depts, demo: bool, threshold: float | None) -> None:
    analysis = backend.analyze_ticket(row["raw_text"])  # live normalizer
    rules = analysis.normalization.applied_rules if analysis.normalization else ()
    ui.section(f"Ticket {row['ticket_id']}", "Select another row above to inspect a different ticket.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        ui.render(ui.text_block("Original message", row["raw_text"]))
        st.write("")
        ui.render(
            ui.text_block("Normalized text (what the classifier sees) · LIVE normalizer", analysis.normalization.normalized if analysis.normalization else "", mono=True)
        )
        if rules:
            ui.render('<div class="sr-label" style="margin-top:.8rem">Lexicon rules applied</div>' + "".join(
                f'<div class="sr-rule"><code>{ui.esc(r.matched)}</code><span class="arrow">→</span><code>{ui.esc(r.replacement)}</code>'
                f'<span class="rid">{ui.esc(r.rule_id)}</span></div>' for r in rules
            ))
        if analysis.entities:
            ui.render('<div class="sr-label" style="margin-top:.8rem">Extracted entities · LIVE extractor</div>'
                      + ui.chips((e.kind, e.value) for e in analysis.entities))
        elif analysis.entities is not None:
            ui.render('<div class="sr-mini" style="margin-top:.8rem">No entities found in the original text.</div>')
    with right:
        status = "demo" if demo else "live"
        body = ui.kv_list([
            ("Predicted issue", labels.get(row["issue_id"], row["issue_id"])),
            ("Department", depts.get(row["department_id"], row["department_id"])),
            ("Customer", row["customer_id"]),
            ("Created", row["created_at"].strftime("%d %b %Y %H:%M")),
        ])
        body += '<div style="margin-top:.7rem">' + ui.priority_badge(row["priority"]) + " " + ui.review_badge(bool(row["needs_review"])) + "</div>"
        body += '<div style="margin-top:.8rem">' + ui.confidence_bar(float(row["confidence"]), threshold) + "</div>"
        ui.render(ui.card("Prediction · " + ("DEMO DATA" if demo else "LIVE"), body))
        if bool(row["needs_review"]):
            st.write("")
            ui.callout("Routed to a person: confidence is low or the issue could not be matched to a supported type.", "warning")


def render() -> None:
    try:
        tx = backend.get_taxonomy()
    except ConfigError as exc:
        ui.callout(f"Could not load config/taxonomy.yaml: {exc}", "danger")
        return
    src = data_access.tickets()
    ui.page_header(
        "Ticket Explorer",
        "Search, filter and inspect tickets with their original text, normalized text, prediction and routing.",
        badges=[ui.provenance_badge("off" if src is None else "demo" if src.is_demo else "live")],
    )
    if src is None:
        ui.empty_state(
            "No tickets to explore yet",
            "This page lists real tickets once CSV ingestion and the classifier are connected. "
            "To try the live analysis of a single message today, open Ticket Analysis.",
            interface=INTERFACE, icon="⌕",
        )
        return
    if src.is_demo:
        ui.demo_banner("Message text and predictions are synthetic; normalized text comes from the real normalizer.")
    labels, depts = issue_labels(tx), department_labels(tx)
    df = src.data
    filtered = _filtered(df, labels, depts)
    ui.mini(f"{len(filtered):,} of {len(df):,} tickets match.")
    if filtered.empty:
        ui.empty_state("No tickets match these filters", "Clear a filter or broaden the search.", icon="∅")
        return
    selected = _table(filtered, labels, depts)
    row = selected if selected is not None else filtered.iloc[0]
    if selected is None:
        ui.mini("Showing the most recent ticket. Click a row to inspect another.")
    _detail(row, labels, depts, src.is_demo, data_access.review_threshold())
