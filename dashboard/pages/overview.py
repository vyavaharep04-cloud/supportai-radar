"""A. Operations Overview."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import backend, charts, components as ui, data_access
from dashboard.compat import stretch
from dashboard.contracts import PRIORITIES, PRIORITY_LABELS, TICKET_COLUMNS, WORKFLOW_STAGES
from dashboard.labels import department_labels, group_label, incident_groups, issue_labels
from supportai.config import ConfigError

TICKET_INTERFACE = (
    "# dashboard.backend.load_tickets() must return a pandas.DataFrame with these columns:\n"
    + "\n".join(f"#   {c}" for c in TICKET_COLUMNS)
    + "\n# (priority is one of P1..P4; needs_review is bool; see dashboard/contracts.py)"
)


def _pipeline_status() -> None:
    caps = backend.capabilities()
    stage_status = []
    for stage in WORKFLOW_STAGES:
        members = [c for c in caps if c.stage == stage]
        stage_status.append((stage, sum(c.connected for c in members), len(members)))
    ui.render('<div class="sr-card-title" style="margin-top:.2rem">Pipeline status</div>')
    ui.workflow_strip(stage_status)


def _taxonomy_card(tx) -> str:
    depts = department_labels(tx)
    per_dept: dict[str, int] = {}
    for issue in tx.issue_types:
        per_dept[issue.department] = per_dept.get(issue.department, 0) + 1
    groups = sorted({i.incident_group for i in tx.issue_types if i.incident_group})
    body = ui.kv_list(
        [("Issue types", str(len(tx.issue_types))), ("Departments", str(len(tx.departments))),
         ("Incident groups", str(len(groups))), ("Taxonomy version", str(tx.version))]
    )
    body += '<div class="sr-label" style="margin-top:.8rem">Issues per department</div>'
    body += ui.kv_list([(depts[d], str(n)) for d, n in per_dept.items()])
    body += '<div class="sr-label" style="margin-top:.8rem">Incident groups</div>'
    body += "".join(ui.badge(group_label(g), "brand") + " " for g in groups)
    return ui.card("Taxonomy in effect · LIVE", body)


def render() -> None:
    try:
        tx = backend.get_taxonomy()
    except ConfigError as exc:
        ui.callout(f"Could not load config/taxonomy.yaml: {exc}", "danger")
        return

    tk = data_access.tickets()
    inc = data_access.incidents()
    demo = any(s is not None and s.is_demo for s in (tk, inc))
    ui.page_header(
        "Operations Overview",
        "UNDERSTAND → DECIDE → ASSIST → DETECT → ACT · Ticket triage and potential-incident monitoring.",
        badges=[ui.provenance_badge("demo" if demo else "live" if tk else "off")],
    )
    _pipeline_status()

    if tk is None:
        left, right = st.columns([2, 1], gap="large")
        with left:
            ui.empty_state(
                "No ticket data connected yet",
                "Ticket volume, priority mix, the review queue and incident status appear here once the "
                "ingestion, classification and urgency modules are connected. Nothing is simulated in this view.",
                interface=TICKET_INTERFACE, icon="◌",
            )
            ui.mini("To preview the intended design with clearly labelled synthetic data, switch on “Design preview” in the sidebar.")
        with right:
            ui.render(_taxonomy_card(tx))
        return

    if tk.is_demo:
        ui.demo_banner()
    df: pd.DataFrame = tk.data
    labels, depts = issue_labels(tx), department_labels(tx)

    alerts = [i for i in (inc.data if inc else []) if i.status == "alert"]
    watches = [i for i in (inc.data if inc else []) if i.status == "watch"]
    p12 = int(df["priority"].isin(["P1", "P2"]).sum())
    review = df[df["needs_review"]]
    st.write("")
    ui.kpi_row([
        ("Tickets", f"{len(df):,}", f"{df['customer_id'].nunique():,} distinct customers"),
        ("High priority (P1+P2)", f"{p12:,}", f"{p12 / len(df):.0%} of tickets", "warning" if p12 else None),
        ("Human-review queue", f"{len(review):,}", f"{len(review) / len(df):.0%} routed to people"),
        ("Potential incidents", f"{len(alerts)} alert · {len(watches)} watch", "inferred from ticket volume",
         "danger" if alerts else "success"),
    ])

    ui.section("Ticket volume and priority")
    c1, c2 = st.columns([2, 1], gap="large")
    with c1:
        charts.show(charts.volume_chart(df, {i: group_label(g) for i, g in incident_groups(tx).items()}), demo=tk.is_demo)
    with c2:
        charts.show(charts.priority_chart(df, PRIORITIES, PRIORITY_LABELS), demo=tk.is_demo)

    ui.section("What needs attention")
    q, i_col, r = st.columns([1.15, 1, 1.15], gap="large")
    with q:
        ui.render('<div class="sr-card-title">Human-review queue · lowest confidence first</div>')
        top = review.sort_values("confidence").head(6)
        rows = "".join(
            f'<div class="sr-kv"><span class="k">{ui.esc(t.ticket_id)} · {ui.esc(labels.get(t.issue_id, t.issue_id))}</span>'
            f'<span class="v">{t.confidence:.0%}</span></div>' for t in top.itertuples()
        )
        ui.render(ui.card(None, rows or '<span class="sr-mini">Queue is empty.</span>'))
    with i_col:
        ui.render('<div class="sr-card-title">Potential incidents</div>')
        if inc is None or not inc.data:
            ui.render(ui.card(None, '<span class="sr-mini">No potential incidents detected.</span>'))
        else:
            rows = "".join(
                f'<div class="sr-kv"><span class="k">{ui.esc(group_label(i.group_id))}<br><span class="sr-mini">{i.unique_customers} customers · {i.ratio:.1f}x baseline</span></span>'
                f'<span class="v">{ui.state_badge(i.status)}</span></div>' for i in inc.data[:4]
            )
            ui.render(ui.card(None, rows))
    with r:
        ui.render('<div class="sr-card-title">Recent activity</div>')
        recent = df.sort_values("created_at", ascending=False).head(6)
        rows = "".join(
            f'<div class="sr-kv"><span class="k">{ui.esc(t.created_at.strftime("%d %b %H:%M"))} · {ui.esc(t.ticket_id)}</span>'
            f'<span class="v">{ui.priority_badge(t.priority)}</span></div>' for t in recent.itertuples()
        )
        ui.render(ui.card(None, rows))

    ui.section("Configuration")
    ui.render(_taxonomy_card(tx))
    ui.mini("Alerts describe patterns in support tickets. They do not confirm an outage or its root cause.")
