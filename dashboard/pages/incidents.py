"""D. Incident Center."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard import backend, charts, components as ui, data_access
from dashboard.compat import stretch
from dashboard.contracts import PRIORITY_LABELS
from dashboard.labels import department_labels, group_label, issue_labels
from dashboard.theme import SERIES
from supportai.config import ConfigError

INTERFACE = (
    "# dashboard.backend.load_incidents() -> list[dashboard.contracts.IncidentView]\n"
    "# IncidentView: incident_id, group_id (taxonomy incident_group), window_start, window_end,\n"
    "#   unique_customers, raw_tickets, baseline_customers, severity, status, ticket_ids, explanation\n"
    "# Detection counts DISTINCT customers in fixed windows and compares them\n"
    "# with preceding consecutive windows. 'other' tickets never participate."
)
DISCLAIMER = ("Potential incidents are inferred from ticket volume across distinct customers. "
              "They do not confirm an outage or establish its root cause.")


def _how_it_works() -> str:
    steps = [
        ("1 · Group", "Each ticket's issue type maps to an incident group in the taxonomy. “Other” tickets are excluded."),
        ("2 · Count customers", "Within each fixed time window, count distinct customers, not tickets, so one person repeating a complaint cannot raise an alert."),
        ("3 · Compare to baseline", "The count is compared with the immediately preceding consecutive windows; empty windows count as zero."),
        ("4 · Explain", "Each alert lists the evidence: customers, window, baseline ratio and linked tickets."),
    ]
    return ui.card("How detection works", "".join(
        f'<div class="sr-kv" style="display:block"><div class="v" style="text-align:left">{ui.esc(t)}</div><div class="sr-mini">{ui.esc(b)}</div></div>' for t, b in steps))


def _card(inc) -> str:
    body = ui.kv_list([
        ("Distinct customers", f"{inc.unique_customers}"), ("Baseline", f"{inc.baseline_customers:.1f}"),
        ("Ratio to baseline", f"{inc.ratio:.1f}x"), ("Raw tickets", f"{inc.raw_tickets}"),
    ])
    head = f'<div style="display:flex;justify-content:space-between;gap:.5rem;margin-bottom:.4rem"><b>{ui.esc(group_label(inc.group_id))}</b>{ui.state_badge(inc.status)}</div>'
    sub = f'<div class="sr-mini" style="margin-bottom:.4rem">{ui.esc(inc.incident_id)} · {ui.esc(inc.window_start.strftime("%d %b %H:%M"))} → {ui.esc(inc.window_end.strftime("%H:%M"))}</div>'
    return ui.card(None, head + sub + body + '<div style="margin-top:.5rem">' + ui.severity_badge(inc.severity) + "</div>")


def _detail(inc, tickets: pd.DataFrame | None, labels, depts, demo: bool = False) -> None:
    ui.section(f"{inc.incident_id} · {group_label(inc.group_id)}", "Evidence for this potential incident.")
    ui.kpi_row([
        ("Distinct customers", inc.unique_customers, f"{inc.raw_tickets} raw tickets"),
        ("Baseline", f"{inc.baseline_customers:.1f}", "historical, same window length"),
        ("Ratio", f"{inc.ratio:.1f}x", "customers ÷ baseline", "danger" if inc.status == "alert" else "warning"),
        ("Window", f"{(inc.window_end - inc.window_start).total_seconds() / 3600:.1f} h", f"{inc.window_start:%d %b %H:%M} → {inc.window_end:%H:%M}"),
    ])
    st.write("")
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        lines = "".join(f'<div class="sr-rule">{ui.esc(line)}</div>' for line in inc.explanation) or '<span class="sr-mini">No explanation provided.</span>'
        ui.render(ui.card("Incident explanation", lines))
        st.write("")
        ui.render(ui.card("Status", f'{ui.state_badge(inc.status)} {ui.severity_badge(inc.severity)}'
                          '<div class="sr-mini" style="margin-top:.5rem">Status is shown for review only; this prototype does not store workflow state.</div>'))
    with right:
        linked = None if tickets is None else tickets[tickets["ticket_id"].isin(inc.ticket_ids)]
        if linked is None or linked.empty:
            ui.empty_state("No linked tickets available", "Linked tickets appear when the ticket table includes this incident's ticket ids.", icon="∅")
            return
        mix = linked["issue_id"].map(lambda i: labels.get(i, i)).value_counts().reset_index()
        mix.columns = ["issue", "tickets"]
        chart = alt.Chart(mix).mark_bar(cornerRadiusEnd=4, color=SERIES[1]).encode(
            y=alt.Y("issue:N", sort="-x", title=None, axis=alt.Axis(labelLimit=200)), x=alt.X("tickets:Q", title="Linked tickets"),
            tooltip=["issue:N", "tickets:Q"]).properties(height=max(90, 34 * len(mix)))
        charts.show(chart, demo=demo)
        view = pd.DataFrame({
            "Ticket": linked["ticket_id"], "Time": linked["created_at"].dt.strftime("%d %b %H:%M"), "Customer": linked["customer_id"],
            "Issue": linked["issue_id"].map(lambda i: labels.get(i, i)), "Priority": linked["priority"].map(lambda p: f"{p} {PRIORITY_LABELS.get(p, '')}".strip()),
            "Message": linked["raw_text"],
        }).sort_values("Time")
        st.dataframe(view, hide_index=True, height=260, **stretch())


def render() -> None:
    try:
        tx = backend.get_taxonomy()
    except ConfigError as exc:
        ui.callout(f"Could not load config/taxonomy.yaml: {exc}", "danger")
        return
    src = data_access.incidents()
    ui.page_header("Incident Center", "Potential incidents inferred from unusual numbers of distinct customers reporting related problems.",
                   badges=[ui.provenance_badge("off" if src is None else "demo" if src.is_demo else "live")])
    ui.callout(DISCLAIMER, "info")
    if src is None:
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            ui.empty_state("No incident detector connected yet",
                           "Alerts, evidence and explanations appear here once the incident detection module is connected. "
                           "No alerts are simulated in this view.", interface=INTERFACE, icon="◎")
        with right:
            ui.render(_how_it_works())
        return
    if src.is_demo:
        ui.demo_banner("Incidents shown come from a scripted illustrative detector over synthetic tickets.")
    incidents = src.data
    if not incidents:
        ui.empty_state("No potential incidents met the alert thresholds", "No evaluated window crossed the detector thresholds. Sparse data or insufficient history can limit detection; this does not prove that no incident occurred.", icon="✓")
        return
    tk = data_access.tickets()
    labels, depts = issue_labels(tx), department_labels(tx)

    alerts = sum(i.status == "alert" for i in incidents)
    ui.kpi_row([("Active alerts", alerts, "above alert level", "danger" if alerts else "success"),
                ("Watch", sum(i.status == "watch" for i in incidents), "elevated, below alert level", "warning"),
                ("Peak distinct customers", max(i.unique_customers for i in incidents), "in a single window"),
                ("Highest ratio", f"{max(i.ratio for i in incidents):.1f}x", "versus baseline")])
    ui.section("Potential incidents", "Most recent first.")
    cols = st.columns(min(3, len(incidents)), gap="medium")
    for k, inc in enumerate(incidents[:6]):
        with cols[k % len(cols)]:
            ui.render(_card(inc))
            st.write("")
    ids = [i.incident_id for i in incidents]
    pick = st.selectbox("Investigate", ids, format_func=lambda i: next(f"{x.incident_id} · {group_label(x.group_id)} · {x.status.upper()}" for x in incidents if x.incident_id == i))
    _detail(next(i for i in incidents if i.incident_id == pick), tk.data if tk else None, labels, depts, src.is_demo)
