"""E. Incident Replay."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts, components as ui, data_access
from dashboard.labels import group_label

INTERFACE = (
    "# dashboard.backend.load_replay() -> dashboard.contracts.ReplayData\n"
    "#   tickets : TICKET_COLUMNS + group_id + segment ('normal' | 'decoy' | 'planted')\n"
    "#   timeline: ts, group_id, distinct_customers, raw_tickets, baseline, threshold,\n"
    "#             state ('normal' | 'watch' | 'alert')  -- one row per (tick, group)\n"
    "# 'segment' is the synthetic generator's ground truth, available only in simulated scenarios."
)
KEY_TS, KEY_GROUP = "replay_ts", "replay_group"


def _set_index(ticks: list[pd.Timestamp], index: int) -> None:
    st.session_state[KEY_TS] = ticks[max(0, min(len(ticks) - 1, index))]


def _legend() -> str:
    return ui.card("Reading the replay", "".join([
        '<div class="sr-rule">● <b>Normal</b> · ordinary background tickets</div>',
        '<div class="sr-rule">◆ <b>Decoy</b> · designed to look like an incident but should not alert (one customer repeating; a mild bump)</div>',
        '<div class="sr-rule">▲ <b>Planted</b> · the synthetic incident the detector should find</div>',
        '<div class="sr-rule">Dashed red line = alert threshold · dotted grey = baseline · strip = detector state</div>',
    ]))


def render() -> None:
    src = data_access.replay()
    ui.page_header("Incident Replay", "Step through a simulated timeline and watch the detector state change as tickets arrive.",
                   badges=[ui.provenance_badge("off" if src is None else "demo" if src.is_demo else "live")])
    if src is not None and src.is_demo:
        ui.callout("Replay uses a simulated scenario with known planted incidents and decoys. It illustrates detector behaviour; it does not describe real outages.", "info")
    elif src is not None:
        ui.callout("Live replay uses configured ticket data and the real incident detector. Ticket markers are observed records, not planted/decoy labels; alerts indicate potential incidents, not confirmed outages.", "info")
    if src is None:
        ui.empty_state("No replay scenario connected yet",
                       "The replay needs the incident detector and a simulated scenario. No timeline is invented here.", interface=INTERFACE, icon="▶")
        return
    if src.is_demo:
        ui.demo_banner("Scenario and detector states are scripted for design preview.")
    data = src.data
    tl, tickets = data.timeline, data.tickets
    ticks = sorted(tl["ts"].unique())
    ticks = [pd.Timestamp(t) for t in ticks]
    groups = sorted(tl["group_id"].unique())
    if KEY_TS not in st.session_state or st.session_state[KEY_TS] not in ticks:
        st.session_state[KEY_TS] = ticks[0]
    st.session_state.setdefault(KEY_GROUP, groups[0])

    ui.mini(data.description)
    c1, c2 = st.columns([1.2, 2.2], gap="large")
    with c1:
        st.radio("Incident group", groups, key=KEY_GROUP, format_func=group_label)
    with c2:
        st.select_slider("Replay position", options=ticks, key=KEY_TS, format_func=lambda t: t.strftime("%d %b %H:%M"))
    group = st.session_state[KEY_GROUP]
    now = st.session_state[KEY_TS]
    idx = ticks.index(now)
    gtl = tl[tl["group_id"] == group]
    first_alert = gtl.loc[gtl["state"] == "alert", "ts"].min()

    b = st.columns(5)
    with b[0]:
        st.button("⏮ Start", on_click=_set_index, args=(ticks, 0), key="rp_start")
    with b[1]:
        st.button("◀ Back 1 h", on_click=_set_index, args=(ticks, idx - 6), key="rp_back")
    with b[2]:
        st.button("Forward 1 h ▶", on_click=_set_index, args=(ticks, idx + 6), key="rp_fwd")
    with b[3]:
        st.button("Jump to first alert", on_click=_set_index, args=(ticks, ticks.index(first_alert) if pd.notna(first_alert) else idx), key="rp_alert", disabled=bool(pd.isna(first_alert)))
    with b[4]:
        st.button("⏭ End", on_click=_set_index, args=(ticks, len(ticks) - 1), key="rp_end")

    row = gtl[gtl["ts"] == now].iloc[0]
    seen = tickets[(tickets["group_id"] == group) & (pd.to_datetime(tickets["created_at"]) <= now)]
    ui.kpi_row([
        ("Detector state", str(row["state"]).upper(), "at the selected time", {"alert": "danger", "watch": "warning"}.get(row["state"], "success")),
        ("Distinct customers", int(row["distinct_customers"]), f"in the last {data.window_minutes} minutes"),
        ("Baseline / threshold", f"{row['baseline']:.1f} / {row['threshold']:.0f}", "customers"),
        ("Raw tickets in window", int(row["raw_tickets"]), "includes repeat submissions"),
        ("Tickets so far", len(seen), group_label(group)),
    ])
    st.write("")
    charts.show(charts.replay_chart(tl, tickets, group, now), demo=src.is_demo)

    left, right = st.columns([1, 1], gap="large")
    with left:
        ui.render(_legend())
    with right:
        seq = gtl[gtl["ts"] <= now].reset_index(drop=True)
        changes = seq[seq["state"] != seq["state"].shift()].iloc[1:] if len(seq) else seq
        lines = "".join(
            f'<div class="sr-rule">{ui.esc(r.ts.strftime("%d %b %H:%M"))} {ui.state_badge(r.state)} <span class="sr-mini">{int(r.distinct_customers)} distinct customers</span></div>'
            for r in changes.itertuples()
        ) or '<span class="sr-mini">No state changes yet: the detector has stayed NORMAL.</span>'
        ui.render(ui.card("Alert transitions up to this point", lines))
