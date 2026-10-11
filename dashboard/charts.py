"""Chart builders.  Pure functions: DataFrame in, Altair chart out.

Charts never compute metrics, detections or predictions; they only draw the
tables they are given.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from dashboard.compat import stretch
from dashboard.labels import NO_GROUP_LABEL
from dashboard.theme import (
    FAINT,
    MUTED,
    PRIORITY_COLORS,
    SEGMENT_COLORS,
    SERIES,
    STATE_COLORS,
    TEXT,
)

INK_COLOR = "#0B1B2B"

_FONT = "Segoe UI, system-ui, sans-serif"


def _style(chart: alt.Chart | alt.LayerChart | alt.VConcatChart, height: int | None = None):
    chart = chart.configure_view(strokeWidth=0)
    chart = chart.configure_axis(
        labelFont=_FONT, titleFont=_FONT, labelColor=MUTED, titleColor=MUTED,
        gridColor="#E8ECF2", domainColor="#C9D1DE", tickColor="#C9D1DE", labelFontSize=11, titleFontSize=11,
    )
    chart = chart.configure_legend(
        labelFont=_FONT, titleFont=_FONT, labelColor=TEXT, titleColor=MUTED, orient="top", direction="horizontal",
    )
    chart = chart.configure_title(font=_FONT, color=TEXT, fontSize=13, anchor="start")
    return chart


def show(chart, height: int | None = None, demo: bool = False) -> None:
    """Render with consistent sizing.  ``demo`` stamps the chart itself, so the
    label survives if the chart is copied out of the page."""
    if height is not None:
        chart = chart.properties(height=height)
    if demo:
        chart = chart.properties(
            title=alt.TitleParams("DEMO DATA · synthetic", anchor="end", color="#7A2E0E", fontSize=11, fontWeight="bold")
        )
    st.altair_chart(_style(chart), theme=None, **stretch())


# --------------------------------------------------------------------------


def volume_chart(tickets: pd.DataFrame, group_of: dict[str, str]) -> alt.Chart:
    """Tickets per hour, stacked by incident group (``group_of``: issue_id -> readable group name)."""
    df = tickets.copy()
    df["hour"] = pd.to_datetime(df["created_at"]).dt.floor("h")
    df["group"] = df["issue_id"].map(group_of).fillna(NO_GROUP_LABEL)
    counts = df.groupby(["hour", "group"]).size().reset_index(name="tickets")
    names = sorted(g for g in counts["group"].unique() if g != NO_GROUP_LABEL)
    domain = names + ([NO_GROUP_LABEL] if NO_GROUP_LABEL in set(counts["group"]) else [])
    colours = SERIES[: len(names)] + (["#B7C0CE"] if len(domain) > len(names) else [])
    return (
        alt.Chart(counts)
        .mark_bar(size=9)
        .encode(
            x=alt.X("hour:T", title=None, axis=alt.Axis(format="%d %b %H:%M", labelAngle=0, tickCount=8)),
            y=alt.Y("tickets:Q", title="Tickets per hour"),
            color=alt.Color("group:N", scale=alt.Scale(domain=domain, range=colours), title="Incident group"),
            tooltip=[alt.Tooltip("hour:T", format="%d %b %H:%M"), "group:N", "tickets:Q"],
        )
        .properties(height=240)
    )


def priority_chart(tickets: pd.DataFrame, order: tuple[str, ...], labels: dict[str, str]) -> alt.Chart:
    counts = tickets["priority"].value_counts().reindex(list(order), fill_value=0).reset_index()
    counts.columns = ["priority", "tickets"]
    counts["label"] = counts["priority"].map(lambda p: f"{p} {labels.get(p, '')}".strip())
    label_order = counts["label"].tolist()
    return (
        alt.Chart(counts)
        .mark_bar(cornerRadiusEnd=4)
        .encode(
            y=alt.Y("label:N", sort=label_order, title=None, axis=alt.Axis(labelLimit=140)),
            x=alt.X("tickets:Q", title="Tickets"),
            color=alt.Color(
                "priority:N",
                scale=alt.Scale(domain=list(order), range=[PRIORITY_COLORS[p] for p in order]),
                legend=None,
            ),
            tooltip=["label:N", "tickets:Q"],
        )
        .properties(height=170)
    )


def replay_chart(
    timeline: pd.DataFrame,
    tickets: pd.DataFrame,
    group_id: str,
    now: pd.Timestamp,
) -> alt.LayerChart:
    """Distinct-customer curve for one group up to ``now`` with baseline, threshold,
    a detector-state strip and the tickets plotted by synthetic segment.

    One layered chart (not a concat) so Streamlit can size it to its container.
    The strip and the ticket markers live in a band below zero on the same y axis;
    the axis only labels values >= 0.
    """
    tl = timeline[timeline["group_id"] == group_id].copy()
    x_min, x_max = tl["ts"].min(), tl["ts"].max()
    x_scale = alt.Scale(domain=[x_min.isoformat(), x_max.isoformat()])
    seen = tl[tl["ts"] <= now].copy()

    top = float(max(tl["threshold"].max(), tl["distinct_customers"].max(), 1)) * 1.12
    step = max(1, int(round(top / 5 / 5)) * 5) if top > 10 else 2
    tick_values = list(range(0, int(top) + step, step))
    y_scale = alt.Scale(domain=[-0.24 * top, top], nice=False)
    y_axis = alt.Axis(values=tick_values, title="Distinct customers (trailing window)")

    def x(field="ts", **kw):
        return alt.X(f"{field}:T", scale=x_scale, **kw)

    line = (
        alt.Chart(seen)
        .mark_line(strokeWidth=2.4, color=SERIES[0], interpolate="step-after")
        .encode(
            x=x(title=None, axis=alt.Axis(format="%d %b %H:%M", labelAngle=0, tickCount=8)),
            y=alt.Y("distinct_customers:Q", scale=y_scale, axis=y_axis),
            tooltip=[alt.Tooltip("ts:T", format="%d %b %H:%M"), "distinct_customers:Q", "baseline:Q", "threshold:Q", "state:N"],
        )
    )
    baseline = alt.Chart(tl).mark_line(strokeDash=[2, 3], strokeWidth=1.6, color=FAINT).encode(x=x(), y=alt.Y("baseline:Q", scale=y_scale))
    threshold = alt.Chart(tl).mark_line(strokeDash=[7, 4], strokeWidth=1.8, color=STATE_COLORS["alert"]).encode(x=x(), y=alt.Y("threshold:Q", scale=y_scale))
    cursor = alt.Chart(pd.DataFrame({"ts": [now]})).mark_rule(color=INK_COLOR, strokeWidth=1.6).encode(x=x())

    step_td = tl["ts"].diff().median() if len(tl) > 1 else pd.Timedelta(minutes=10)
    seen["end"] = seen["ts"] + step_td
    seen["y0"], seen["y1"] = -0.075 * top, -0.025 * top
    strip = (
        alt.Chart(seen)
        .mark_rect()
        .encode(
            x=x(), x2="end:T", y=alt.Y("y0:Q", scale=y_scale), y2="y1:Q",
            color=alt.Color("state:N", scale=alt.Scale(domain=list(STATE_COLORS), range=list(STATE_COLORS.values())), title="Detector state"),
            tooltip=[alt.Tooltip("ts:T", format="%d %b %H:%M"), "state:N"],
        )
    )

    pts = tickets[tickets["group_id"] == group_id].copy()
    pts["created_at"] = pd.to_datetime(pts["created_at"])
    pts = pts[pts["created_at"] <= now]
    pts["lane"] = -0.16 * top
    seg_domain = list(SEGMENT_COLORS)
    ticket_pts = (
        alt.Chart(pts)
        .mark_point(filled=True, size=40, opacity=0.8)
        .encode(
            x=x("created_at"), y=alt.Y("lane:Q", scale=y_scale),
            color=alt.Color("segment:N", scale=alt.Scale(domain=seg_domain, range=[SEGMENT_COLORS[k] for k in seg_domain]), title="Ticket (synthetic label)"),
            shape=alt.Shape("segment:N", scale=alt.Scale(domain=seg_domain, range=["circle", "diamond", "triangle-up"]), title="Ticket (synthetic label)"),
            tooltip=["ticket_id:N", "customer_id:N", alt.Tooltip("created_at:T", format="%d %b %H:%M"), "segment:N"],
        )
    )
    return (
        alt.layer(baseline, threshold, line, cursor, strip, ticket_pts)
        .resolve_scale(color="independent", shape="independent")
        .properties(height=320)
    )


def confusion_chart(confusion: pd.DataFrame, labels: dict[str, str]) -> alt.LayerChart:
    long = confusion.reset_index().melt(id_vars=confusion.index.name or "index", var_name="predicted", value_name="count")
    long = long.rename(columns={confusion.index.name or "index": "true"})
    order = list(confusion.index)
    long["true_label"] = long["true"].map(lambda i: labels.get(i, i))
    long["pred_label"] = long["predicted"].map(lambda i: labels.get(i, i))
    label_order = [labels.get(i, i) for i in order]
    base = alt.Chart(long).encode(
        x=alt.X("pred_label:N", sort=label_order, title="Predicted", axis=alt.Axis(labelAngle=-40, labelLimit=150)),
        y=alt.Y("true_label:N", sort=label_order, title="True", axis=alt.Axis(labelLimit=150)),
    )
    heat = base.mark_rect(cornerRadius=2).encode(
        color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), legend=None),
        tooltip=["true_label:N", "pred_label:N", "count:Q"],
    )
    text = base.mark_text(fontSize=11, font=_FONT).encode(
        text=alt.condition(alt.datum.count > 0, alt.Text("count:Q"), alt.value("")),
        color=alt.condition(alt.datum.count > long["count"].max() * 0.5, alt.value("white"), alt.value(TEXT)),
    )
    return (heat + text).properties(height=380)


def threshold_chart(thresholds: pd.DataFrame) -> alt.Chart:
    long = thresholds.melt(id_vars="threshold", value_vars=["review_rate", "auto_accuracy"], var_name="metric", value_name="value")
    long["metric"] = long["metric"].map({"review_rate": "Sent to human review", "auto_accuracy": "Accuracy of auto-routed"})
    return (
        alt.Chart(long)
        .mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2.4)
        .encode(
            x=alt.X("threshold:Q", title="Confidence threshold for auto-routing", axis=alt.Axis(format=".0%")),
            y=alt.Y("value:Q", title=None, axis=alt.Axis(format=".0%"), scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("metric:N", scale=alt.Scale(range=[SERIES[2], SERIES[0]]), title=None),
            strokeDash=alt.StrokeDash("metric:N", legend=None),
            tooltip=["metric:N", alt.Tooltip("threshold:Q", format=".0%"), alt.Tooltip("value:Q", format=".1%")],
        )
        .properties(height=260)
    )
