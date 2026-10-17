"""Build a chronological replay from configured, locally analyzed tickets."""
from __future__ import annotations

import math
import pandas as pd
from dashboard.contracts import ReplayData
from supportai.incidents import detect_incidents, load_incident_params


def run_replay(tickets: pd.DataFrame) -> ReplayData | None:
    """Return a replay using the same configured detector as the live incident view."""
    params = load_incident_params()
    window_minutes = int(params["window_minutes"])
    required = {"ticket_id", "customer_id", "created_at", "group_id"}
    if tickets.empty or not required.issubset(tickets.columns):
        return None

    data = tickets.copy()
    data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce", utc=True)
    for col in ("ticket_id", "customer_id", "group_id"):
        data[col] = data[col].astype("string").str.strip()
    data = data.dropna(subset=list(required))
    data = data[
        data["ticket_id"].ne("") & data["customer_id"].ne("")
        & data["group_id"].ne("") & data["group_id"].ne("other")
    ].copy()
    if data.empty:
        return None

    window = pd.Timedelta(minutes=window_minutes)
    data["window_start"] = data["created_at"].dt.floor(f"{window_minutes}min")
    incidents = detect_incidents(data, **params)
    states = {(item.group_id, pd.Timestamp(item.window_start)): item.status for item in incidents}

    rows = []
    grouped_counts = {}
    first_global = data["window_start"].min()
    last_global = data["window_start"].max()
    all_starts = pd.date_range(first_global, last_global, freq=window)
    for group_id, group in data.groupby("group_id", sort=True):
        grouped_counts[str(group_id)] = group.groupby("window_start").agg(
            distinct_customers=("customer_id", "nunique"),
            raw_tickets=("ticket_id", "size"),
        ).sort_index()

    # Give every group the same replay ticks so the page's shared time slider
    # always has a corresponding row for whichever group is selected.
    for group_id, counts in grouped_counts.items():
        first = counts.index.min()
        for start in all_starts:
            history_index = pd.date_range(
                end=start - window, periods=int(params["min_baseline_windows"]), freq=window
            )
            eligible = history_index[0] >= first
            if eligible:
                history = counts["distinct_customers"].reindex(history_index, fill_value=0)
                baseline = float(history.mean())
            else:
                baseline = float("nan")
            current = counts.loc[start] if start in counts.index else None
            distinct = int(current["distinct_customers"]) if current is not None else 0
            raw = int(current["raw_tickets"]) if current is not None else 0
            state = states.get((group_id, pd.Timestamp(start)), "normal")
            threshold = (
                max(float(params["min_customers"]), float(params["alert_ratio"]) * baseline)
                if math.isfinite(baseline) else float(params["min_customers"])
            )
            rows.append({
                "ts": start + window, "group_id": group_id,
                "distinct_customers": distinct, "raw_tickets": raw,
                "baseline": baseline, "threshold": threshold, "state": state,
            })

    timeline = pd.DataFrame(rows)
    if timeline.empty:
        return None
    replay_tickets = data.drop(columns=["window_start"]).copy()
    replay_tickets["segment"] = "observed"
    return ReplayData(
        tickets=replay_tickets,
        timeline=timeline.sort_values(["ts", "group_id"]).reset_index(drop=True),
        window_minutes=window_minutes,
        description=(
            "Live replay of configured ticket data using the same incident settings as "
            "the live incident view. Ticket markers are observed records, not synthetic labels."
        ),
    )
