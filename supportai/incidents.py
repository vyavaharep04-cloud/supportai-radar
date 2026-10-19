"""Detect potential support incidents from ticket-volume anomalies."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd
from supportai.config import CONFIG_DIR, ConfigError, read_yaml

@dataclass(frozen=True)
class Incident:
    """A potential incident inferred from unusual ticket activity."""

    incident_id: str
    group_id: str
    window_start: pd.Timestamp
    window_end: pd.Timestamp
    unique_customers: int
    raw_tickets: int
    baseline_customers: float
    severity: str
    status: str
    ticket_ids: tuple[str, ...]
    explanation: tuple[str, ...]

    @property
    def ratio(self) -> float:
        """Current distinct-customer count divided by the baseline."""
        if self.baseline_customers <= 0:
            return math.inf if self.unique_customers else 0.0
        return self.unique_customers / self.baseline_customers


def detect_incidents(
    tickets: pd.DataFrame,
    *,
    window_minutes: int = 30,
    min_customers: int = 5,
    watch_ratio: float = 2.0,
    alert_ratio: float = 3.0,
    min_baseline_windows: int = 3,
) -> list[Incident]:
    """Detect unusual activity using distinct-customer counts.

    Required columns: ticket_id, customer_id, created_at, group_id.

    Windows are fixed, non-overlapping UTC windows. Naive timestamps are
    interpreted as UTC. The baseline uses the immediately preceding
    consecutive windows, including zero-ticket windows, and never includes
    the current window. Rows without valid timestamps, IDs, or groups and
    the taxonomy's `other` group are ignored.
    """
    if isinstance(window_minutes, bool) or not isinstance(window_minutes, int) or window_minutes <= 0:
        raise ValueError("window_minutes must be a positive integer.")
    if isinstance(min_customers, bool) or not isinstance(min_customers, int) or min_customers < 1:
        raise ValueError("min_customers must be a positive integer.")
    if isinstance(min_baseline_windows, bool) or not isinstance(min_baseline_windows, int) or min_baseline_windows < 1:
        raise ValueError("min_baseline_windows must be a positive integer.")
    if (
        not math.isfinite(watch_ratio)
        or not math.isfinite(alert_ratio)
        or watch_ratio < 1
        or alert_ratio < watch_ratio
    ):
        raise ValueError("Require finite ratios with 1 <= watch_ratio <= alert_ratio.")

    required = ("ticket_id", "customer_id", "created_at", "group_id")
    missing = set(required) - set(tickets.columns)
    if missing:
        raise ValueError(
            "Missing incident input columns: " + ", ".join(sorted(missing))
        )
    if tickets.empty:
        return []

    data = tickets.loc[:, list(required)].copy()
    # Normalizing to UTC avoids mixed-offset object dtype and comparison errors.
    data["created_at"] = pd.to_datetime(data["created_at"], errors="coerce", utc=True)
    for column in ("ticket_id", "customer_id", "group_id"):
        data[column] = data[column].astype("string").str.strip()

    data = data.dropna(subset=list(required))
    data = data[
        data["ticket_id"].ne("")
        & data["customer_id"].ne("")
        & data["group_id"].ne("")
        & data["group_id"].ne("other")
    ].copy()
    if data.empty:
        return []

    window = pd.Timedelta(minutes=window_minutes)
    data["window_start"] = data["created_at"].dt.floor(f"{window_minutes}min")
    incidents: list[Incident] = []

    for group_id, group in data.groupby("group_id", sort=True):
        windows = (
            group.groupby("window_start")
            .agg(
                unique_customers=("customer_id", "nunique"),
                raw_tickets=("ticket_id", "size"),
            )
            .sort_index()
        )
        first_window = windows.index.min()

        for start, current in windows.iterrows():
            # Use the immediately preceding consecutive windows. Missing
            # windows count as zero instead of silently extending the baseline.
            history_index = pd.date_range(
                end=start - window,
                periods=min_baseline_windows,
                freq=window,
            )
            if history_index[0] < first_window:
                continue
            history = windows["unique_customers"].reindex(
                history_index, fill_value=0
            )
            baseline = float(history.mean())
            customer_count = int(current["unique_customers"])
            if customer_count < min_customers:
                continue

            ratio = math.inf if baseline <= 0 else customer_count / baseline
            if ratio >= alert_ratio:
                status, severity = "alert", "high"
            elif ratio >= watch_ratio:
                status, severity = "watch", "medium"
            else:
                continue

            end = start + window
            matched = group[
                (group["created_at"] >= start)
                & (group["created_at"] < end)
            ]
            incident_id = f"{group_id}-{start.strftime('%Y%m%dT%H%M%S')}"

            if math.isfinite(ratio):
                baseline_explanation = (
                    f"Observed activity was {ratio:.2f}x the historical "
                    f"baseline of {baseline:.2f} distinct customers per window."
                )
            else:
                baseline_explanation = (
                    "No customers were observed in the historical baseline "
                    "windows; current activity exceeds that zero baseline."
                )

            explanation = (
                f"{customer_count} distinct customers reported issues in "
                f"{group_id} during a {window_minutes}-minute window.",
                baseline_explanation,
                f"{int(current['raw_tickets'])} tickets came from "
                f"{customer_count} distinct customers.",
                "This is a potential incident inferred from ticket volume, "
                "not a confirmed outage.",
            )
            incidents.append(
                Incident(
                    incident_id=incident_id,
                    group_id=str(group_id),
                    window_start=start,
                    window_end=end,
                    unique_customers=customer_count,
                    raw_tickets=int(current["raw_tickets"]),
                    baseline_customers=baseline,
                    severity=severity,
                    status=status,
                    ticket_ids=tuple(matched["ticket_id"].astype(str)),
                    explanation=explanation,
                )
            )

    return sorted(
        incidents,
        key=lambda incident: (
            incident.window_end,
            incident.unique_customers,
            incident.group_id,
        ),
        reverse=True,
    )

DEFAULT_SETTINGS_PATH = CONFIG_DIR / "incidents.yaml"

SETTING_NAMES = (
    "window_minutes",
    "min_customers",
    "watch_ratio",
    "alert_ratio",
    "min_baseline_windows",
)


def load_incident_params(
    path=DEFAULT_SETTINGS_PATH,
) -> dict[str, int | float]:
    """Load and validate incident-detector settings from YAML."""
    data = read_yaml(path)

    missing = [name for name in SETTING_NAMES if name not in data]
    unknown = sorted(set(data) - set(SETTING_NAMES))

    if missing or unknown:
        raise ConfigError(
            f"{path}: expected exactly the settings {list(SETTING_NAMES)}"
            f" (missing {missing}, unexpected {unknown})."
        )

    params = {name: data[name] for name in SETTING_NAMES}

    empty = pd.DataFrame(
        {
            column: []
            for column in (
                "ticket_id",
                "customer_id",
                "created_at",
                "group_id",
            )
        }
    )

    try:
        detect_incidents(empty, **params)
    except (TypeError, ValueError) as exc:
        raise ConfigError(
            f"{path}: invalid incident setting. {exc}"
        ) from exc

    params["watch_ratio"] = float(params["watch_ratio"])
    params["alert_ratio"] = float(params["alert_ratio"])

    return params



def explain_incident(incident: Incident) -> list[str]:
    """Return evidence-based explanation lines for a potential incident.

    The explanation reports observed ticket-volume evidence only; it must not
    imply that an outage or root cause has been confirmed.
    """
    if not isinstance(incident, Incident):
        raise TypeError("incident must be an Incident instance")
    lines = [str(line).strip() for line in incident.explanation if str(line).strip()]
    disclaimer = "This is a potential incident inferred from ticket volume, not a confirmed outage."
    if not any("not a confirmed outage" in line.lower() for line in lines):
        lines.append(disclaimer)
    return lines
