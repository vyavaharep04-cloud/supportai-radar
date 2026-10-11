"""DEMO DATA for design previews.  Nothing in this module is real output.

What is synthetic here
----------------------
* Tickets, customers, timestamps and "predicted" labels are produced by a seeded
  generator.  The labels are the generator's own ground truth with injected
  noise; they are NOT predictions from any model.
* The replay detector state uses scripted illustrative thresholds.  It is NOT
  the project's incident detector.
* Metrics are computed from the synthetic labels above and are NOT evaluation
  results.

What is real here
-----------------
* Issue ids / departments / incident groups come from the real taxonomy (the
  generator refuses to run if its templates drift from ``config/taxonomy.yaml``).
* ``normalized_text`` is produced by the real Hinglish normalizer, and ``entities``
  by the real entity extractor (when connected).

This module is only reached when the user switches on "Design preview" in the
sidebar, and only through ``dashboard.data_access``.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from dashboard.backend import extract_entity_views
from dashboard.contracts import EvaluationReport, IncidentView, PRIORITIES, ReplayData
from supportai.normalizer import get_default_normalizer
from supportai.taxonomy import Taxonomy

DEMO_START = pd.Timestamp("2026-03-14 00:00")  # fixed, obviously synthetic
DEMO_HOURS = 48
WINDOW_MINUTES = 60
REVIEW_THRESHOLD = 0.55  # demo-only; the real threshold belongs to the classifier config
TICK_MINUTES = 10
SEED = 11


class PreviewDataError(RuntimeError):
    """The demo templates no longer match the real taxonomy."""


# Example messages per issue id (English + Hinglish).  {amt} / {oid} are filled in.
TEMPLATES: dict[str, list[str]] = {
    "payment_failed": [
        "Payment failed again for order {oid}, please help",
        "Payment nahi ho raha, card se try kiya {amt} ka",
        "UPI payment failed but I want to place the order",
    ],
    "money_deducted_no_order": [
        "Paisa kat gaya but order nahi aaya!!! {amt} deducted, no order {oid}",
        "Money debited from my account, order not confirmed. Txn for {amt}",
        "Mere paise cut ho gaye lekin order create nahi hua",
    ],
    "double_charge": [
        "I was charged twice for order {oid}, {amt} deducted two times",
        "Double charge ho gaya, {amt} do baar kata",
        "Same payment debited 2 times, please reverse",
    ],
    "checkout_error": [
        "Checkout me error aa raha hai, order place nahi ho raha",
        "App crashes on the payment page, cannot complete checkout",
        "Coupon apply karte hi checkout error show hota hai",
    ],
    "refund_pending": [
        "Refund kab milega? Order {oid} cancel kiye 7 din ho gaye",
        "Still waiting for my refund of {amt}, it's been 10 days",
        "Refund abhi tak nahi aaya, {amt} stuck hai",
    ],
    "unauthorized_activity": [
        "Kisi aur ne mere account se order kiya, unauthorized transaction {amt}",
        "Someone logged in without my permission and placed order {oid}",
        "My account was hacked, I did not make this purchase",
    ],
    "login_problem": [
        "Login nahi ho raha, OTP baar baar aa raha hai",
        "Cannot log in, password reset link is not working",
        "Account block ho gaya, please help me login",
    ],
    "order_delay": [
        "Order {oid} abhi tak nahi aaya, 5 din ho gaye",
        "My order is delayed, was supposed to arrive yesterday",
        "Delivery kab hogi? Tracking me kuch update nahi",
    ],
    "wrong_damaged_item": [
        "Wrong item aaya, toota hua hai, replacement chahiye",
        "Received a damaged product in order {oid}",
        "Galat saman bheja hai, order {oid} ka",
    ],
    "cancel_modify_order": [
        "Please cancel order {oid}, galat address dala",
        "Cancel kardo plz, I ordered by mistake",
        "Want to change delivery address for order {oid}",
    ],
    "other": [
        "Hello, do you have an offline store in Pune?",
        "Need a GST invoice for last month",
        "Thank you for the quick help yesterday",
    ],
}

# background mix of issue types (sums to 1); demo-only weights
_MIX: dict[str, float] = {
    "payment_failed": 0.13, "money_deducted_no_order": 0.05, "double_charge": 0.04, "checkout_error": 0.09,
    "refund_pending": 0.16, "unauthorized_activity": 0.03, "login_problem": 0.12, "order_delay": 0.17,
    "wrong_damaged_item": 0.08, "cancel_modify_order": 0.08, "other": 0.05,
}
_PRIORITY_BIAS: dict[str, list[float]] = {  # P1..P4 weights; demo-only
    "unauthorized_activity": [0.7, 0.25, 0.05, 0.0], "double_charge": [0.2, 0.5, 0.25, 0.05],
    "money_deducted_no_order": [0.15, 0.5, 0.3, 0.05], "payment_failed": [0.05, 0.3, 0.5, 0.15],
    "checkout_error": [0.05, 0.3, 0.5, 0.15], "login_problem": [0.04, 0.26, 0.5, 0.2],
    "refund_pending": [0.02, 0.2, 0.5, 0.28], "order_delay": [0.03, 0.2, 0.52, 0.25],
    "wrong_damaged_item": [0.02, 0.2, 0.5, 0.28], "cancel_modify_order": [0.01, 0.1, 0.4, 0.49],
    "other": [0.0, 0.05, 0.35, 0.6],
}


def _check_templates(taxonomy: Taxonomy) -> None:
    ids = set(taxonomy.issue_ids())
    for name, table in (("TEMPLATES", TEMPLATES), ("_MIX", _MIX), ("_PRIORITY_BIAS", _PRIORITY_BIAS)):
        if set(table) != ids:
            missing = sorted(ids - set(table))
            extra = sorted(set(table) - ids)
            raise PreviewDataError(
                f"preview_data.{name} is out of sync with config/taxonomy.yaml (missing {missing}, unexpected {extra}). "
                "Update dashboard/preview_data.py."
            )


def _diurnal(hour_of_day: float) -> float:
    return 0.55 + 0.75 * np.exp(-((hour_of_day - 13.0) ** 2) / 30.0) + 0.35 * np.exp(-((hour_of_day - 20.0) ** 2) / 12.0)


def _make_ticket(rng, taxonomy, issue_id, ts, customer, seq, segment):
    template = TEMPLATES[issue_id][rng.integers(len(TEMPLATES[issue_id]))]
    amount = f"₹{int(rng.choice([199, 349, 499, 799, 1299, 2499])):,}"
    order = f"ORD{int(rng.integers(100000, 999999))}"  # format the real extractor recognises
    raw = template.format(amt=amount, oid=order)
    group = next(i.incident_group for i in taxonomy.issue_types if i.id == issue_id)
    return {
        "ticket_id": f"T-{seq:05d}", "customer_id": customer, "created_at": ts, "raw_text": raw,
        "true_issue_id": issue_id, "group_id": group, "segment": segment,
    }


def _generate_tickets(taxonomy: Taxonomy) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    ids = list(_MIX)
    probs = np.array([_MIX[i] for i in ids])
    probs = probs / probs.sum()
    rows: list[dict] = []
    seq = 1
    customers = [f"C-{n}" for n in range(1000, 1900)]

    # 1. normal background: ~14 tickets/hour, shaped by time of day
    for h in range(DEMO_HOURS):
        n = rng.poisson(14 * _diurnal(h % 24))
        for _ in range(n):
            ts = DEMO_START + pd.Timedelta(minutes=float(h * 60 + rng.uniform(0, 60)))
            issue = ids[rng.choice(len(ids), p=probs)]
            rows.append(_make_ticket(rng, taxonomy, issue, ts, customers[rng.integers(len(customers))], seq, "normal"))
            seq += 1

    # 2. DECOY A: one customer files 26 refund tickets in 25 minutes (high volume, ONE customer)
    spam_start = DEMO_START + pd.Timedelta(hours=12, minutes=5)
    for k in range(26):
        ts = spam_start + pd.Timedelta(minutes=k * 25 / 26)
        rows.append(_make_ticket(rng, taxonomy, "refund_pending", ts, "C-9001", seq, "decoy"))
        seq += 1

    # 3. DECOY B: benign evening bump in delivery tickets from many customers (reaches 'watch', never 'alert')
    for h in (20, 21, 22):
        for _ in range(rng.poisson(7)):
            ts = DEMO_START + pd.Timedelta(minutes=float(h * 60 + rng.uniform(0, 60)))
            issue = ["order_delay", "order_delay", "wrong_damaged_item", "cancel_modify_order"][rng.integers(4)]
            rows.append(_make_ticket(rng, taxonomy, issue, ts, customers[rng.integers(len(customers))], seq, "decoy"))
            seq += 1

    # 4. PLANTED incident: payment/checkout surge from many distinct customers (hour 30 -> 31.5)
    plant_customers = [f"C-{n}" for n in range(2000, 2060)]
    for k in range(64):
        ts = DEMO_START + pd.Timedelta(hours=30, minutes=float(rng.uniform(0, 90)))
        issue = ["checkout_error", "payment_failed", "money_deducted_no_order", "payment_failed"][rng.integers(4)]
        rows.append(_make_ticket(rng, taxonomy, issue, ts, plant_customers[rng.integers(len(plant_customers))], seq, "planted"))
        seq += 1

    df = pd.DataFrame(rows).sort_values("created_at", kind="stable").reset_index(drop=True)
    df["ticket_id"] = [f"T-{i + 1:05d}" for i in range(len(df))]
    return df


def _add_demo_predictions(df: pd.DataFrame, taxonomy: Taxonomy) -> pd.DataFrame:
    """Fabricated 'predictions': ground truth with noise.  Labelled DEMO everywhere it is shown."""
    rng = np.random.default_rng(SEED + 1)
    ids = taxonomy.issue_ids()
    normalizer = get_default_normalizer()  # REAL normalizer
    out = df.copy()
    correct = rng.random(len(out)) > 0.13
    pred = [t if ok else rng.choice([i for i in ids if i != t]) for t, ok in zip(out["true_issue_id"], correct)]
    conf = np.where(correct, np.clip(rng.normal(0.84, 0.09, len(out)), 0.45, 0.99), np.clip(rng.normal(0.52, 0.12, len(out)), 0.2, 0.85))
    out["issue_id"] = pred
    out["confidence"] = conf.round(3)
    out["needs_review"] = (out["confidence"] < REVIEW_THRESHOLD) | (out["issue_id"] == "other")
    dept = {i.id: i.department for i in taxonomy.issue_types}
    out["department_id"] = [("human_review" if nr else dept[i]) for i, nr in zip(out["issue_id"], out["needs_review"])]
    out["true_priority"] = [PRIORITIES[rng.choice(4, p=_PRIORITY_BIAS[t])] for t in out["true_issue_id"]]
    flip = rng.random(len(out)) < 0.12
    out["priority"] = [
        PRIORITIES[min(3, max(0, PRIORITIES.index(p) + int(rng.choice([-1, 1]))))] if f else p
        for p, f in zip(out["true_priority"], flip)
    ]
    out["normalized_text"] = [normalizer.normalize(t) for t in out["raw_text"]]
    # entities: REAL extractor output on the raw text (None-safe if the module is absent)
    out["entities"] = [
        ", ".join(f"{e.kind} {e.value}" for e in (extract_entity_views(t) or ())) for t in out["raw_text"]
    ]
    return out


def tickets(taxonomy: Taxonomy) -> pd.DataFrame:
    _check_templates(taxonomy)
    return _add_demo_predictions(_generate_tickets(taxonomy), taxonomy)


# --------------------------------------------------------------------------
# Replay / incidents (scripted illustrative detector)
# --------------------------------------------------------------------------

ALERT_MIN_CUSTOMERS, ALERT_FACTOR = 18, 4.0
WATCH_MIN_CUSTOMERS, WATCH_FACTOR = 13, 2.5


def _timeline(df: pd.DataFrame) -> pd.DataFrame:
    ticks = pd.date_range(DEMO_START + pd.Timedelta(minutes=TICK_MINUTES), DEMO_START + pd.Timedelta(hours=DEMO_HOURS), freq=f"{TICK_MINUTES}min")
    rows = []
    window = pd.Timedelta(minutes=WINDOW_MINUTES)
    for group, sub in df.groupby("group_id"):
        times = sub["created_at"].to_numpy()
        customers = sub["customer_id"].to_numpy()
        series = []
        for ts in ticks:
            mask = (times > (ts - window).to_datetime64()) & (times <= ts.to_datetime64())
            series.append((ts, len(set(customers[mask])), int(mask.sum())))
        baseline_pool = [d for t, d, _ in series if t <= DEMO_START + pd.Timedelta(hours=24)]
        baseline = float(np.median(baseline_pool)) if baseline_pool else 1.0
        baseline = max(baseline, 2.0)
        threshold = max(ALERT_MIN_CUSTOMERS, ALERT_FACTOR * baseline)
        watch = max(WATCH_MIN_CUSTOMERS, WATCH_FACTOR * baseline)
        for ts, distinct, raw in series:
            state = "alert" if distinct >= threshold else "watch" if distinct >= watch else "normal"
            rows.append({"ts": ts, "group_id": group, "distinct_customers": distinct, "raw_tickets": raw,
                         "baseline": baseline, "threshold": threshold, "state": state})
    return pd.DataFrame(rows).sort_values(["group_id", "ts"]).reset_index(drop=True)


def replay(taxonomy: Taxonomy, tickets_df: pd.DataFrame) -> ReplayData:
    group_tickets = tickets_df[tickets_df["group_id"].notna()].copy()
    return ReplayData(
        tickets=group_tickets,
        timeline=_timeline(group_tickets),
        window_minutes=WINDOW_MINUTES,
        description=(
            "Synthetic 48-hour scenario: normal background traffic, two decoys "
            "(a single customer repeating tickets; a benign evening bump) and one planted "
            "payment/checkout surge. Detector thresholds here are scripted for illustration."
        ),
    )


def _severity(ratio: float) -> str:
    return "critical" if ratio >= 8 else "high" if ratio >= 5 else "medium" if ratio >= 3 else "low"


def incidents(taxonomy: Taxonomy, tickets_df: pd.DataFrame, replay_data: ReplayData) -> list[IncidentView]:
    out: list[IncidentView] = []
    tl = replay_data.timeline
    for group, sub in tl.groupby("group_id"):
        sub = sub.reset_index(drop=True)
        flagged = sub["state"].isin(["alert", "watch"])
        if not flagged.any():
            continue
        # contiguous episodes of non-normal state
        episode = (flagged != flagged.shift()).cumsum()
        for _, ep in sub[flagged].groupby(episode[flagged]):
            peak = ep.loc[ep["distinct_customers"].idxmax()]
            status = "alert" if (ep["state"] == "alert").any() else "watch"
            start = ep["ts"].min() - pd.Timedelta(minutes=WINDOW_MINUTES)
            end = ep["ts"].max()
            linked = replay_data.tickets[
                (replay_data.tickets["group_id"] == group)
                & (replay_data.tickets["created_at"] > start)
                & (replay_data.tickets["created_at"] <= end)
            ]
            ratio = peak["distinct_customers"] / peak["baseline"]
            top = Counter(linked["issue_id"]).most_common(3)
            repeats = int(len(linked) - linked["customer_id"].nunique())
            lines = (
                f"{int(peak['distinct_customers'])} distinct customers reported {group} issues within {WINDOW_MINUTES} minutes "
                f"(historical baseline {peak['baseline']:.1f}; {ratio:.1f}x).",
                f"{len(linked)} tickets in total; {repeats} are repeat submissions by customers already counted.",
                "Most common issue types: " + ", ".join(f"{i} ({n})" for i, n in top) + ".",
                "This is a potential incident inferred from ticket volume. It does not confirm an outage or its cause.",
            )
            out.append(IncidentView(
                incident_id=f"INC-{len(out) + 1:03d}", group_id=group, window_start=start, window_end=end,
                unique_customers=int(peak["distinct_customers"]), raw_tickets=len(linked), baseline_customers=float(peak["baseline"]),
                severity=_severity(ratio), status=status, ticket_ids=tuple(linked["ticket_id"]), explanation=lines,
            ))
    return sorted(out, key=lambda i: i.window_start, reverse=True)


# --------------------------------------------------------------------------
# Evaluation (computed from the synthetic labels above; NOT a model result)
# --------------------------------------------------------------------------


def evaluation(taxonomy: Taxonomy, tickets_df: pd.DataFrame, replay_data: ReplayData) -> EvaluationReport:
    ids = taxonomy.issue_ids()
    true, pred = tickets_df["true_issue_id"], tickets_df["issue_id"]
    confusion = pd.crosstab(true, pred).reindex(index=ids, columns=ids, fill_value=0)
    confusion.index.name = "true"
    rows = []
    for i in ids:
        tp = int(confusion.loc[i, i])
        fp = int(confusion[i].sum() - tp)
        fn = int(confusion.loc[i].sum() - tp)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        rows.append({"issue_id": i, "precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0, "support": int(confusion.loc[i].sum())})
    per_class = pd.DataFrame(rows)

    crit_true = tickets_df["true_priority"] == "P1"
    crit_pred = tickets_df["priority"] == "P1"
    tp = int((crit_true & crit_pred).sum())
    thresholds = []
    correct = tickets_df["true_issue_id"] == tickets_df["issue_id"]
    for t in (0.40, 0.50, 0.55, 0.60, 0.70, 0.80):
        auto = tickets_df["confidence"] >= t
        thresholds.append({"threshold": t, "review_rate": float(1 - auto.mean()), "auto_accuracy": float(correct[auto].mean()) if auto.any() else 0.0})

    tl = replay_data.timeline
    scenarios = []
    planted_ts = replay_data.tickets.loc[replay_data.tickets["segment"] == "planted", "created_at"]
    first_alert = tl.loc[(tl["state"] == "alert") & (tl["group_id"] == "payment_checkout"), "ts"].min()
    scenarios.append({"scenario": "Payment/checkout surge", "kind": "planted", "detected": bool(pd.notna(first_alert)),
                      "delay_minutes": float((first_alert - planted_ts.min()).total_seconds() / 60) if pd.notna(first_alert) else None,
                      "false_alerts": 0})
    for name, group, lo, hi in (("Repeat submitter (1 customer)", "refunds", 12, 13), ("Evening delivery bump", "orders_delivery", 20, 23)):
        window = tl[(tl["group_id"] == group) & (tl["ts"] >= DEMO_START + pd.Timedelta(hours=lo)) & (tl["ts"] <= DEMO_START + pd.Timedelta(hours=hi + 1))]
        scenarios.append({"scenario": name, "kind": "decoy", "detected": False, "delay_minutes": None,
                          "false_alerts": int((window["state"] == "alert").sum())})
    return EvaluationReport(
        accuracy=float(correct.mean()), macro_f1=float(per_class["f1"].mean()), per_class=per_class, confusion=confusion,
        critical_recall=tp / int(crit_true.sum()) if crit_true.sum() else 0.0,
        critical_precision=tp / int(crit_pred.sum()) if crit_pred.sum() else 0.0,
        thresholds=pd.DataFrame(thresholds), incident_scenarios=pd.DataFrame(scenarios),
        dataset_note="Computed from synthetic demo labels with injected noise. Not a measurement of any model.",
    )
