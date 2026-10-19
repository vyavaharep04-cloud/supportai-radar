"""F. Evaluation."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import backend, charts, components as ui, data_access
from dashboard.compat import stretch
from dashboard.labels import issue_labels
from supportai.config import ConfigError

INTERFACE = (
    "# dashboard.backend.load_evaluation() -> dashboard.contracts.EvaluationReport\n"
    "#   accuracy, macro_f1, critical_recall, critical_precision : float\n"
    "#   per_class  : DataFrame[issue_id, precision, recall, f1, support]\n"
    "#   confusion  : DataFrame, rows = true issue_id, columns = predicted issue_id\n"
    "#   thresholds : DataFrame[threshold, review_rate, auto_accuracy]\n"
    "#   incident_scenarios : DataFrame[scenario, kind, detected, delay_minutes, false_alerts]\n"
    "#   dataset_note : str  (say what data the numbers were measured on)"
)
CAVEAT = ("Evaluation on synthetic data does not demonstrate real-world accuracy. "
          "Gold-set examples should be written independently of the generator.")


def render() -> None:
    try:
        tx = backend.get_taxonomy()
    except ConfigError as exc:
        ui.callout(f"Could not load config/taxonomy.yaml: {exc}", "danger")
        return
    src = data_access.evaluation()
    ui.page_header("Evaluation", "Classification quality, critical-ticket safety, human-review trade-offs and incident-detection behaviour.",
                   badges=[ui.provenance_badge("off" if src is None else "demo" if src.is_demo else "live")])
    ui.callout(CAVEAT, "warning")
    if src is None:
        ui.empty_state("No evaluation results yet",
                       "Metrics appear here after the classifier and incident detector have been evaluated. "
                       "No numbers are shown until they have actually been measured.", interface=INTERFACE, icon="∑")
        return
    if src.is_demo:
        # This is a demo result bundle, so expose its provenance before any
        # metrics; tests and users can reliably distinguish it from live data.
        ui.demo_banner("Every number below is computed from synthetic labels with injected noise. It is not a measurement of any model.")
        st.markdown('<div class="sr-demo-banner"><b>DEMO DATA</b> — synthetic evaluation preview; these metrics are not model predictions.</div>', unsafe_allow_html=True)
    ev = src.data
    labels = issue_labels(tx)
    if ev.dataset_note:
        ui.mini("Data: " + ev.dataset_note)
        kpis = [
        ("Accuracy", f"{ev.accuracy:.1%}", "all issue types"),
        ("Macro F1", f"{ev.macro_f1:.3f}", "mean over issue types"),
    ]

    if ev.critical_recall is not None:
        kpis.append(
            (
                "Critical recall (P1)",
                f"{ev.critical_recall:.1%}",
                "share of true P1 tickets caught",
                "success" if ev.critical_recall >= 0.9 else "warning",
            )
        )

    if ev.critical_precision is not None:
        kpis.append(
            (
                "Critical precision (P1)",
                f"{ev.critical_precision:.1%}",
                "share of flagged P1 that are P1",
            )
        )

    ui.kpi_row(kpis)

    ui.section("Classification", "Where the classifier confuses issue types, and per-type scores.")
    a, b = st.columns([1.15, 1], gap="large")
    with a:
        charts.show(charts.confusion_chart(ev.confusion, labels), demo=src.is_demo)
    with b:
        pc = ev.per_class.copy()
        pc["Issue"] = pc["issue_id"].map(lambda i: labels.get(i, i))
        st.dataframe(pc[["Issue", "precision", "recall", "f1", "support"]].rename(columns=str.title), hide_index=True, height=380,
                     column_config={c: st.column_config.NumberColumn(c, format="%.2f") for c in ("Precision", "Recall", "F1")}, **stretch())

    ui.section("Human-review threshold", "Higher thresholds send more tickets to people and make the auto-routed set more accurate.")
    t1, t2 = st.columns([1.3, 1], gap="large")
    with t1:
        charts.show(charts.threshold_chart(ev.thresholds), demo=src.is_demo)
    with t2:
        th = ev.thresholds.copy()
        th = pd.DataFrame({"Threshold": th["threshold"].map("{:.0%}".format), "Sent to review": th["review_rate"].map("{:.1%}".format),
                           "Auto-routed accuracy": th["auto_accuracy"].map("{:.1%}".format)})
        st.dataframe(th, hide_index=True, **stretch())

        ui.section(
        "Incident detection",
        "Detection delay for planted incidents and false alerts on decoys.",
    )

    sc = ev.incident_scenarios.copy()

    if sc.empty:
        ui.empty_state(
            "Incident evaluation not available yet",
            "Incident replay scenarios have not been evaluated. "
            "No detection or false-alert metrics are reported here yet.",
            icon="🔬",
        )
    else:
        sc["Detected"] = sc["detected"].map(
            {True: "Yes", False: "No alert"}
        )
        sc["Delay"] = sc["delay_minutes"].map(
            lambda m: "—" if pd.isna(m) else f"{m:.0f} min"
        )

        out = sc.rename(
            columns={
                "scenario": "Scenario",
                "kind": "Kind",
                "false_alerts": "False alerts",
            }
        )[
            ["Scenario", "Kind", "Detected", "Delay", "False alerts"]
        ]

        st.dataframe(out, hide_index=True, **stretch())