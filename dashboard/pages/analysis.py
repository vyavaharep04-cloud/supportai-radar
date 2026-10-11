"""C. Ticket Analysis: live message analysis with the real decision trace."""

from __future__ import annotations

import dataclasses

import streamlit as st

from dashboard import backend, components as ui, data_access
from dashboard.contracts import CAPABILITY_SPECS, TicketAnalysis
from dashboard.labels import department_labels, issue_labels
from supportai.config import ConfigError

# Example INPUT messages (inputs only: nothing here is an output of any model).
EXAMPLES: tuple[str, ...] = (
    "Paisa kat gaya but order nahi aaya!!!",
    "Mere paise cut ho gaye, refund kab milega?",
    "Payment failed but ₹1,299.50 debited. Pls help",
    "Login nahi ho raha, password bhi nahi chal raha",
    "I didn’t get my refund!!! It’s been 10 days",
    "Wrong item aaya, toota hua hai, replacement chahiye",
)
KEY_TEXT, KEY_RUN = "analysis_text", "analysis_submitted"


def _spec(key: str):
    return next(s for s in CAPABILITY_SPECS if s.key == key)


def _off(key: str, what: str) -> str:
    spec = _spec(key)
    return (
        f'<div class="note">Not connected yet. {ui.esc(what)} Expected: '
        f'<code>{ui.esc(spec.interface)}</code> in <code>{ui.esc(spec.module)}</code>.</div>'
    )


def _stage_normalize(a: TicketAnalysis) -> str:
    n = a.normalization
    info = backend.normalizer_info()
    body = ui.text_block("Original message", a.raw_text)
    body += '<div style="height:.6rem"></div>' + ui.text_block("Normalized text", n.normalized if n else "", mono=True)
    if n and n.applied_rules:
        body += '<div class="sr-label" style="margin-top:.8rem">Lexicon rules applied</div>' + "".join(
            f'<div class="sr-rule"><code>{ui.esc(r.matched)}</code><span class="arrow">→</span><code>{ui.esc(r.replacement)}</code>'
            f'<span class="rid">{ui.esc(r.rule_id)}</span></div>' for r in n.applied_rules
        )
    else:
        body += '<div class="note" style="margin-top:.6rem">No lexicon rules changed this message.</div>'
    body += (f'<div class="note" style="margin-top:.6rem">Lexicon v{ui.esc(info["version"])} · fingerprint '
             f'<code>{ui.esc(str(info["fingerprint"])[:12])}</code>. Entities and urgency read the original text, not this view.</div>')
    return ui.stage_card(1, "Understand · normalize", "live", body)


def _stage_entities(a: TicketAnalysis) -> str:
    if a.entities is None:
        return ui.stage_card(2, "Understand · extract entities", "off", _off("entities", "Amounts, order IDs and similar details will appear here."))
    body = ui.chips((e.kind, e.value) for e in a.entities) if a.entities else '<div class="note">No entities found in the original text.</div>'
    return ui.stage_card(2, "Understand · extract entities", "live", body)


def _stage_classify(a: TicketAnalysis, labels) -> str:
    c = a.classification
    if c is None:
        return ui.stage_card(3, "Decide · classify issue", "off", _off("classifier", "The predicted issue type, confidence and review decision will appear here."))
    body = ui.kv_list([("Predicted issue", labels.get(c.issue_id, c.issue_id))])
    body += ui.confidence_bar(c.confidence) + '<div style="margin-top:.5rem">' + ui.review_badge(c.needs_human_review) + "</div>"
    if c.review_reason:
        body += f'<div class="note" style="margin-top:.4rem">{ui.esc(c.review_reason)}</div>'
    if c.runner_up:
        body += '<div class="sr-label" style="margin-top:.7rem">Runner-up</div>' + ui.chips((labels.get(i, i), f"{p:.0%}") for i, p in c.runner_up)
    return ui.stage_card(3, "Decide · classify issue", "live", body)


def _stage_route(a: TicketAnalysis, depts) -> str:
    if a.classification is None or a.department_id is None:
        return ui.stage_card(4, "Decide · route to department", "off", _off("classifier", "The department (or Human Review) is decided together with the classification."))
    return ui.stage_card(4, "Decide · route to department", "live", ui.kv_list([("Department", depts.get(a.department_id, a.department_id))]))


def _stage_urgency(a: TicketAnalysis) -> str:
    u = a.urgency
    if u is None:
        return ui.stage_card(5, "Decide · urgency and safety rules", "off", _off("urgency", "Priority, scoring factors and hard priority floors will appear here."))
    body = ui.kv_list([("Priority", ui.priority_badge(u.priority)), ("Score", f"{u.score:.1f}")], raw_values=True)
    if u.factors:
        body += '<div class="sr-label" style="margin-top:.7rem">Urgency factors</div>' + ui.kv_list([(f.name, f"{f.points:+.1f} · {f.detail}") for f in u.factors])
    if u.floors_applied:
        body += '<div class="sr-label" style="margin-top:.7rem">Safety rules applied</div>' + "".join(ui.badge(f, "warning") + " " for f in u.floors_applied)
    return ui.stage_card(5, "Decide · urgency and safety rules", "live", body)


def _stage_assist(a: TicketAnalysis) -> str:
    p = a.playbook
    if p is None:
        return ui.stage_card(6, "Assist · reply and checklist", "off", _off("playbooks", "A suggested reply and an action checklist will appear here."))
    body = ui.text_block("Suggested reply", p.reply)
    if p.checklist:
        body += '<div class="sr-label" style="margin-top:.8rem">Action checklist</div>' + "".join(f'<div class="sr-rule">☐ {ui.esc(s)}</div>' for s in p.checklist)
    return ui.stage_card(6, "Assist · reply and checklist", "live", body)


def _summary(a: TicketAnalysis, labels, depts) -> str:
    dash = '<span style="color:#8795A8">— not connected</span>'
    c, u = a.classification, a.urgency
    rows = [
        ("Issue", ui.esc(labels.get(c.issue_id, c.issue_id)) if c else dash),
        ("Department", ui.esc(depts.get(a.department_id, a.department_id)) if a.department_id else dash),
        ("Priority", ui.priority_badge(u.priority) if u else dash),
        ("Human review", ui.review_badge(c.needs_human_review) if c else dash),
    ]
    return ui.card("Decision summary", ui.kv_list(rows, raw_values=True))


DEMO_TRACE_MESSAGE = "Paisa kat gaya but order nahi aaya!!! ₹1,299.50 deducted, order ORD482913"


def _demo_trace_tab() -> None:
    ui.demo_banner("Stages marked DEMO DATA below are fixed, hand-written examples, not computed from any input. "
                   "Stages marked LIVE are real backend output for the message shown.")
    st.markdown(f"**Message:** “{DEMO_TRACE_MESSAGE}”")
    real = backend.analyze_ticket(DEMO_TRACE_MESSAGE)
    cols = st.columns(2, gap="large")
    with cols[0]:
        ui.render(_stage_entities(real))  # LIVE when the extractor is connected
        ui.render(ui.stage_card(3, "Decide · classify issue", "demo",
                  ui.kv_list([("Predicted issue", "Money Deducted, No Order")]) + ui.confidence_bar(0.91, 0.55)
                  + '<div style="margin-top:.5rem">' + ui.review_badge(False) + "</div>"))
        ui.render(ui.stage_card(4, "Decide · route to department", "demo", ui.kv_list([("Department", "Payments & Billing")])))
    with cols[1]:
        ui.render(ui.stage_card(5, "Decide · urgency and safety rules", "demo",
                  ui.kv_list([("Priority", ui.priority_badge("P2")), ("Score", "7.5")], raw_values=True)
                  + '<div class="sr-label" style="margin-top:.7rem">Urgency factors</div>'
                  + ui.kv_list([("Money at risk", "+3.0 · amount over threshold"), ("Repeated wording", "+1.5 · “!!!”"), ("Issue base", "+3.0 · payment issue")])))
        ui.render(ui.stage_card(6, "Assist · reply and checklist", "demo",
                  ui.text_block("Suggested reply", "Sorry about the trouble. We can see the debit and are checking it against your order now.")
                  + '<div class="sr-label" style="margin-top:.8rem">Action checklist</div>'
                  + "".join(f'<div class="sr-rule">☐ {ui.esc(s)}</div>' for s in ("Verify transaction ID", "Check order creation log", "Start refund if no order exists"))))


def render() -> None:
    try:
        tx = backend.get_taxonomy()
    except ConfigError as exc:
        ui.callout(f"Could not load config/taxonomy.yaml: {exc}", "danger")
        return
    labels, depts = issue_labels(tx), department_labels(tx)
    caps = backend.capabilities()
    live_n = sum(c.connected for c in caps if c.key in {"normalization", "entities", "classifier", "urgency", "playbooks"})
    ui.page_header(
        "Ticket Analysis",
        "Paste a customer message (English or Hinglish) and see each processing stage and its decision trace.",
        badges=[ui.badge(f"{live_n} of 5 analysis stages connected", "brand" if live_n else "muted")],
    )

    st.session_state.setdefault(KEY_TEXT, "")
    st.selectbox("Load an example message", ("",) + EXAMPLES, key="analysis_example", format_func=lambda s: s or "Choose an example…",
                 on_change=lambda: st.session_state.update({KEY_TEXT: st.session_state["analysis_example"]}) if st.session_state["analysis_example"] else None)
    st.text_area("Customer message", key=KEY_TEXT, height=110, placeholder="e.g. Paisa kat gaya but order nahi aaya")
    if st.button("Analyze message", type="primary"):
        st.session_state[KEY_RUN] = st.session_state[KEY_TEXT]

    tabs = st.tabs(["Live analysis", "Design preview of a full trace"] if data_access.preview_enabled() else ["Live analysis"])
    with tabs[0]:
        submitted = st.session_state.get(KEY_RUN, "")
        if not submitted.strip():
            ui.empty_state("Nothing analysed yet", "Enter a message above and press “Analyze message”. Normalization runs live with the real Hinglish normalizer.", icon="✎")
        else:
            analysis = backend.analyze_ticket(submitted)
            main, side = st.columns([2.1, 1], gap="large")
            with main:
                for stage in (_stage_normalize(analysis), _stage_entities(analysis), _stage_classify(analysis, labels),
                              _stage_route(analysis, depts), _stage_urgency(analysis), _stage_assist(analysis)):
                    ui.render(stage)
            with side:
                ui.render(_summary(analysis, labels, depts))
                st.write("")
                ui.render(ui.card("Why some stages are empty", '<div class="sr-mini">Only stages whose backend module exists run here. '
                          "Nothing is guessed or simulated for the rest.</div>"))
            with st.expander("Raw decision trace (JSON)"):
                st.json(dataclasses.asdict(analysis))
    if len(tabs) > 1:
        with tabs[1]:
            _demo_trace_tab()
