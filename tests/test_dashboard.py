"""Tests for the Streamlit dashboard (dashboard/ and app.py).

Focus: the dashboard must be HONEST.
* live mode never shows demo data or fabricated results;
* demo data is always labelled and can never override live backend data;
* the UI reads the real taxonomy/normalizer instead of duplicating them;
* ticket text is never rendered as HTML.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from dashboard import backend, components, contracts, data_access, preview_data  # noqa: E402
from supportai.normalizer import get_default_normalizer  # noqa: E402
from supportai.taxonomy import load_taxonomy  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "dashboard"
PAGES = ["overview", "explorer", "analysis", "incidents", "replay", "evaluation"]
SCRIPT = "from dashboard.shell import bootstrap\nfrom dashboard.pages import {page}\nbootstrap(preview={preview})\n{page}.render()\n"


def run_page(page: str, preview: bool) -> AppTest:
    return AppTest.from_string(SCRIPT.format(page=page, preview=preview), default_timeout=60).run()


def markdown_of(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


def has_demo_banner(at: AppTest) -> bool:
    """True only if the banner element is rendered (the stylesheet also contains the class name)."""
    return 'class="sr-demo-banner"' in markdown_of(at)


# --------------------------------------------------------------------------
# Backend adapter honesty
# --------------------------------------------------------------------------


def test_capabilities_report_only_what_really_exists():
    caps = {c.key: c for c in backend.capabilities()}
    assert caps["normalization"].connected and caps["taxonomy"].connected
    assert caps["entities"].connected == (ROOT / "supportai" / "entities.py").exists()
    # Backend modules that do not exist yet must be reported as not connected.
        # Existing modules without adapters must remain disconnected.
    for key in (
        "ingestion",
        "playbooks",
        "incidents",
        "replay",
        "evaluation",
    ):
        spec_module = caps[key].module
        module_file = ROOT / (spec_module.replace(".", "/") + ".py")
        if not module_file.exists():
            assert not caps[key].connected, f"{key} reported as connected but {spec_module} does not exist"


def test_a_module_that_exists_without_an_adapter_is_not_reported_as_connected(monkeypatch):
    monkeypatch.setattr(backend, "_module_exists", lambda name: True)

    caps = {c.key: c for c in backend.capabilities()}

    # Capabilities with adapters remain connected.
    assert caps["normalization"].connected
    assert caps["taxonomy"].connected
    assert caps["classifier"].connected

    # Implemented adapters are connected; the explainer is a separate unfinished feature.
    assert caps["ingestion"].connected
    assert caps["incidents"].connected
    for key in ("playbooks", "incident_explainer", "replay", "evaluation"):
        assert not caps[key].connected
        assert "adapter is not written" in caps[key].detail


def test_capability_keys_are_unique_and_cover_the_frozen_workflow():
    keys = [s.key for s in contracts.CAPABILITY_SPECS]
    assert len(keys) == len(set(keys))
    stages = {s.stage for s in contracts.CAPABILITY_SPECS}
    assert set(contracts.WORKFLOW_STAGES) <= stages


def test_analyze_ticket_uses_real_backend_stages_and_invents_nothing():
    from supportai.entities import extract_entities

    text = (
        "Paisa kat gaya but order nahi aaya!!! "
        "₹1,299.50 deducted, order ORD482913"
    )

    analysis = backend.analyze_ticket(text)

    assert analysis.raw_text == text
    assert analysis.normalization == get_default_normalizer().explain(text)

    # Entities must come from the real extractor.
    assert analysis.entities == backend.entity_views(
        extract_entities(text)
    )

    # Classification must come from the trained model.
    assert analysis.classification is not None
    assert analysis.classification.issue_id in load_taxonomy().issue_ids()
    assert 0.0 <= analysis.classification.confidence <= 1.0

    # Department must be provided by the backend.
    assert analysis.department_id is not None

    if analysis.classification.needs_human_review:
        assert analysis.department_id == "human_review"

    # These stages have not been integrated yet.
    # Urgency is integrated; playbooks are not yet integrated.
    assert analysis.urgency is not None
    assert analysis.urgency.priority in {"P1", "P2", "P3", "P4"}
    assert analysis.playbook is None


def test_entity_views_translates_every_entity_kind():
    views = backend.entity_views({
        "amounts": [1299.5, 499.0], "order_ids": ["ORD1"], "transaction_ids": ["TXN9"],
        "payment_methods": ["credit_card"], "products": ["mobile_phone"],
    })
    assert [(v.kind, v.value) for v in views] == [
        ("amount", "₹1,299.50"), ("amount", "₹499"), ("order id", "ORD1"), ("transaction id", "TXN9"),
        ("payment method", "credit card"), ("product", "mobile phone"),
    ]
    assert backend.entity_views({}) == ()


def test_entities_are_extracted_from_raw_text_not_normalized_text():
    # the normalizer lowercases and rewrites text; the extractor must still see the original
    views = backend.analyze_ticket("Payment of Rs.499 failed for order ORD123456").entities
    assert ("order id", "ORD123456") in {(v.kind, v.value) for v in views}


def test_entities_stage_is_none_when_the_module_is_not_connected(monkeypatch):
    monkeypatch.setattr(backend, "is_connected", lambda key: False)
    assert backend.analyze_ticket("order ORD123456").entities is None


@pytest.mark.parametrize("loader", ["load_tickets", "load_replay", "load_evaluation"])
def test_unconnected_loaders_return_none(loader):
    assert getattr(backend, loader)() is None


def test_incident_page_explains_missing_live_ticket_data():
    at = run_page("incidents", preview=False)
    assert not at.exception
    md = markdown_of(at)
    assert "No live ticket data connected yet" in md
    assert "SUPPORTAI_TICKETS_CSV" in md


def test_incident_loader_returns_dashboard_view_models(monkeypatch):
    rows = []
    base = pd.Timestamp("2026-10-10T00:00:00Z")
    for window in range(4):
        for customer in range(2):
            rows.append({
                "ticket_id": f"H-{window}-{customer}",
                "customer_id": f"C-{window}-{customer}",
                "created_at": base + pd.Timedelta(minutes=30 * window),
                "group_id": "payment_checkout",
            })
    for customer in range(6):
        rows.append({
            "ticket_id": f"I-{customer}",
            "customer_id": f"INC-{customer}",
            "created_at": base + pd.Timedelta(minutes=120),
            "group_id": "payment_checkout",
        })
    monkeypatch.setattr(backend, "load_tickets", lambda: pd.DataFrame(rows))

    incidents = backend.load_incidents()

    assert incidents is not None
    assert len(incidents) == 1
    assert isinstance(incidents[0], contracts.IncidentView)
    assert incidents[0].group_id == "payment_checkout"
    assert incidents[0].status == "alert"



def test_configured_ticket_loader_returns_none_without_path(monkeypatch):
    monkeypatch.delenv("SUPPORTAI_TICKETS_CSV", raising=False)
    backend._read_inference_csv.cache_clear()
    backend._analyze_input_csv.cache_clear()
    assert backend.load_tickets() is None


def test_configured_ticket_loader_runs_predictions_not_ground_truth(tmp_path, monkeypatch):
    from types import SimpleNamespace

    path = tmp_path / "incoming.csv"
    pd.DataFrame({
        "ticket_id": ["00100", "00101"],
        "customer_id": ["00010", "00011"],
        "timestamp": ["2026-10-12 10:00:00", "2026-10-12 10:05:00"],
        "raw_text": ["UPI payment failed", "My order delivery is late"],
        "issue_type": ["order_delay", "payment_failed"],
        "department": ["wrong source label", "wrong source label"],
    }).to_csv(path, index=False)
    monkeypatch.setenv("SUPPORTAI_TICKETS_CSV", str(path))
    backend._read_inference_csv.cache_clear()
    backend._analyze_input_csv.cache_clear()

    fake_decisions = [
        SimpleNamespace(
            raw_text="UPI payment failed",
            normalization=SimpleNamespace(normalized="upi payment failed"),
            classification=SimpleNamespace(issue_type="payment_failed", confidence=0.9),
            department_id="payments_billing",
            urgency=SimpleNamespace(priority="P3"),
            needs_human_review=False,
            entities={},
        ),
        SimpleNamespace(
            raw_text="My order delivery is late",
            normalization=SimpleNamespace(normalized="my order delivery is late"),
            classification=SimpleNamespace(issue_type="order_delay", confidence=0.8),
            department_id="orders_delivery",
            urgency=SimpleNamespace(priority="P3"),
            needs_human_review=False,
            entities={},
        ),
    ]
    monkeypatch.setattr(backend, "analyze_dataframe", lambda *a, **k: fake_decisions)

    result = backend.load_tickets()

    assert result is not None
    assert set(contracts.TICKET_COLUMNS) <= set(result.columns)
    assert len(result) == 2
    assert result["issue_id"].tolist() == ["payment_failed", "order_delay"]
    assert result["ground_truth_issue_id"].tolist() == ["order_delay", "payment_failed"]
    assert result["ticket_id"].tolist() == ["00100", "00101"]
    assert result["customer_id"].tolist() == ["00010", "00011"]
    assert result["priority"].isin(contracts.PRIORITIES).all()
    assert pd.api.types.is_datetime64_any_dtype(result["created_at"])


def test_configured_ticket_loader_rejects_missing_input_columns(tmp_path, monkeypatch):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"ticket_id": ["T-1"], "raw_text": ["payment failed"]}).to_csv(path, index=False)
    monkeypatch.setenv("SUPPORTAI_TICKETS_CSV", str(path))
    backend._read_inference_csv.cache_clear()
    backend._analyze_input_csv.cache_clear()

    with pytest.raises(ValueError, match="missing required input columns"):
        backend.load_tickets()


def test_configured_ticket_loader_keeps_rows_with_invalid_timestamps(tmp_path, monkeypatch):
    path = tmp_path / "incoming.csv"
    pd.DataFrame({
        "ticket_id": ["T-200"],
        "customer_id": ["C-20"],
        "timestamp": ["not-a-date"],
        "raw_text": ["UPI payment failed"],
    }).to_csv(path, index=False)
    monkeypatch.setenv("SUPPORTAI_TICKETS_CSV", str(path))
    backend._read_inference_csv.cache_clear()
    backend._analyze_input_csv.cache_clear()
    from types import SimpleNamespace
    fake = SimpleNamespace(
        raw_text="UPI payment failed",
        normalization=SimpleNamespace(normalized="upi payment failed"),
        classification=SimpleNamespace(issue_type="payment_failed", confidence=0.9),
        department_id="payments_billing",
        urgency=SimpleNamespace(priority="P3"),
        needs_human_review=False,
        entities={},
    )
    monkeypatch.setattr(backend, "analyze_dataframe", lambda *a, **k: [fake])

    result = backend.load_tickets()

    assert result is not None and len(result) == 1
    assert pd.isna(result.loc[0, "created_at"])
    assert "timestamp_warning" in result.attrs


def test_taxonomy_comes_from_the_real_config():
    assert backend.get_taxonomy().issue_ids() == load_taxonomy().issue_ids()


# --------------------------------------------------------------------------
# Demo data: isolated, labelled, in sync with the taxonomy
# --------------------------------------------------------------------------


def test_demo_templates_match_the_real_taxonomy():
    preview_data._check_templates(load_taxonomy())  # raises PreviewDataError on drift


def test_demo_template_drift_is_detected(monkeypatch):
    broken = dict(preview_data.TEMPLATES)
    broken.pop("double_charge")
    monkeypatch.setattr(preview_data, "TEMPLATES", broken)
    with pytest.raises(preview_data.PreviewDataError, match="double_charge"):
        preview_data.tickets(load_taxonomy())


def test_demo_tickets_satisfy_the_ticket_contract_and_use_real_ids():
    tx = load_taxonomy()
    df = preview_data.tickets(tx)
    assert set(contracts.TICKET_COLUMNS) <= set(df.columns)
    assert set(df["issue_id"]) <= set(tx.issue_ids())
    assert set(df["department_id"]) <= {d.id for d in tx.departments}
    assert set(df["priority"]) <= set(contracts.PRIORITIES)
    # 'other' must always go to human review (a taxonomy rule the demo must respect)
    other = df[df["issue_id"] == "other"]
    assert (other["department_id"] == "human_review").all() and other["needs_review"].all()


def test_demo_normalized_text_is_produced_by_the_real_normalizer():
    df = preview_data.tickets(load_taxonomy())
    normalizer = get_default_normalizer()
    sample = df.sample(40, random_state=1)
    assert [normalizer.normalize(t) for t in sample["raw_text"]] == list(sample["normalized_text"])


def test_demo_is_deterministic():
    tx = load_taxonomy()
    pd.testing.assert_frame_equal(preview_data.tickets(tx), preview_data.tickets(tx))


def test_demo_scenario_has_one_planted_alert_and_decoys_that_do_not_alert():
    tx = load_taxonomy()
    df = preview_data.tickets(tx)
    replay = preview_data.replay(tx, df)
    tl = replay.timeline
    alert_groups = set(tl.loc[tl["state"] == "alert", "group_id"])
    assert alert_groups == {"payment_checkout"}
    # the repeat-submitter decoy has high raw volume but very few distinct customers
    refunds = tl[tl["group_id"] == "refunds"]
    assert (refunds["raw_tickets"] - refunds["distinct_customers"]).max() >= 15
    assert (refunds["state"] != "alert").all()
    for inc in preview_data.incidents(tx, df, replay):
        assert any("does not confirm an outage" in line for line in inc.explanation)


def test_pages_never_import_the_demo_module_directly():
    offenders = []
    for path in (DASHBOARD / "pages").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""] + [a.name for a in node.names]
            elif isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            if any("preview_data" in n for n in names):
                offenders.append(path.name)
    assert not offenders, f"pages must get demo data only via data_access: {offenders}"


def test_dashboard_does_not_train_models_or_hardcode_taxonomy_ids():
    issue_ids = load_taxonomy().issue_ids()
    for path in DASHBOARD.rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                module = getattr(node, "module", None) or ""
                names = [module] + [a.name for a in node.names]
                assert not any(n.split(".")[0] in {"sklearn", "joblib"} for n in names), f"{path.name} imports an ML library"
        if path.name == "preview_data.py":
            continue  # demo templates are keyed by id and validated against the taxonomy above
        for issue_id in issue_ids:
            if issue_id == "other":
                continue
            assert f'"{issue_id}"' not in source and f"'{issue_id}'" not in source, f"{path.name} hardcodes {issue_id}"


# --------------------------------------------------------------------------
# data_access: live always wins, demo only on request
# --------------------------------------------------------------------------


def _live_tickets() -> pd.DataFrame:
    df = preview_data.tickets(load_taxonomy()).head(5)
    return df[list(contracts.TICKET_COLUMNS)].copy()


def test_live_mode_returns_no_data_when_nothing_is_connected():
    at = AppTest.from_string(
        "from dashboard import data_access\nfrom dashboard.shell import bootstrap\nimport streamlit as st\n"
        "bootstrap(preview=False)\n"
        "st.write(repr([data_access.tickets(), data_access.incidents(), data_access.replay(), data_access.evaluation()]))\n"
    ).run()
    assert not at.exception
    assert "[None, None, None, None]" in markdown_of(at)


def test_demo_data_is_always_marked_as_demo():
    at = run_page("overview", preview=True)
    assert has_demo_banner(at)
    assert "DEMO DATA" in markdown_of(at)


def test_live_data_wins_over_demo_even_when_preview_is_on(monkeypatch):
    monkeypatch.setattr(backend, "load_tickets", _live_tickets)
    at = AppTest.from_string(
        "from dashboard import data_access\nimport streamlit as st\nfrom dashboard.shell import bootstrap\n"
        "bootstrap(preview=True)\nsrc = data_access.tickets()\nst.write(f'demo={src.is_demo} rows={len(src.data)}')\n"
    )
    at.run()
    assert not at.exception
    assert "demo=False rows=5" in markdown_of(at)


def test_malformed_live_tickets_raise_a_clear_contract_error(monkeypatch):
    monkeypatch.setattr(backend, "load_tickets", lambda: pd.DataFrame({"ticket_id": ["T-1"]}))
    with pytest.raises(data_access.ContractError, match="missing columns"):
        data_access.tickets()


# --------------------------------------------------------------------------
# Escaping
# --------------------------------------------------------------------------


def test_esc_neutralises_html_markdown_and_latex():
    out = components.esc('<script>alert("x")</script> **b** _i_ $x$ \n second')
    assert "<" not in out and ">" not in out and '"' not in out
    assert "$" not in out and "*" not in out and "_" not in out and "\n" not in out


def test_components_never_emit_blank_lines_or_indented_html():
    html_snippets = [
        components.kpi_card("A", 1, "hint"), components.card("T", "body"), components.priority_badge("P1"),
        components.text_block("L", "line1\n\nline3"), components.confidence_bar(0.4, 0.5),
        components.stage_card(1, "T", "live", "body"),
    ]
    for snippet in html_snippets:
        assert "\n" not in snippet.strip()


def test_hostile_ticket_text_is_rendered_inert_on_the_analysis_page():
    hostile = '<img src=x onerror=alert(1)> **bold** $$math$$'
    at = AppTest.from_string(SCRIPT.format(page="analysis", preview=False)).run()
    at.text_area(key="analysis_text").set_value(hostile)
    at.button[0].click().run()
    assert not at.exception
    md = markdown_of(at)
    assert "<img" not in md
    assert "&lt;img" in md
    assert "**bold**" not in md  # asterisks are entity-encoded so Markdown cannot style them


# --------------------------------------------------------------------------
# Page smoke tests (live and demo)
# --------------------------------------------------------------------------


@pytest.mark.parametrize("preview", [False, True])
@pytest.mark.parametrize("page", PAGES)
def test_every_page_renders_without_exceptions(page, preview):
    at = run_page(page, preview)
    assert not at.exception, [str(e.value)[:300] for e in at.exception]


@pytest.mark.parametrize("page", PAGES)
def test_live_mode_shows_no_demo_content_and_says_not_connected(page):
    at = run_page(page, preview=False)
    md = markdown_of(at)
    assert not has_demo_banner(at)
    assert "DEMO DATA" not in md
    if page != "analysis":
        assert "connected" in md.lower()
    assert len(at.dataframe) == 0  # no ticket/metric tables without a backend


@pytest.mark.parametrize("page", [p for p in PAGES if p != "analysis"])
def test_demo_mode_banner_present_on_every_data_page(page):
    assert has_demo_banner(run_page(page, preview=True))


def test_incident_and_evaluation_pages_carry_their_standing_caveats():
    assert "do not confirm an outage" in markdown_of(run_page("incidents", False))
    assert "does not demonstrate real-world accuracy" in markdown_of(run_page("evaluation", False))
    assert "do not confirm an outage" in markdown_of(run_page("incidents", True))


# --------------------------------------------------------------------------
# Interaction tests
# --------------------------------------------------------------------------


def test_analysis_runs_live_normalization_and_marks_unconnected_stages():
    at = AppTest.from_string(
        SCRIPT.format(page="analysis", preview=False)
    ).run()

    at.selectbox(key="analysis_example").select(
        "Paisa kat gaya but order nahi aaya!!!"
    ).run()

    assert (
        at.text_area(key="analysis_text").value
        == "Paisa kat gaya but order nahi aaya!!!"
    )

    at.button[0].click().run()
    assert not at.exception

    md = markdown_of(at)

    expected = get_default_normalizer().normalize(
        "Paisa kat gaya but order nahi aaya!!!"
    )
    assert expected in md

    # Classification is now connected.
    assert "NOT CONNECTED" in md  # Some stages remain unconnected.

    # Urgency and playbooks are still unconnected.
    assert "Urgency" in md
    assert "Playbook" in md


def test_analysis_shows_real_extracted_entities_as_chips():
    at = AppTest.from_string(SCRIPT.format(page="analysis", preview=False)).run()
    at.text_area(key="analysis_text").set_value("Refund of ₹2,499 pending for order ORD123456 paid by UPI")
    at.button[0].click().run()
    assert not at.exception
    md = markdown_of(at)
    for expected in ("₹2,499", "ORD123456", "upi"):
        assert expected in md
    assert "Understand · extract entities" in md


def test_analysis_empty_input_shows_empty_state_not_a_result():
    at = AppTest.from_string(SCRIPT.format(page="analysis", preview=False)).run()
    at.button[0].click().run()
    assert "Nothing analysed yet" in markdown_of(at)


def test_explorer_search_filters_the_ticket_table():
    at = AppTest.from_string(SCRIPT.format(page="explorer", preview=True)).run()
    total = len(at.dataframe[0].value)
    at.text_input(key="ex_query").set_value("refund").run()
    assert not at.exception
    filtered = at.dataframe[0].value
    assert 0 < len(filtered) < total
    assert filtered["Message"].str.lower().str.contains("refund").all()


def test_explorer_no_match_shows_empty_state():
    at = AppTest.from_string(SCRIPT.format(page="explorer", preview=True)).run()
    at.text_input(key="ex_query").set_value("zzzz-no-such-text").run()
    assert "No tickets match these filters" in markdown_of(at)


def test_replay_jump_to_alert_shows_alert_state_for_the_planted_group():
    at = AppTest.from_string(SCRIPT.format(page="replay", preview=True)).run()
    at.radio(key="replay_group").set_value("payment_checkout").run()
    at.button(key="rp_alert").click().run()
    assert not at.exception
    kpi = re.search(r'Detector state</div><div class="value">([A-Z]+)<', markdown_of(at))
    assert kpi and kpi.group(1) == "ALERT"


def test_replay_decoy_group_never_reaches_alert():
    at = AppTest.from_string(SCRIPT.format(page="replay", preview=True)).run()
    at.radio(key="replay_group").set_value("refunds").run()
    assert at.button(key="rp_alert").disabled  # no alert exists for this group
    at.button(key="rp_end").click().run()
    kpi = re.search(r'Detector state</div><div class="value">([A-Z]+)<', markdown_of(at))
    assert kpi and kpi.group(1) != "ALERT"


def test_full_app_navigation_toggle_switches_between_live_and_demo():
    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    assert not at.exception
    assert not has_demo_banner(at)
    at.sidebar.toggle[0].set_value(True).run()
    assert not at.exception
    assert has_demo_banner(at)
    at.sidebar.toggle[0].set_value(False).run()
    assert not at.exception
    assert not has_demo_banner(at)
