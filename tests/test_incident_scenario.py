"""End-to-end incident tests on the project's generated replay scenario.

Chain under test: generator -> classifier -> pipeline -> PREDICTED incident group
-> detector with the settings from config/incidents.yaml.

The detector only ever receives ticket_id, customer_id, created_at and the
predicted group. Ground truth (the generator's `scenario` column) is used only
to label the outcome. A small classifier is trained in memory from the
project's own training generator, so these tests do not depend on the
gitignored models/ directory.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pandas as pd
import pytest

from dashboard import backend
from scripts.generate_synthetic_data import (
    RANDOM_SEED,
    generate_replay_tickets,
    generate_training_tickets,
)
from supportai import pipeline
from supportai.classifier import TicketClassifier
from supportai.incidents import detect_incidents, load_incident_params
from supportai.taxonomy import load_taxonomy

PLANTED = "planted_payment_incident"
PROMO_DECOY = "decoy_order_delay_promo"
REPEAT_DECOY = "decoy_single_customer_repeat"
PLANTED_GROUP = "payment_checkout"
SEEDS = (RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2)

# Verified band for min_customers on this scenario (see config/incidents.yaml).
BAND = range(22, 31)


@pytest.fixture(scope="module")
def classifier():
    model = TicketClassifier()
    model.train_from_dataframe(generate_training_tickets())
    return model


def build_scenario(model, seed):
    """Return the source rows and the detector input built from PREDICTED groups."""
    taxonomy = load_taxonomy()
    source = generate_replay_tickets(seed=seed)
    decisions = pipeline.analyze_dataframe(source, classifier=model, taxonomy=taxonomy)
    frame = pd.DataFrame(
        {
            "ticket_id": source["ticket_id"],
            "customer_id": source["customer_id"],
            "created_at": pd.to_datetime(source["timestamp"], utc=True),
            "group_id": [
                taxonomy.get_issue(d.classification.issue_type).incident_group for d in decisions
            ],
        }
    )
    planted = pd.to_datetime(source.loc[source["scenario"] == PLANTED, "timestamp"], utc=True)
    return SimpleNamespace(
        seed=seed,
        source=source,
        frame=frame,
        planted_start=planted.min(),
        planted_end=planted.max() + pd.Timedelta(minutes=1),
    )


@pytest.fixture(scope="module")
def scenarios(classifier):
    return {seed: build_scenario(classifier, seed) for seed in SEEDS}


@pytest.fixture(scope="module")
def scenario(scenarios):
    return scenarios[RANDOM_SEED]


def configured(scenario):
    return detect_incidents(scenario.frame, **load_incident_params())


def in_planted_incident(incident, scenario):
    return (
        incident.group_id == PLANTED_GROUP
        and incident.window_end > scenario.planted_start
        and incident.window_start < scenario.planted_end
    )


def false_flags(incidents, scenario):
    return [i for i in incidents if not in_planted_incident(i, scenario)]


# --------------------------------------------------------------------------
# Detection quality with the configured settings
# --------------------------------------------------------------------------


def test_planted_payment_incident_is_detected_as_an_alert(scenario):
    alerts = [i for i in configured(scenario) if i.status == "alert" and in_planted_incident(i, scenario)]
    assert alerts, "the planted payment incident was missed"


def test_detection_delay_is_one_window_on_the_project_seed(scenario):
    alerts = [i for i in configured(scenario) if i.status == "alert" and in_planted_incident(i, scenario)]
    delay = min(i.window_end for i in alerts) - scenario.planted_start
    assert delay == pd.Timedelta(minutes=30)


def test_every_flagged_window_belongs_to_the_planted_incident(scenario):
    incidents = configured(scenario)
    assert incidents, "nothing was flagged at all"
    assert false_flags(incidents, scenario) == [], "alerts or watches appeared outside the planted incident"


def test_order_delay_decoy_burst_does_not_alert(scenario):
    source, frame = scenario.source, scenario.frame
    decoy = frame[source["scenario"] == PROMO_DECOY]
    # Precondition, so this test cannot pass vacuously: the decoy really is a burst of many customers.
    per_window = decoy.groupby(decoy["created_at"].dt.floor("30min"))["customer_id"].nunique()
    assert per_window.max() >= 14
    incidents = configured(scenario)
    assert not [i for i in incidents if i.group_id == "orders_delivery"]
    decoy_ids = set(decoy["ticket_id"])
    assert not [i for i in incidents if decoy_ids & set(i.ticket_ids)]


def test_single_customer_repeat_decoy_is_ignored(scenario):
    source, frame = scenario.source, scenario.frame
    decoy = frame[source["scenario"] == REPEAT_DECOY]
    assert len(decoy) >= 8 and decoy["customer_id"].nunique() == 1  # precondition
    decoy_ids = set(decoy["ticket_id"])
    assert not [i for i in configured(scenario) if decoy_ids & set(i.ticket_ids)]


@pytest.mark.parametrize("seed", SEEDS)
def test_detection_is_robust_across_generator_seeds(scenarios, seed):
    scenario = scenarios[seed]
    incidents = configured(scenario)
    assert any(i.status == "alert" and in_planted_incident(i, scenario) for i in incidents)
    assert false_flags(incidents, scenario) == []
    first_alert = min(i.window_end for i in incidents if i.status == "alert" and in_planted_incident(i, scenario))
    assert first_alert - scenario.planted_start <= pd.Timedelta(minutes=60)


@pytest.mark.parametrize("min_customers", BAND)
def test_every_value_in_the_verified_band_works(scenario, min_customers):
    incidents = detect_incidents(scenario.frame, **{**load_incident_params(), "min_customers": min_customers})
    assert any(i.status == "alert" and in_planted_incident(i, scenario) for i in incidents)
    assert false_flags(incidents, scenario) == []


def test_configured_value_sits_inside_the_verified_band():
    assert load_incident_params()["min_customers"] in BAND


def test_detector_defaults_alone_would_raise_false_alerts_on_this_scenario(scenario):
    """Documents why the configuration exists: the detector's built-in min_customers=5
    flags decoy and background windows as alerts on this scenario."""
    incidents = detect_incidents(scenario.frame)
    assert any(i.status == "alert" for i in false_flags(incidents, scenario))


def test_detection_is_deterministic_and_independent_of_row_order(scenario):
    def key(incidents):
        return sorted((i.group_id, i.window_start, i.status, i.unique_customers, i.raw_tickets) for i in incidents)

    first = key(configured(scenario))
    assert key(configured(scenario)) == first
    shuffled = scenario.frame.sample(frac=1.0, random_state=7).reset_index(drop=True)
    assert key(detect_incidents(shuffled, **load_incident_params())) == first


# --------------------------------------------------------------------------
# The dashboard's live path
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_scenario(tmp_path_factory, classifier):
    """Point the dashboard at the replay CSV, scored by the in-memory classifier.

    Module-scoped so the 600 tickets are scored once for all live-path tests;
    the environment variable, patch and caches are restored afterwards.
    """
    path = tmp_path_factory.mktemp("live") / "replay_tickets.csv"
    generate_replay_tickets().to_csv(path, index=False)
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("SUPPORTAI_TICKETS_CSV", str(path))
        patch.setattr(pipeline, "get_pipeline_classifier", lambda: classifier)
        backend._read_inference_csv.cache_clear()
        backend._analyze_input_csv.cache_clear()
        yield
    backend._read_inference_csv.cache_clear()
    backend._analyze_input_csv.cache_clear()


def test_live_incident_path_passes_the_configured_settings(monkeypatch):
    seen = {}

    def spy(tickets, **kwargs):
        seen.update(kwargs)
        return []

    monkeypatch.setattr("supportai.incidents.detect_incidents", spy)
    monkeypatch.setattr(
        backend,
        "load_tickets",
        lambda: pd.DataFrame(
            {"ticket_id": ["T1"], "customer_id": ["C1"], "created_at": [pd.Timestamp("2026-10-10")], "group_id": ["refunds"]}
        ),
    )
    assert backend.load_incidents() == []
    assert seen == load_incident_params()
    assert seen["min_customers"] == 25


def test_live_incident_path_finds_only_the_planted_incident(live_scenario, scenario):
    views = backend.load_incidents()
    assert views, "the live path returned no incidents"
    assert {v.group_id for v in views} == {PLANTED_GROUP}
    assert any(v.status == "alert" for v in views)
    for view in views:
        assert view.window_end > scenario.planted_start and view.window_start < scenario.planted_end


def test_live_path_matches_the_detector_run_directly(live_scenario, scenario):
    live = sorted((v.group_id, v.window_start, v.status, v.unique_customers) for v in backend.load_incidents())
    direct = sorted((i.group_id, i.window_start, i.status, i.unique_customers) for i in configured(scenario))
    assert live == direct


def test_live_incident_path_ignores_ground_truth_labels(live_scenario, tmp_path, monkeypatch):
    """Predictions, not the dataset's labels, must drive incident groups.

    Every ground-truth label column is rotated so it is wrong for almost every
    row. The incidents must be identical to those from the correctly labelled CSV.
    """
    source = generate_replay_tickets()
    for column in ("issue_type", "department", "incident_group"):
        source[column] = source[column].shift(37).fillna(source[column].iloc[-1]).to_numpy()
    scrambled = tmp_path / "scrambled_labels.csv"
    source.to_csv(scrambled, index=False)

    def summary(views):
        return sorted((v.group_id, v.window_start, v.status, v.unique_customers) for v in views)

    correct = summary(backend.load_incidents())
    monkeypatch.setenv("SUPPORTAI_TICKETS_CSV", str(scrambled))
    backend._read_inference_csv.cache_clear()
    backend._analyze_input_csv.cache_clear()
    assert summary(backend.load_incidents()) == correct
    backend._read_inference_csv.cache_clear()
    backend._analyze_input_csv.cache_clear()


def test_incident_center_shows_one_alert_for_the_scenario(live_scenario):
    from streamlit.testing.v1 import AppTest

    script = (
        "from dashboard.shell import bootstrap\n"
        "from dashboard.pages import incidents\n"
        "bootstrap(preview=False)\n"
        "incidents.render()\n"
    )
    at = AppTest.from_string(script, default_timeout=120).run()
    assert not at.exception
    html = " ".join(block.value for block in at.markdown)
    alerts = re.search(r'Active alerts</div><div class="value">(\d+)<', html)
    assert alerts and alerts.group(1) == "1"
    assert 'class="sr-demo-banner"' not in html  # live data, not the demo preview
