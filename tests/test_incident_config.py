"""Tests for config/incidents.yaml and supportai.incidents.load_incident_params."""

from __future__ import annotations

import inspect

import pandas as pd
import pytest
import yaml

from supportai.config import ConfigError
from supportai.incidents import (
    DEFAULT_SETTINGS_PATH,
    SETTING_NAMES,
    detect_incidents,
    load_incident_params,
)

VALID = {
    "window_minutes": 30,
    "min_customers": 25,
    "watch_ratio": 2.0,
    "alert_ratio": 3.0,
    "min_baseline_windows": 3,
}


def write(tmp_path, data, name="incidents.yaml"):
    path = tmp_path / name
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_shipped_config_is_at_the_expected_path():
    assert DEFAULT_SETTINGS_PATH.name == "incidents.yaml"
    assert DEFAULT_SETTINGS_PATH.parent.name == "config"
    assert DEFAULT_SETTINGS_PATH.is_file()


def test_shipped_config_has_the_agreed_initial_values():
    params = load_incident_params()
    assert params == VALID
    assert isinstance(params["min_customers"], int)
    assert isinstance(params["watch_ratio"], float)
    assert isinstance(params["alert_ratio"], float)


def test_only_min_customers_differs_from_the_detectors_built_in_defaults():
    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(detect_incidents).parameters.items()
        if parameter.default is not inspect.Parameter.empty
    }
    assert set(defaults) == set(SETTING_NAMES)
    differing = {name for name, value in load_incident_params().items() if defaults[name] != value}
    assert differing == {"min_customers"}


def test_loaded_settings_are_accepted_by_the_detector():
    tickets = pd.DataFrame(
        {"ticket_id": ["T1"], "customer_id": ["C1"], "created_at": [pd.Timestamp("2026-10-10")], "group_id": ["refunds"]}
    )
    assert detect_incidents(tickets, **load_incident_params()) == []


def test_integer_ratios_are_returned_as_floats(tmp_path):
    params = load_incident_params(write(tmp_path, {**VALID, "watch_ratio": 2, "alert_ratio": 3}))
    assert params["watch_ratio"] == 2.0 and isinstance(params["watch_ratio"], float)
    assert params["alert_ratio"] == 3.0 and isinstance(params["alert_ratio"], float)


def test_custom_path_values_are_used(tmp_path):
    assert load_incident_params(write(tmp_path, {**VALID, "min_customers": 12}))["min_customers"] == 12


@pytest.mark.parametrize("name", SETTING_NAMES)
def test_every_setting_is_required(tmp_path, name):
    data = {key: value for key, value in VALID.items() if key != name}
    with pytest.raises(ConfigError, match=name):
        load_incident_params(write(tmp_path, data))


def test_unknown_settings_are_rejected_so_typos_are_not_ignored(tmp_path):
    data = {**VALID, "min_customer": 25}
    with pytest.raises(ConfigError, match="min_customer"):
        load_incident_params(write(tmp_path, data))


@pytest.mark.parametrize(
    "overrides",
    [
        {"min_customers": 0},
        {"min_customers": -1},
        {"min_customers": True},
        {"min_customers": "25"},
        {"min_customers": 25.0},
        {"window_minutes": 0},
        {"window_minutes": 30.5},
        {"min_baseline_windows": 0},
        {"watch_ratio": 0.5},
        {"watch_ratio": 4.0, "alert_ratio": 3.0},
        {"watch_ratio": float("nan")},
        {"alert_ratio": float("inf")},
        {"alert_ratio": "3.0"},
        {"alert_ratio": None},
    ],
    ids=lambda overrides: ",".join(f"{k}={v!r}" for k, v in overrides.items()),
)
def test_invalid_values_raise_config_error(tmp_path, overrides):
    path = tmp_path / "incidents.yaml"
    # safe_dump writes nan/inf as .nan/.inf, which PyYAML reads back as floats
    path.write_text(yaml.safe_dump({**VALID, **overrides}), encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid incident setting"):
        load_incident_params(path)


def test_config_errors_name_the_file(tmp_path):
    path = write(tmp_path, {**VALID, "min_customers": 0})
    with pytest.raises(ConfigError, match="incidents.yaml"):
        load_incident_params(path)


def test_missing_malformed_and_non_mapping_files_raise_config_error(tmp_path):
    with pytest.raises(ConfigError):
        load_incident_params(tmp_path / "does_not_exist.yaml")
    broken = tmp_path / "broken.yaml"
    broken.write_text("window_minutes: [30\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_incident_params(broken)
    with pytest.raises(ConfigError):
        load_incident_params(write(tmp_path, ["not", "a", "mapping"], name="list.yaml"))
