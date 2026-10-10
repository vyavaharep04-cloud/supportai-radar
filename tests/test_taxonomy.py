"""Tests for the SupportAI Radar taxonomy."""

import pytest

from supportai.config import ConfigError
from supportai.taxonomy import load_taxonomy


EXPECTED_ISSUE_IDS = [
    "payment_failed",
    "money_deducted_no_order",
    "double_charge",
    "checkout_error",
    "refund_pending",
    "unauthorized_activity",
    "login_problem",
    "order_delay",
    "wrong_damaged_item",
    "cancel_modify_order",
    "other",
]


def test_default_taxonomy_loads():
    taxonomy = load_taxonomy()

    assert taxonomy.version == 1
    assert len(taxonomy.departments) == 5
    assert len(taxonomy.issue_types) == 11


def test_issue_ids_match_frozen_taxonomy():
    taxonomy = load_taxonomy()

    assert taxonomy.issue_ids() == EXPECTED_ISSUE_IDS


def test_all_department_references_resolve():
    taxonomy = load_taxonomy()
    department_ids = {department.id for department in taxonomy.departments}

    for issue in taxonomy.issue_types:
        assert issue.department in department_ids


def test_department_for_payment_issue():
    taxonomy = load_taxonomy()

    department = taxonomy.department_for("money_deducted_no_order")

    assert department.id == "payments_billing"
    assert department.display_name == "Payments & Billing"


def test_checkout_error_uses_payment_incident_group():
    taxonomy = load_taxonomy()

    issue = taxonomy.get_issue("checkout_error")

    assert issue.incident_group == "payment_checkout"


def test_other_always_goes_to_human_review():
    taxonomy = load_taxonomy()

    issue = taxonomy.get_issue("other")
    department = taxonomy.department_for("other")

    assert department.id == "human_review"
    assert issue.incident_group is None


def test_unknown_issue_raises_key_error():
    taxonomy = load_taxonomy()

    with pytest.raises(KeyError):
        taxonomy.get_issue("unknown_issue")


def test_duplicate_department_id_is_rejected(tmp_path):
    taxonomy = {
        "version": 1,
        "departments": [
            {"id": "payments_billing", "display_name": "Payments"},
            {"id": "payments_billing", "display_name": "Billing"},
        ],
        "issue_types": [
            {
                "id": "other",
                "display_name": "Other",
                "description": "Unclassified ticket",
                "department": "payments_billing",
                "incident_group": None,
            }
        ],
    }

    path = tmp_path / "duplicate_department.yaml"

    import yaml

    path.write_text(yaml.safe_dump(taxonomy), encoding="utf-8")

    with pytest.raises(ConfigError, match="Duplicate department ID"):
        load_taxonomy(path)


def test_duplicate_issue_id_is_rejected(tmp_path):
    taxonomy = {
        "version": 1,
        "departments": [
            {"id": "human_review", "display_name": "Human Review"}
        ],
        "issue_types": [
            {
                "id": "other",
                "display_name": "Other",
                "description": "Unclassified ticket",
                "department": "human_review",
                "incident_group": None,
            },
            {
                "id": "other",
                "display_name": "Other again",
                "description": "Duplicate",
                "department": "human_review",
                "incident_group": None,
            },
        ],
    }

    path = tmp_path / "duplicate_issue.yaml"

    import yaml

    path.write_text(yaml.safe_dump(taxonomy), encoding="utf-8")

    with pytest.raises(ConfigError, match="Duplicate issue type ID"):
        load_taxonomy(path)


def test_unknown_department_reference_is_rejected(tmp_path):
    taxonomy = {
        "version": 1,
        "departments": [
            {"id": "human_review", "display_name": "Human Review"}
        ],
        "issue_types": [
            {
                "id": "other",
                "display_name": "Other",
                "description": "Unclassified ticket",
                "department": "missing_department",
                "incident_group": None,
            }
        ],
    }

    path = tmp_path / "unknown_department.yaml"

    import yaml

    path.write_text(yaml.safe_dump(taxonomy), encoding="utf-8")

    with pytest.raises(ConfigError, match="unknown department"):
        load_taxonomy(path)


def test_other_cannot_belong_to_another_department(tmp_path):
    taxonomy = {
        "version": 1,
        "departments": [
            {"id": "payments_billing", "display_name": "Payments"}
        ],
        "issue_types": [
            {
                "id": "other",
                "display_name": "Other",
                "description": "Unclassified ticket",
                "department": "payments_billing",
                "incident_group": None,
            }
        ],
    }

    path = tmp_path / "invalid_other_department.yaml"

    import yaml

    path.write_text(yaml.safe_dump(taxonomy), encoding="utf-8")

    with pytest.raises(ConfigError, match="must belong"):
        load_taxonomy(path)


def test_other_cannot_have_incident_group(tmp_path):
    taxonomy = {
        "version": 1,
        "departments": [
            {"id": "human_review", "display_name": "Human Review"}
        ],
        "issue_types": [
            {
                "id": "other",
                "display_name": "Other",
                "description": "Unclassified ticket",
                "department": "human_review",
                "incident_group": "payment_checkout",
            }
        ],
    }

    path = tmp_path / "invalid_other_incident.yaml"

    import yaml

    path.write_text(yaml.safe_dump(taxonomy), encoding="utf-8")

    with pytest.raises(ConfigError, match="incident_group set to null"):
        load_taxonomy(path)