from pathlib import Path

import pandas as pd
import pytest

from supportai.pipeline import analyze_dataframe, analyze_text, get_pipeline_classifier
from supportai.taxonomy import load_taxonomy


@pytest.fixture
def classifier(tmp_path):
    from supportai.classifier import TicketClassifier

    rows = []
    examples = {
        "payment_failed": [
            "upi payment failed",
            "payment unsuccessful",
            "cannot pay by card",
        ],
        "order_delay": [
            "my delivery is late",
            "where is my order",
            "package has not arrived",
        ],
    }
    for issue_id, messages in examples.items():
        for i, message in enumerate(messages):
            rows.append({"raw_text": message, "issue_type": issue_id})
    return TicketClassifier().train_from_dataframe(pd.DataFrame(rows))


def test_analyze_text_returns_all_decision_stages(classifier):
    result = analyze_text(
        "My UPI payment failed for ₹1,299",
        classifier=classifier,
        taxonomy=load_taxonomy(),
    )
    assert result.raw_text == "My UPI payment failed for ₹1,299"
    assert result.normalization.normalized
    assert "amounts" in result.entities
    assert result.classification.issue_type in {"payment_failed", "order_delay"}
    assert 0 <= result.classification.confidence <= 1
    assert result.department_id
    assert result.urgency.priority in {"P1", "P2", "P3", "P4"}


def test_other_or_low_confidence_is_routed_to_human_review(classifier):
    result = analyze_text(
        "Completely unrelated words about purple mountains",
        classifier=classifier,
        taxonomy=load_taxonomy(),
    )
    if result.needs_human_review:
        assert result.department_id == "human_review"


def test_analyze_dataframe_matches_single_ticket_analysis(classifier):
    frame = pd.DataFrame(
        {"raw_text": ["upi payment failed", "my delivery is late"]}
    )
    batch = analyze_dataframe(frame, classifier=classifier, taxonomy=load_taxonomy())
    singles = [
        analyze_text(text, classifier=classifier, taxonomy=load_taxonomy())
        for text in frame["raw_text"]
    ]
    assert batch == singles


def test_analyze_dataframe_requires_text_column(classifier):
    with pytest.raises(ValueError, match="raw_text"):
        analyze_dataframe(pd.DataFrame({"message": ["hello"]}), classifier=classifier)


def test_analyze_dataframe_rejects_blank_rows(classifier):
    with pytest.raises(ValueError, match="row"):
        analyze_dataframe(
            pd.DataFrame({"raw_text": ["payment failed", "  "]}),
            classifier=classifier,
        )


def test_default_classifier_accessor_loads_local_model_if_present():
    # The app's default model is an ignored local artifact, so skip this
    # environment-specific assertion in a clean clone.
    from supportai.classifier import DEFAULT_MODEL_PATH

    if not Path(DEFAULT_MODEL_PATH).is_file():
        pytest.skip("default model has not been trained in this checkout")
    assert get_pipeline_classifier().is_trained
