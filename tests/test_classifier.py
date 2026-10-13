import pandas as pd
import pytest

from supportai.classifier import TicketClassifier


@pytest.fixture
def training_csv(tmp_path):
    path = tmp_path / "training.csv"

    rows = []

    examples = {
        "payment_failed": [
            "payment failed while paying with UPI",
            "upi payment did not work",
            "payment unsuccessful please help",
        ],
        "order_delay": [
            "my order has not arrived",
            "delivery is late",
            "where is my delayed order",
        ],
        "login_problem": [
            "I cannot log into my account",
            "login is not working",
            "unable to sign in",
        ],
    }

    for issue_type, messages in examples.items():
        for index, message in enumerate(messages):
            rows.append(
                {
                    "ticket_id": f"{issue_type}_{index}",
                    "customer_id": f"C_{issue_type}_{index}",
                    "timestamp": "",
                    "raw_text": message,
                    "issue_type": issue_type,
                    "department": "test_department",
                    "incident_group": "test_group",
                    "scenario": "background",
                    "is_incident_ticket": False,
                }
            )

    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_classifier_trains(training_csv):
    model = TicketClassifier().train(training_csv)

    assert model.is_trained
    assert set(model.classes_) == {
        "payment_failed",
        "order_delay",
        "login_problem",
    }


def test_prediction_returns_valid_result(training_csv):
    model = TicketClassifier().train(training_csv)

    result = model.predict("My UPI payment failed")

    assert result.issue_type in model.classes_
    assert 0.0 <= result.confidence <= 1.0
    assert abs(sum(result.probabilities.values()) - 1.0) < 1e-6


def test_untrained_model_cannot_predict():
    model = TicketClassifier()

    with pytest.raises(RuntimeError, match="not been trained"):
        model.predict("Payment failed")


def test_empty_message_is_rejected(training_csv):
    model = TicketClassifier().train(training_csv)

    with pytest.raises(ValueError, match="non-empty"):
        model.predict("   ")


def test_review_threshold(training_csv):
    model = TicketClassifier(confidence_threshold=0.60).train(training_csv)

    from supportai.classifier import ClassificationResult

    low_confidence = ClassificationResult(
        issue_type="other",
        confidence=0.40,
        probabilities={"other": 0.40, "payment_failed": 0.60},
    )

    assert model.needs_review(low_confidence)


def test_save_and_reload(training_csv, tmp_path):
    model_path = tmp_path / "classifier.joblib"

    original = TicketClassifier().train(training_csv)
    original.save(model_path)

    loaded = TicketClassifier.load(model_path)

    assert loaded.is_trained
    assert loaded.predict("My UPI payment failed") == original.predict(
        "My UPI payment failed"
    )