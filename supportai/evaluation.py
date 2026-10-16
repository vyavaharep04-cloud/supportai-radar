"""Held-out evaluation for SupportAI Radar."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from scripts.evaluate_classifier import build_evaluation_data
from scripts.generate_synthetic_data import TEMPLATES
from supportai.classifier import TicketClassifier


EVALUATION_SEED = 20261012
DEFAULT_THRESHOLDS = (0.40, 0.50, 0.60, 0.70, 0.80)


def evaluate_classifier(
    thresholds: tuple[float, ...] = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    """Evaluate a freshly trained classifier on held-out templates.

    The split is deterministic. Templates are separated before text
    variations are generated, reducing template-level data leakage.

    This evaluates synthetic templates, not real-world ticket performance.
    """

    train_df, test_df = build_evaluation_data()

    model = TicketClassifier()
    model.train_from_dataframe(train_df)

    predictions = [
        model.predict(text)
        for text in test_df["raw_text"].tolist()
    ]

    truth = test_df["issue_type"].tolist()
    predicted_labels = [prediction.issue_type for prediction in predictions]
    labels = sorted(TEMPLATES)

    report = classification_report(
        truth,
        predicted_labels,
        labels=labels,
        output_dict=True,
        zero_division=0,
    )

    per_class = pd.DataFrame(
        [
            {
                "issue_id": label,
                "precision": report[label]["precision"],
                "recall": report[label]["recall"],
                "f1": report[label]["f1-score"],
                "support": int(report[label]["support"]),
            }
            for label in labels
        ]
    )

    matrix = confusion_matrix(
        truth,
        predicted_labels,
        labels=labels,
    )

    confusion = pd.DataFrame(
        matrix,
        index=labels,
        columns=labels,
    )
    confusion.index.name = "true_issue_id"
    confusion.columns.name = "predicted_issue_id"

    # A ticket is auto-routed only if confidence meets the threshold.
    # All other tickets require human review.
    confidence = [prediction.confidence for prediction in predictions]
    threshold_rows = []

    for threshold in thresholds:
        auto_indices = [
            index
            for index, score in enumerate(confidence)
            if score >= threshold
        ]

        auto_accuracy = (
            sum(
                truth[index] == predicted_labels[index]
                for index in auto_indices
            )
            / len(auto_indices)
            if auto_indices
            else None
        )

        threshold_rows.append(
            {
                "threshold": threshold,
                "review_rate": (
                    1 - len(auto_indices) / len(test_df)
                ),
                "auto_accuracy": auto_accuracy,
                "auto_routed_count": len(auto_indices),
            }
        )

    return {
        "accuracy": float(accuracy_score(truth, predicted_labels)),
        "macro_f1": float(
            f1_score(
                truth,
                predicted_labels,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "per_class": per_class,
        "confusion": confusion,
        "thresholds": pd.DataFrame(threshold_rows),
        "critical_recall": None,
        "critical_precision": None,
        "training_samples": len(train_df),
        "test_samples": len(test_df),
        "issue_classes": len(labels),
        "dataset_note": (
            "Synthetic held-out template evaluation. Templates are split "
            "before variations are generated. Results do not establish "
            "real-world accuracy. Priority ground-truth labels are "
            "unavailable, so critical-ticket precision and recall "
            "cannot be calculated."
        ),
    }