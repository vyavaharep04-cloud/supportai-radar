"""Evaluate SupportAI Radar using held-out base templates."""

import random
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ROOT = Path(__file__).resolve().parents[1]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.generate_synthetic_data import (
    TEMPLATES,
    _fill_template,
    _maps,
)
from supportai.classifier import TicketClassifier


SEED = 20261012
TRAIN_TEMPLATES_PER_CLASS = 7


def build_evaluation_data():
    """Split base templates before generating text variations."""

    rng = random.Random(SEED)
    departments, groups = _maps()

    train_rows = []
    test_rows = []

    for issue_id, templates in TEMPLATES.items():
        shuffled = list(templates)
        rng.shuffle(shuffled)

        train_templates = shuffled[:TRAIN_TEMPLATES_PER_CLASS]
        test_templates = shuffled[TRAIN_TEMPLATES_PER_CLASS:]

        for split, selected_templates, destination in (
            ("train", train_templates, train_rows),
            ("test", test_templates, test_rows),
        ):
            for index, template in enumerate(selected_templates):
                template_id = f"{issue_id}:{template}"

                for variation in range(8):
                    variation_seed = (
                        SEED
                        + sum(ord(char) for char in template_id) * 1009
                        + variation * 17
                    )

                    text = _fill_template(
                        template,
                        random.Random(variation_seed),
                    )

                    destination.append(
                        {
                            "ticket_id": (
                                f"{split}-{issue_id}-{index}-{variation}"
                            ),
                            "customer_id": (
                                f"{split}-customer-{issue_id}-{index}-{variation}"
                            ),
                            "timestamp": "",
                            "raw_text": text,
                            "issue_type": issue_id,
                            "department": departments[issue_id],
                            "incident_group": groups[issue_id] or "",
                            "scenario": f"evaluation_{split}",
                            "is_incident_ticket": False,
                        }
                    )

    return pd.DataFrame(train_rows), pd.DataFrame(test_rows)


def main():
    train_df, test_df = build_evaluation_data()

    print("Held-out template evaluation")
    print("=" * 40)
    print(f"Training examples: {len(train_df)}")
    print(f"Test examples: {len(test_df)}")
    print(f"Issue types: {train_df['issue_type'].nunique()}")

    model = TicketClassifier()
    model.train_from_dataframe(train_df)  # Added in the next step.

    predictions = [
        model.predict(text).issue_type
        for text in test_df["raw_text"]
    ]

    truth = test_df["issue_type"].tolist()

    print(f"\nAccuracy: {accuracy_score(truth, predictions):.3f}")
    print(f"Macro-F1: {f1_score(truth, predictions, average='macro', zero_division=0):.3f}")

    print("\nClassification report:")
    print(
        classification_report(
            truth,
            predictions,
            labels=sorted(TEMPLATES),
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        truth,
        predictions,
        labels=sorted(TEMPLATES),
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=sorted(TEMPLATES),
        columns=sorted(TEMPLATES),
    )

    output_dir = ROOT / "data" / "metrics"
    output_dir.mkdir(parents=True, exist_ok=True)

    matrix_df.to_csv(output_dir / "heldout_confusion_matrix.csv")

    print(f"Confusion matrix saved to: {output_dir}")


if __name__ == "__main__":
    main()