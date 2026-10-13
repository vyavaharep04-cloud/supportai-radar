"""Train and inspect the SupportAI Radar ticket classifier."""

import sys
from pathlib import Path

# Make the project root importable when running this script directly.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from supportai.classifier import TicketClassifier
from supportai.ingestion import load_tickets


TRAINING_FILE = ROOT / "data" / "generated" / "training_tickets.csv"
MODEL_FILE = ROOT / "models" / "ticket_classifier.joblib"


def main():
    print("Loading training dataset...")
    df = load_tickets(TRAINING_FILE)

    print(f"Training tickets: {len(df)}")
    print("\nGround-truth issue distribution:")
    print(df["ground_truth_issue_id"].value_counts().sort_index())

    print("\nTraining classifier...")
    model = TicketClassifier()
    model.train(TRAINING_FILE)

    model.save(MODEL_FILE)

    print(f"\nModel saved to: {MODEL_FILE}")
    print(f"Number of classes: {len(model.classes_)}")
    print("\nLearned issue types:")
    for issue in model.classes_:
        print(f"- {issue}")

    examples = [
        "My UPI payment failed but money was deducted!",
        "Bhai mera refund abhi tak nahi aaya",
        "I cannot log into my account",
        "Mere order ka delivery abhi tak nahi hua",
        "Someone accessed my account without permission",
    ]

    print("\nSample predictions:")
    for text in examples:
        result = model.predict(text)

        print(f"\nTicket: {text}")
        print(f"Predicted issue: {result.issue_type}")
        print(f"Model confidence: {result.confidence:.3f}")
        print(f"Needs review: {model.needs_review(result)}")


if __name__ == "__main__":
    main()