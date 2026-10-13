"""TF-IDF + Logistic Regression classifier for SupportAI Radar."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from supportai.ingestion import load_tickets
from supportai.normalizer import get_default_normalizer


MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
DEFAULT_MODEL_PATH = MODEL_DIR / "ticket_classifier.joblib"


@dataclass(frozen=True)
class ClassificationResult:
    """Model prediction for one support ticket."""

    issue_type: str
    confidence: float
    probabilities: dict[str, float]


class TicketClassifier:
    """Explainable text classifier using word and character TF-IDF."""

    def __init__(self, confidence_threshold: float = 0.60) -> None:
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")

        self.confidence_threshold = confidence_threshold
        self.normalizer = get_default_normalizer()
        self.pipeline: Pipeline | None = None
        self.classes_: list[str] = []

    @property
    def is_trained(self) -> bool:
        return self.pipeline is not None

    def train(self, training_csv: str | Path) -> "TicketClassifier":
        """Train on ground-truth issue labels loaded from a CSV file."""

        df = load_tickets(training_csv)
        return self.train_from_dataframe(df)

    def train_from_dataframe(
        self,
        df: pd.DataFrame,
    ) -> "TicketClassifier":
        """Train directly from a dataframe containing ticket text and labels."""

        if df.empty:
            raise ValueError("Training dataframe cannot be empty.")

        if "raw_text" not in df.columns:
            raise ValueError("Training dataframe must contain 'raw_text'.")

        # Prefer explicitly preserved ground truth when available.
        if "ground_truth_issue_id" in df.columns:
            label_column = "ground_truth_issue_id"
        elif "issue_type" in df.columns:
            label_column = "issue_type"
        else:
            raise ValueError(
                "Training dataframe must contain 'ground_truth_issue_id' "
                "or 'issue_type'."
            )

        labels = df[label_column].astype("string")

        if labels.isna().any() or labels.str.strip().eq("").any():
            raise ValueError("Training labels cannot be missing or empty.")

        texts = [
            self.normalizer.normalize(text)
            for text in df["raw_text"].tolist()
        ]

        if any(not text.strip() for text in texts):
            raise ValueError(
                "Training data contains an empty normalized ticket message."
            )

        if labels.nunique() < 2:
            raise ValueError(
                "Training requires at least two distinct issue types."
            )

        features = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        analyzer="word",
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "character",
                    TfidfVectorizer(
                        analyzer="char",
                        ngram_range=(3, 5),
                        min_df=1,
                        sublinear_tf=True,
                        max_features=100_000,
                    ),
                ),
            ]
        )

        self.pipeline = Pipeline(
            [
                ("features", features),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        )

        self.pipeline.fit(texts, labels)
        self.classes_ = list(
            self.pipeline.named_steps["classifier"].classes_
        )

        return self

    def predict(self, raw_text: str) -> ClassificationResult:
        """Classify a raw support message."""

        if self.pipeline is None:
            raise RuntimeError("The classifier has not been trained.")

        if not isinstance(raw_text, str) or not raw_text.strip():
            raise ValueError("raw_text must be a non-empty string.")

        normalized = self.normalizer.normalize(raw_text)

        probabilities = self.pipeline.predict_proba([normalized])[0]
        best_index = int(probabilities.argmax())

        probability_map = {
            label: float(probability)
            for label, probability in zip(self.classes_, probabilities)
        }

        return ClassificationResult(
            issue_type=self.classes_[best_index],
            confidence=float(probabilities[best_index]),
            probabilities=probability_map,
        )

    def needs_review(self, result: ClassificationResult) -> bool:
        """Flag predictions that fall below the configured threshold."""

        return result.confidence < self.confidence_threshold

    def save(self, model_path: str | Path = DEFAULT_MODEL_PATH) -> Path:
        """Save the complete classifier, including its normalizer."""

        if self.pipeline is None:
            raise RuntimeError("Cannot save an untrained classifier.")

        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self, path)
        return path

    @classmethod
    def load(cls, model_path: str | Path = DEFAULT_MODEL_PATH) -> "TicketClassifier":
        """Load a previously saved classifier."""

        path = Path(model_path)

        if not path.is_file():
            raise FileNotFoundError(f"Model file not found: {path}")

        model = joblib.load(path)

        if not isinstance(model, cls):
            raise ValueError("The file does not contain a TicketClassifier.")

        if not model.is_trained:
            raise ValueError("The saved classifier has not been trained.")

        return model


def train_default_model(
    training_csv: str | Path,
    model_path: str | Path = DEFAULT_MODEL_PATH,
) -> TicketClassifier:
    """Train and save the default ticket classifier."""

    model = TicketClassifier()
    model.train(training_csv)
    model.save(model_path)
    return model