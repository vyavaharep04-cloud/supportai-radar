# SupportAI Radar

### Intelligent Multilingual Support Routing & Incident Detection

SupportAI Radar is a local-first support triage prototype that
processes English and Hinglish customer support messages.

It helps support teams classify customer issues, assign urgency,
recommend responses, and identify potential incidents when multiple
customers report related problems.

## Core Features

- Hinglish text normalization
- Rule-based entity extraction
- Machine-learning-based issue classification
- Department routing
- Urgency and risk scoring
- Human review for uncertain predictions
- Suggested replies and action checklists
- Explainable ticket decisions
- Incident detection using unique customer counts
- Simulated incident replay
- Model evaluation and automated tests

## Technology Stack

- Python
- pandas and NumPy
- scikit-learn
- Streamlit
- PyYAML
- pytest

## Execution

The prototype is designed to run locally on Windows using Streamlit.

It does not require paid APIs, cloud deployment, or a GPU.

## Project Status

Currently under development.

## Important Limitations

Incident alerts are based on support-ticket patterns and do not
confirm an actual service outage or establish its root cause.

Synthetic data is used for prototype development and testing.
Evaluation results should not be interpreted as proof of
real-world performance.