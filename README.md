# SupportAI Radar

**Intelligent Multilingual Support Routing & Incident Detection**

SupportAI Radar is a local-first prototype for support ticket triage in English and Hinglish. It normalizes messages, extracts useful entities, predicts issue categories, routes uncertain cases for human review, scores urgency, suggests agent-assist replies, and flags unusual ticket-volume patterns as *potential incidents*.

## Implemented capabilities

- Hinglish text normalization with an explainable lexicon
- Entity extraction from raw ticket text
- ML issue classification and confidence-based human-review routing
- Taxonomy-based department mapping
- Explainable urgency and priority scoring
- Issue-specific reply suggestions and action checklists
- Potential-incident detection using distinct-customer counts and historical windows
- Evidence-based incident explanations and chronological replay
- Held-out synthetic-template classifier evaluation and human-review threshold trade-offs
- Streamlit dashboard with clearly separated live-input and synthetic-preview modes
- Automated unit and dashboard tests

## Local setup (Windows / PowerShell)

Run these commands from the repository root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m pytest -q
streamlit run app.py
```

If PowerShell blocks virtual-environment activation, use the Python executable directly:
`\.venv\Scripts\python.exe -m pytest -q`.

## Connect a ticket CSV

Set `SUPPORTAI_TICKETS_CSV` to a CSV file containing these required columns:

- `ticket_id` — unique ticket identifier
- `customer_id` — stable customer identifier used for distinct-customer counts
- `timestamp` — ticket creation timestamp
- `raw_text` — original customer message

Identifiers are read as strings to preserve leading zeroes. Optional source labels are not treated as model predictions. The dashboard generates its own predictions from `raw_text`.

Example PowerShell session:

```powershell
$env:SUPPORTAI_TICKETS_CSV = "data/generated/replay_tickets.csv"
streamlit run app.py
```

To clear the setting for the current terminal session:

```powershell
Remove-Item Env:SUPPORTAI_TICKETS_CSV
```

Use the sidebar's **Design preview** toggle to explore clearly labelled synthetic demo data. Live ticket data takes precedence over preview data.

## Evaluation and limitations

Classifier evaluation uses held-out synthetic templates. Templates are separated before variations are generated to reduce template-level leakage, but these metrics **do not establish real-world accuracy**. Priority ground-truth labels are not available, so critical-ticket precision and recall are not reported.

Incident alerts are inferred from support-ticket patterns. They **do not confirm a real service outage or establish its root cause**. Baselines require sufficient consecutive historical windows; sparse or invalid timestamp data can limit detection.

Playbooks are suggestions for human agents. They do not execute refunds, change orders, authenticate customers, or guarantee outcomes. Never collect passwords, PINs, OTPs, or full payment credentials in ticket text.

This is a local prototype, not a production incident-management or customer-support system. Validate the classifier and operating thresholds on independently labelled, representative real-world data before operational use.
