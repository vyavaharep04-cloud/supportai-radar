# SupportAI Radar dashboard

Streamlit front end for the triage and incident-detection engine. It is a
presentation layer only: it contains no classification, scoring or detection logic.

## Run it (Windows, from the repository root)

```powershell
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Tests (the dashboard tests use Streamlit's headless `AppTest`, no browser needed):

```powershell
python -m pytest
```

Developed and tested against Streamlit 1.65. The features used (`st.navigation`,
material icons, dataframe row selection, `[theme.sidebar]`) need roughly 1.40+;
older versions are untested. No dependency was added: charts use Altair, which
Streamlit already installs.

## Live mode and Design preview

| | What you see |
|---|---|
| **Live mode** (default) | Only real backend output. Today that is the **taxonomy**, the **Hinglish normalizer** and **entity extraction**. Every page whose backend module does not exist shows an honest empty state plus the interface it is waiting for. |
| **Design preview** (sidebar toggle) | Clearly labelled **DEMO DATA** so the full design can be reviewed. A banner appears on the page and charts carry a "DEMO DATA · synthetic" title. |

Live backend data always wins: if a loader returns real data, the demo is ignored
even when the toggle is on (tested).

### What is real today

* **Ticket Analysis**: the normalization stage runs the real `HinglishNormalizer`
  (`explain()`), including applied lexicon rules and lexicon version/fingerprint;
  the entity stage runs the real `supportai.entities.extract_entities` on the
  **raw** text. Entity chips also appear in Ticket Explorer's detail view.
* **Taxonomy** on Overview (issue types, departments, incident groups) and all
  display names, read from `config/taxonomy.yaml`.
* **Pipeline status** and the sidebar **Backend status** list, computed from which
  backend modules actually exist.

### What is preview-only (DEMO DATA)

Overview KPIs and charts, Ticket Explorer table and predictions, the full decision
trace tab, Incident Center, Incident Replay and Evaluation. In the demo,
`normalized_text` comes from the real normalizer and `entities` from the real
extractor; labels, priorities, confidences, detector states and metrics are synthetic (`dashboard/preview_data.py`).

## Layout

```
app.py                      entry point: streamlit run app.py
.streamlit/config.toml      theme (light main area, dark sidebar), telemetry off
dashboard/
  shell.py                  page config, sidebar, navigation
  theme.py                  design tokens + the single stylesheet
  components.py             reusable components (all dynamic text is escaped)
  charts.py                 Altair chart builders (DataFrame in, chart out)
  backend.py                THE ONLY door to backend modules (adapter)
  contracts.py              view models + table schemas the UI expects
  data_access.py            decides live vs demo vs nothing, in one place
  labels.py                 display names from the real taxonomy
  preview_data.py           DEMO DATA generator (isolated)
  compat.py                 Streamlit version shim (width="stretch")
  pages/                    overview, explorer, analysis, incidents, replay, evaluation
tests/test_dashboard.py     honesty, isolation, escaping, smoke and interaction tests
```

## Connecting a backend module (for the backend owner)

Pages never import backend intelligence modules. To connect one, edit **one
function in `dashboard/backend.py`** so it returns the shape in
`dashboard/contracts.py`, then set `adapter_ready=True` for its entry in
`CAPABILITY_SPECS`. No page changes are needed.

| Capability | Proposed module | Edit in `backend.py` | Return shape |
|---|---|---|---|
| Ingestion + classifier + urgency | `supportai.ingestion`, `.classifier`, `.urgency` | `load_tickets()` | `DataFrame` with `contracts.TICKET_COLUMNS` |
| Entities | `supportai.entities` | **connected** (`entity_views()` translates the dict) | `TicketAnalysis.entities: tuple[EntityView]` |
| Classifier + review routing | `supportai.classifier` | `analyze_ticket()` | `ClassificationView`, `department_id` |
| Urgency | `supportai.urgency` | `analyze_ticket()` | `UrgencyView` (factors, floors) |
| Playbooks | `supportai.playbooks` | `analyze_ticket()` | `PlaybookView` |
| Incident detector + explainer | `supportai.incidents` | `load_incidents()` | `list[IncidentView]` |
| Replay | `supportai.replay` | `load_replay()` | `ReplayData` |
| Evaluation | `supportai.evaluation` | `load_evaluation()` | `EvaluationReport` |

Module names and all shapes are **UI-side proposals**. Change them freely, but
change `contracts.py` and the adapter together.

`analyze_ticket()` already returns a `TicketAnalysis` with only `normalization`
filled in; each stage that stays `None` renders as "not connected". Filling a stage
makes its card go live automatically.

A malformed `load_tickets()` result raises a clear `ContractError` naming the
missing columns.

## Mapping the backend's synthetic CSVs

`scripts/generate_synthetic_data.py` writes the replay scenario with different
column names from the dashboard's view schema. The adapter should rename, not the
generator:

| Generator column | Dashboard column | Note |
|---|---|---|
| `ticket_id`, `customer_id`, `raw_text` | same | |
| `timestamp` | `created_at` | parse to datetime |
| `issue_type` | `issue_id` | becomes the *predicted* issue once the classifier runs; keep the generator's value as ground truth for evaluation |
| `department` | `department_id` | same: predicted by the backend, may be `human_review` |
| `incident_group` | `group_id` | empty string for `other` |
| `scenario` | `segment` | `background` → `normal`; `decoy_*` → `decoy`; `planted_*` → `planted` |
| `incident_id`, `is_incident_ticket` | (optional) | ground truth for replay and evaluation |
| (not in CSV) | `normalized_text`, `confidence`, `priority`, `needs_review`, `entities` | produced by the pipeline |

The generator's two decoys (promo burst from many distinct customers; one customer
repeating) are the same two decoy types the demo preview illustrates.

## Findings in `supportai/entities.py` (not changed by the dashboard)

The dashboard shows the extractor's real output, so two false-positive problems
are now visible in Ticket Analysis. Both come from patterns that accept any long
word after a keyword (because of `re.IGNORECASE`):

| Message | Extracted |
|---|---|
| `order nahi aaya` | order id `nahi` |
| `my order delayed` | order id `delayed` |
| `transaction failed` | transaction id `failed` |
| `I was refunded twice` | transaction id `refunded` |
| `no reference number given` | transaction id `reference` |

Suggested fix: require at least one digit in the captured identifier, for example
`...[\s:#_-]*(?=[A-Z0-9-]*\d)([A-Z0-9-]{4,})\b`, and add negative tests for the
phrases above. Captured values also keep leading punctuation (`#ORD-123456`,
`UTR: 4023918273`); decide whether that is intended. The dashboard deliberately
does not filter these results, to avoid duplicating or hiding backend behaviour.

## Assumptions to confirm

1. **Priority levels** are labelled P1 Critical, P2 High, P3 Medium, P4 Low
   (`contracts.PRIORITIES`). The UI only labels a priority string from the backend.
2. **Incident groups** have ids only in the taxonomy; display names are derived from
   the id (`labels.group_label`). Add display names to the taxonomy if you prefer.
3. The review threshold is not shown in live mode (the UI does not know it); the
   confidence bar shows a threshold marker only in the demo.

## Known limitations

* **Not verified in a real browser.** Charts were schema-validated and rendered to
  images, and pages run headlessly, but the final Streamlit styling (sidebar,
  native widgets, spacing) has not been seen in a browser. Please look at it once
  and report anything off.
* Fonts are a system stack (Segoe UI on Windows); no remote fonts are loaded.
* The dashboard stores no state (no database): incident "status" is display only.
* Dark mode is not supported; the theme is fixed to light with a dark sidebar.
