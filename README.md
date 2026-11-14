# MLPilot

**A model-agnostic MLOps platform with an AI Investigator Agent.**

MLPilot demonstrates an end-to-end machine-learning operations workflow using a retail demand-forecasting model as its reference implementation. The platform separates the MLOps orchestration from the subject model so the workflow can be extended to other models.

## What it does

MLPilot connects the following lifecycle:

```text
Data + Model
    |
    v
Training & Evaluation -----> MLflow Tracking / Model Registry
                                      |
                                      v
                               Model Deployment
                                      |
                                      v
                                Model Serving
                                      |
                                      v
                         Scheduled / Manual Monitoring
                                      |
                           +----------+----------+
                           |                     |
                         Healthy              Alert / Drift
                                                 |
                                                 v
                                        AI Investigator
                                      (LangGraph workflow)
                                                 |
                                   Diagnostic tool selection
                                                 |
                                  Root-cause analysis
                                                 |
                                    Human approval #1
                                                 |
                                  Remediation + retraining
                                                 |
                                   Candidate evaluation
                                                 |
                                    Human approval #2
                                                 |
                                   Promote & reload model
                                                 |
                                  Health/version verification
                                                 |
                                    Rollback on failure
```

The Investigator uses diagnostic tools for data quality, feature/data drift, and prediction-error analysis. It recommends a corrective action based on available evidence. Drift by itself is not treated as proof of model degradation: when data quality is healthy and measured performance has not degraded, the workflow records a no-action outcome instead of proposing retraining.

## Main capabilities

- **Training and evaluation:** time-series cross-validation, MAE/RMSE comparison against the current production model, and candidate evaluation.
- **Experiment tracking and registry:** MLflow runs, model versions, and a production model alias.
- **Schema-aware serving:** FastAPI prediction endpoints load the registered production model and its feature schema.
- **Monitoring:** data-quality checks, Kolmogorov–Smirnov feature-drift tests, and model-error monitoring.
- **AI investigation:** LangGraph coordinates diagnostic tool selection, evidence-based root-cause analysis, and recommendations.
- **Human-in-the-loop controls:** separate approval gates for remediation and production deployment.
- **Deployment safeguards:** candidate version checks, serving health/version verification, and production-alias rollback if deployment validation fails.
- **Persistence:** PostgreSQL-backed LangGraph checkpoints in the deployed API configuration and persisted remediation history.
- **Dashboard:** Streamlit pages for overview, predictions, monitoring, AI investigation, remediation history, and model management.
- **Automation:** GitHub Actions runs the monitoring/investigation trigger on a six-hour schedule. It does not approve remediation or deployment.

## Technology

| Area | Technologies |
|---|---|
| API and serving | Python, FastAPI, Uvicorn |
| ML | scikit-learn, pandas, NumPy, SciPy |
| Experiment tracking / registry | MLflow |
| Agent orchestration | LangGraph, LangChain, Groq |
| Persistence | PostgreSQL, Backblaze B2-compatible object storage |
| Dashboard | Streamlit |
| Packaging / CI | Docker, GitHub Actions |

## Repository layout

```text
src/
  agents/       Investigator graph, diagnostics, remediation tools
  api/          FastAPI endpoints and checkpoint integration
  monitoring/   Data-quality, drift, and performance monitoring
  pipeline/     Feature engineering, training, prediction, deployment
  tracking/     MLflow configuration and tracking helpers
frontend/       Streamlit dashboard
tests/          Automated tests and benchmark scripts
docs/           Frontend/backend API contract
models/         Local baseline model artifact
notebooks/      Demand-forecasting model development
```

## Run locally

### 1. Requirements

- Python 3.12
- Git
- A configured MLflow tracking backend and artifact store
- A registered model named `MLPilot-Demand-Forecasting` with a `production` alias
- The subject-model dataset at `data/prepwise_cleaned_dataset (3).csv`

The dataset is not included in this repository. Supply a dataset you are authorized to use and ensure it follows the columns expected by the demand-forecasting preparation pipeline.

### 2. Install dependencies

```bash
git clone https://github.com/Jeevan-1010/MLPilot.git
cd MLPilot

python -m venv .venv
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure environment

Create a local `.env` file. Keep credentials out of Git and never commit this file.

| Variable | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | MLflow tracking server / registry URI |
| `MLFLOW_S3_ENDPOINT_URL` | S3-compatible artifact endpoint when using B2 |
| `MLFLOW_ARTIFACTS_BUCKET` | Object-storage bucket used for monitoring data |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Object-storage credentials |
| `AWS_DEFAULT_REGION` | Object-storage region |
| `GROQ_API_KEY` | Groq API key for LLM-based investigation |
| `LANGGRAPH_CHECKPOINT_DB_URI` | PostgreSQL URI for persistent LangGraph checkpoints |
| `MLPILOT_ADMIN_TOKEN` | Secret required by all `/admin/*` API endpoints |
| `MLPILOT_API_URL` | API base URL used by deployment verification and automation |

Use a PostgreSQL URI for `LANGGRAPH_CHECKPOINT_DB_URI`. Keep it separate from `MLFLOW_TRACKING_URI`. The API can use an in-memory checkpoint saver for local development when the PostgreSQL variable is omitted; that mode does not persist investigations across process restarts.

### 4. Start the API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### 5. Start the dashboard

In a second terminal:

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

Configure the dashboard's API base URL and admin authentication values through its server-side environment/secrets configuration. Do not put the admin token in browser-side code or commit credentials.

## API overview

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | API status and loaded model version |
| `GET` | `/health` | Serving model/schema health |
| `GET` | `/metrics` | Request and latency counters |
| `POST` | `/predict` | Generate a prediction |
| `POST` | `/admin/monitoring/run` | Run monitoring |
| `POST` | `/admin/investigator/start` | Start or find an investigation |
| `GET` | `/admin/investigator/{thread_id}` | Retrieve investigation state |
| `POST` | `/admin/investigator/{thread_id}/approve` | Approve/reject the pending stage |
| `POST` | `/admin/reload-model` | Reload the production model |
| `GET` | `/admin/remediation-history` | Retrieve remediation history |

All `/admin/*` endpoints require the `X-Admin-Token` header. See [the API contract](docs/FRONTEND_API_CONTRACT.md) for request/response behavior and approval-stage details.

## Safety and deployment behavior

1. An investigation can finish with no action; no candidate is trained in that case.
2. If a corrective action is recommended, a human must approve remediation before candidate training/registration.
3. A passing candidate waits for a separate human deployment approval.
4. Deployment checks that production has not changed since candidate evaluation.
5. The production alias is promoted, the serving API is reloaded, and the loaded model version is verified.
6. If deployment validation fails, MLPilot attempts to restore the previous production alias and reload the previous model.

The scheduled workflow only starts the investigation. It never bypasses either approval gate.

## Tests and CI

Run the automated test suite:

```bash
python -m pytest -v
python -m compileall -q src
python -m py_compile frontend/app.py
```

GitHub Actions runs the test suite, frontend syntax validation, backend syntax checks, and Docker image build on pushes/pull requests to `main`.

Benchmark scripts are available under `tests/` for API, batch, concurrent-request, and model-inference measurements. Run them against a controlled environment before publishing performance figures; do not treat local benchmark results as production SLAs.

## Current scope and limitations

- The included subject-model implementation is retail demand forecasting; the MLOps workflow is designed to be extended to other model types.
- Monitoring currently compares fixed recent/reference data windows. Thresholds and window sizes are configurable in code and should be calibrated for a new dataset/model.
- The deployed workflow uses a process-local active-investigation guard. PostgreSQL checkpoints persist workflow state, but a multi-instance deployment needs a database-backed distributed lock to coordinate starts across instances.
- Human approval is intentionally required before remediation and production deployment.
- A successful CI run validates tests, syntax, and the Docker build; it does not replace a live end-to-end deployment/rollback exercise.

## Project status

Active project. The demand-forecasting reference implementation, monitoring pipeline, LangGraph Investigator, approval workflow, Streamlit dashboard, CI checks, and scheduled investigation trigger are implemented. Further model adapters, multi-instance coordination, and broader live operational testing remain future work.
