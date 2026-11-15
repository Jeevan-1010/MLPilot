# MLPilot

**An end-to-end MLOps platform for model training, monitoring, investigation, and human-approved deployment — supported by LangGraph-powered AI agents.**

**Core components:** AI Agents · Data Quality & Drift Monitoring · Model Performance Analysis · Automated Remediation & Retraining · Candidate Evaluation & MLflow Registry · Approval-Gated Deployment & Rollback.

MLPilot is an end-to-end, AI-agent-powered MLOps project built around a retail demand-forecasting use case using a Random Forest model. It brings together training and retraining, candidate evaluation, experiment tracking, model registration, FastAPI serving, continuous monitoring, investigation, and controlled production deployment. At its center is a LangGraph-based AI Investigator that coordinates diagnostic tools, evaluates evidence, and guides the workflow toward an appropriate next step.

The AI Investigator coordinates specialized diagnostic components for data quality, feature drift, data drift, and prediction-error analysis. Based on the evidence, it can recommend a corrective action, initiate the approved remediation and retraining pipeline, evaluate and register a candidate model, and wait for a separate human approval before production deployment. Deployment includes model reload and version verification, with rollback handling if validation fails.

The AI Investigator is the coordinating agent; monitoring, diagnostic, remediation, training, and deployment are specialized components and tools it orchestrates.

## Architecture

```text
Reference data + Random Forest model
                 |
                 v
         Training / Retraining
                 |
                 v
          Candidate evaluation
                 |
                 v
        MLflow tracking / registry
                 |
                 v
          Model deployment
                 |
                 v
          FastAPI serving
                 |
                 v
       Scheduled / manual monitoring
                 |
          +------+------+
          |             |
        Healthy      Alert / drift
                        |
                        v
                AI Investigator
              (LangGraph workflow)
                        |
                        v
              Diagnostic tools
                        |
                        v
               Root-cause decision
                        |
                        v
               Human approval #1
                        |
                        v
             Candidate remediation /
                    retraining
                        |
                        v
              Candidate evaluation
                        |
                        v
               Human approval #2
                        |
                        v
             Production deployment
                        |
                        v
             Reload and verify model
                        |
                        v
                Rollback on failure
```

The Investigator coordinates data-quality, feature-drift, and prediction-error diagnostics. Drift alone is not treated as proof of model degradation: when data quality is healthy and measured model performance has not degraded, the workflow can record a no-action outcome rather than trigger unnecessary retraining.

## Key capabilities

- **ML workflow:** Train and retrain the reference Random Forest demand-forecasting model, evaluate candidate results, and track experiments.
- **Model registry:** Register and manage model versions through MLflow.
- **Model serving:** Serve predictions through FastAPI and expose model health/version information.
- **Monitoring:** Run data-quality checks, feature-drift detection, and prediction-error monitoring.
- **AI Investigator:** Use a LangGraph workflow to select diagnostic tools, gather evidence, make a root-cause decision, and recommend next steps.
- **Human approval gates:** Require approval before remediation/candidate training and a separate approval before production deployment.
- **Deployment safeguards:** Check the candidate, promote the approved model, reload the serving API, verify the loaded version, and attempt rollback if validation fails.
- **Dashboard:** Use the Streamlit interface for predictions, monitoring, investigation, remediation history, and model management.
- **Scheduled monitoring:** Use GitHub Actions to trigger monitoring/investigation every six hours. The scheduled workflow cannot approve remediation or production deployment.

## Technology stack

| Area | Technologies |
|---|---|
| API and serving | Python, FastAPI, Uvicorn |
| Machine learning | scikit-learn, pandas, NumPy, SciPy |
| Experiment tracking / registry | MLflow |
| Agent orchestration | LangGraph, LangChain, Groq |
| Persistence / storage | PostgreSQL, Backblaze B2-compatible object storage |
| Dashboard | Streamlit |
| Packaging / CI | Docker, GitHub Actions |

## Evaluation results

The results below combine controlled benchmarks, automated tests, synthetic fixtures, and one observed monitoring run. Test-fixture and simulated results are labelled explicitly; they are not production SLAs or guarantees.

### API and inference performance benchmark

The serving configuration was optimized by changing the Random Forest `n_jobs` setting from `-1` to `1`. Before/after measurements were collected under the same benchmark setup.

| Metric | Before | After | Change |
|---|---:|---:|---:|
| Sequential average latency | 45.81 ms | 27.07 ms | 40.9% lower |
| Sequential P95 latency | 53.95 ms | 36.47 ms | 32.4% lower |
| Concurrent average latency | 234.57 ms | 129.73 ms | 44.7% lower |
| Concurrent P50 latency | 209.23 ms | 142.98 ms | 31.7% lower |
| Concurrent P95 latency | 409.35 ms | 192.95 ms | 52.9% lower |
| Concurrent throughput | 41.39 req/s | 74.15 req/s | 79.1% higher |
| Failed requests | 0 | 0 | Maintained |

These are benchmark-environment measurements, not a production latency or throughput SLA.

### Monitoring evaluation

The offline monitoring evaluation uses labelled synthetic fixtures:

- Data-drift tests cover stable and shifted data scenarios.
- The data-quality test set classified **5/5 cases correctly**.
- Drift tests enforce a recall threshold of at least 90% and a false-positive-rate ceiling of 20%; exact drift confusion-matrix values are not claimed here.

See [PR #11 — offline monitoring evaluation](https://github.com/Jeevan-1010/MLPilot/pull/11). These results describe the controlled test set, not production monitoring guarantees.

### Investigator and MLOps evaluation

Results below were measured in the offline evaluation suite using deterministic test doubles and synthetic data where noted.

| Evaluation | Result | Interpretation |
|---|---:|---|
| Investigator tool-selection contract | 5/5 (100%) | Deterministic LLM-double tests |
| Root-cause decision contract | 3/3 (100%) | Decision-contract tests, not live LLM accuracy |
| Full graph execution | 37.84 ms | Test graph with mocked components |
| Approval stages | 2/2 completed | Both workflow gates exercised |
| Training/evaluation pipeline | 618.17 ms average | Two synthetic-data runs |
| Reproducibility | 2/2 identical metric outputs | Compared within 1e-12 tolerance |
| Simulated deployment | 1/1 successful | Fake registry and mocked callbacks |
| Simulated rollback | 1/1 successful | Previous version restored in test |
| In-process API health checks | 20/20 successful | Test-client checks, not production uptime |
| Failure recovery | HTTP 500 then HTTP 200 | Injected failure followed by successful request |

Synthetic pipeline metrics:

| Metric | Baseline | Candidate |
|---|---:|---:|
| MAE | 2.1810 | 0.7409 |
| RMSE | 2.6360 | 1.0017 |

The MAE/RMSE comparison is from a synthetic evaluation fixture and must not be interpreted as a measured improvement on live retail demand data. Deployment and rollback figures are simulated test scenarios, not production deployment statistics.

See [PR #12 — Investigator, pipeline, and API reliability evaluations](https://github.com/Jeevan-1010/MLPilot/pull/12).

### Example monitoring observation

One monitoring run evaluated 30 records and reported:

| Metric | Observed result |
|---|---:|
| Reference MAE | 8,646.35 |
| Current MAE | 6,824.88 |
| Error change | 21.07% lower |
| Drift | Detected |
| Performance degradation | Not detected |
| Data quality | Healthy |

For this observation, the Investigator recommended continued monitoring rather than unnecessary remediation. This is a single run, not evidence of general production model performance.

### CI evaluation

The latest evaluation PR #12 run completed **57/57 tests successfully**. The Python test/syntax job and Docker image build job passed. The monitoring evaluation is tracked separately in PR #11. Both evaluation PRs are open and have not been merged; their results are documented here for review.

## Run locally

### Prerequisites

- Python 3.12
- Git
- An MLflow tracking backend and artifact store
- A registered reference model named `MLPilot-Demand-Forecasting` with a `production` alias for the full serving workflow
- The reference dataset at `data/prepwise_cleaned_dataset (3).csv`

The dataset is not included in this repository. Use data you are authorized to access and ensure it contains the fields expected by the reference-model preparation pipeline.

### Install

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

### Configure environment

Create a local `.env` file. Never commit credentials or share secret values.

| Variable | Purpose |
|---|---|
| `MLFLOW_TRACKING_URI` | MLflow tracking server / registry URI |
| `MLFLOW_S3_ENDPOINT_URL` | S3-compatible artifact endpoint when using B2 |
| `MLFLOW_ARTIFACTS_BUCKET` | Object-storage bucket for monitoring data |
| `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` | Object-storage credentials |
| `AWS_DEFAULT_REGION` | Object-storage region |
| `GROQ_API_KEY` | Groq API key for LLM investigation |
| `LANGGRAPH_CHECKPOINT_DB_URI` | PostgreSQL URI for persistent LangGraph checkpoints |
| `MLPILOT_ADMIN_TOKEN` | Secret required by `/admin/*` API endpoints |
| `MLPILOT_API_URL` | API base URL used by deployment verification / automation |

Keep `LANGGRAPH_CHECKPOINT_DB_URI` separate from `MLFLOW_TRACKING_URI`. Local development can use in-memory checkpoints when the PostgreSQL variable is omitted; investigations will not persist across process restarts in that mode.

### Start the API

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

API documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### Start the dashboard

In a second terminal:

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

Configure the dashboard API base URL and admin authentication through server-side environment variables or Streamlit secrets. Do not expose the admin token in browser-side code.

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
| `POST` | `/admin/investigator/{thread_id}/approve` | Approve or reject the pending stage |
| `POST` | `/admin/reload-model` | Reload the production model |
| `GET` | `/admin/remediation-history` | Retrieve remediation history |

All `/admin/*` endpoints require the `X-Admin-Token` header. See [the API contract](docs/FRONTEND_API_CONTRACT.md) for request/response details and approval-stage behavior.

## Human approval and deployment safety

1. An investigation may conclude with no action; no candidate is trained in that case.
2. If remediation is recommended, a human must approve before candidate training/registration proceeds.
3. A passing candidate requires a separate human approval before production deployment.
4. Before deployment, the workflow checks that the production model has not changed since candidate evaluation.
5. After approval, MLPilot promotes the candidate, reloads the serving API, and verifies the loaded model version.
6. If deployment validation fails, MLPilot attempts to restore the previous production alias and reload the previous model.

The scheduled workflow can trigger monitoring and investigation, but it cannot bypass either approval gate.

## Tests

Run the repository test suite and syntax checks:

```bash
python -m pytest -v
python -m compileall -q src
python -m py_compile frontend/app.py
```

GitHub Actions validates tests, backend/frontend syntax, and the Docker image build. Benchmark scripts in `tests/` can be run in a controlled environment; record the environment and workload when comparing results.

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
notebooks/      Reference demand-forecasting development
```

## Scope and limitations

- The current reference implementation uses a Random Forest model for retail demand forecasting. The README does not claim support for training or retraining across multiple algorithms.
- Monitoring uses reference/recent data windows. Thresholds and window sizes should be calibrated for the dataset and model.
- The deployed workflow has a process-local active-investigation guard. A multi-instance deployment needs a database-backed distributed lock to coordinate investigation starts.
- Human approval is intentionally required before remediation and production deployment.
- Offline tests, synthetic fixtures, mocked LLM/registry components, and local benchmarks do not replace broader live operational testing.

## Live demo

Try the deployed MLPilot dashboard:

**[Open MLPilot Dashboard](https://mlpilot-dashboard.onrender.com/)**

## Project status

I built MLPilot around a Random Forest-based retail demand-forecasting workflow. The production API and model are currently running. The project includes model training and retraining, evaluation, MLflow tracking and registry integration, monitoring, a LangGraph AI Investigator, a two-stage human approval process, the Streamlit dashboard, and scheduled monitoring/investigation.

The evaluation and documentation updates are available in separate pull requests for review.

## Author

**Jeevan M**
