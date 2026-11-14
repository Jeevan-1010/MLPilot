# MLPilot Backend API Contract

Base URL: the deployed FastAPI service URL. Interactive API documentation is available at `/docs`.

All `/admin/*` endpoints require the `X-Admin-Token` header. Never expose this token in browser code. Frontend requests to admin endpoints must go through a trusted backend/proxy or use a separate authenticated session design; do not embed the Render secret in Streamlit/browser JavaScript.

## Existing endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status and loaded model version |
| GET | `/health` | Model/schema health |
| GET | `/metrics` | API request/latency counters |
| POST | `/predict` | Demand prediction |
| POST | `/admin/reload-model` | Reload current MLflow production alias |
| POST | `/admin/monitoring/run` | Run monitoring only |
| GET | `/admin/remediation-history` | Read persisted remediation history |

## Investigator workflow

### Start a monitoring + investigation cycle

`POST /admin/investigator/start`

Headers: `X-Admin-Token: <server-side token>`

No request body is required.

Possible response statuses:
- `awaiting_remediation_approval`: investigator found an actionable remediation and is waiting for approval.
- `awaiting_deployment_approval`: candidate is ready for the second approval.
- `completed`: no action was needed or the workflow completed without a pending approval.
- `already_pending`: an earlier investigation is still waiting for approval.

Save the returned `thread_id`. It identifies this workflow run and must be reused for status and approval requests.

### Get workflow state

`GET /admin/investigator/{thread_id}`

Returns current status, monitoring report, root-cause analysis, proposed action, candidate evaluation and deployment details. When awaiting deployment approval, `pending_approval` includes candidate version and metrics.

### Approve or reject the current stage

`POST /admin/investigator/{thread_id}/approve`

Request JSON:

```json
{
  "approved": true,
  "feedback": "Optional human feedback"
}
```

The same endpoint handles both stages. It resumes whichever interrupt is pending:
1. Remediation approval: `approved=true` permits only the whitelisted remediation action and candidate evaluation/registration. Rejection ends the workflow without training.
2. Deployment approval: `approved=true` promotes the registered candidate, reloads the serving model, and verifies the loaded version. A failed deployment attempts an automatic alias rollback and serving-model reload. Rejection leaves production unchanged.

A successful first approval can return `awaiting_deployment_approval`; render the candidate metrics and ask for a second, explicit approval. Never auto-approve either stage.

## Frontend implementation notes

- Treat `thread_id` as the workflow identifier and persist it in the frontend session so the user can refresh and retrieve state.
- Render distinct states: running, awaiting remediation approval, awaiting deployment approval, completed, rejected, deployed, rolled back, failed.
- Display the monitoring report, diagnostic results, root cause, recommended action, action parameters, production/candidate MAE and RMSE, candidate model version, and deployment result.
- Show separate approve/reject controls at each approval stage. Include an optional feedback field.
- Poll `GET /admin/investigator/{thread_id}` while the workflow is running or after a page refresh.
- Do not expose `MLPILOT_ADMIN_TOKEN`, Neon credentials, MLflow credentials, Groq key, or B2 credentials in frontend source or client-side requests.
- Scheduled monitoring runs every six hours through GitHub Actions. It starts the same API workflow; it does not bypass either human approval checkpoint.

## Operational configuration

The API needs its existing MLflow, Neon, Groq and Backblaze B2 environment variables. In addition:
- `MLPILOT_ADMIN_TOKEN`: strong secret used by all admin endpoints and the scheduled workflow.
- `LANGGRAPH_CHECKPOINT_DB_URI`: Neon PostgreSQL connection URI used for persistent LangGraph checkpoints. Keep this separate from `MLFLOW_TRACKING_URI`.
- GitHub Actions repository secrets: `MLPILOT_API_URL` and `MLPILOT_ADMIN_TOKEN`.

The workflow's active-investigation guard is process-local. PostgreSQL preserves graph checkpoints across restarts, but a multi-instance deployment should use a database-backed global lock before running multiple API instances.
