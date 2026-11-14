import hashlib
import json
from datetime import datetime, timezone

import psycopg
from psycopg.types.json import Jsonb

from src.tracking.mlflow_config import configure_mlflow


def save_remediation_record(state: dict) -> dict:
    """
    Persist remediation execution history in Neon PostgreSQL.
    """

    result = state.get("retraining_result", {})
    action = state.get("remediation_action", {})
    approval = state.get("human_approval", {})

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "root_cause": state.get("root_cause"),
        "recommended_action": state.get("recommended_action"),
        "action_type": action.get("action_type"),
        "parameters": action.get("parameters", {}),
        "human_approval": approval,
        "remediation_status": state.get("remediation_status"),
        "deployment_status": state.get("deployment_status"),
        "deployment_result": state.get("deployment_result"),
        "deployment_approval": state.get("deployment_approval"),
        "candidate_model_version": result.get("model_version"),
        "evaluation": {
            "status": result.get("status"),
            "production_metrics": result.get("production_metrics"),
            "candidate_metrics": result.get("candidate_metrics"),
            "registered": result.get("registered", False),
            "reason": result.get("reason"),
        },
    }

    record_hash = hashlib.sha256(
        json.dumps(
            record,
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()

    with psycopg.connect(configure_mlflow()) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO mlpilot_remediation_history
                    (timestamp, record, record_hash)
                VALUES (%s, %s, %s)
                ON CONFLICT (record_hash) DO NOTHING
                """,
                (
                    record["timestamp"],
                    Jsonb(record),
                    record_hash,
                ),
            )

    return record