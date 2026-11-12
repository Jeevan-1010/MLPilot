import json
from pathlib import Path
from datetime import datetime, timezone


BASE_DIR = Path(__file__).resolve().parents[2]
HISTORY_PATH = BASE_DIR / "artifacts" / "remediation_history.jsonl"


def save_remediation_record(state: dict) -> dict:
    """
    Persist one remediation execution as an append-only JSONL record.
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
        "evaluation": {
            "status": result.get("status"),
            "production_metrics": result.get("production_metrics"),
            "candidate_metrics": result.get("candidate_metrics"),
            "registered": result.get("registered", False),
            "reason": result.get("reason"),
        },
    }

    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)

    with HISTORY_PATH.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, default=str) + "\n")

    return record
