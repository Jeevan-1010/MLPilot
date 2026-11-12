
import json
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query


router = APIRouter()

BASE_DIR = Path(__file__).resolve().parents[2]
HISTORY_PATH = BASE_DIR / "artifacts" / "remediation_history.jsonl"


@router.get("/admin/remediation-history")
def get_remediation_history(
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    Return the latest remediation history records.
    """

    if not HISTORY_PATH.exists():
        return {
            "total_records": 0,
            "returned_records": 0,
            "records": [],
        }

    records = []

    try:
        with HISTORY_PATH.open("r", encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue

                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    raise HTTPException(
                        status_code=500,
                        detail=(
                            "Invalid JSON in remediation history "
                            f"at line {line_number}."
                        ),
                    )

        latest_records = list(reversed(records))[:limit]

        return {
            "total_records": len(records),
            "returned_records": len(latest_records),
            "records": latest_records,
        }

    except HTTPException:
        raise

    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not read remediation history: {error}",
        )
