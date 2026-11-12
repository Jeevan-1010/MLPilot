import psycopg

from fastapi import APIRouter, HTTPException, Query

from src.tracking.mlflow_config import configure_mlflow


router = APIRouter()


@router.get("/admin/remediation-history")
def get_remediation_history(
    limit: int = Query(default=50, ge=1, le=200),
):
    """
    Return the latest remediation history records from Neon.
    """

    try:
        with psycopg.connect(configure_mlflow()) as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT COUNT(*)
                    FROM mlpilot_remediation_history
                """)
                total_records = cur.fetchone()[0]

                cur.execute("""
                    SELECT record
                    FROM mlpilot_remediation_history
                    ORDER BY id DESC
                    LIMIT %s
                """, (limit,))

                records = [row[0] for row in cur.fetchall()]

        return {
            "total_records": total_records,
            "returned_records": len(records),
            "records": records,
        }

    except psycopg.Error:
        raise HTTPException(
            status_code=500,
            detail="Could not retrieve remediation history from Neon.",
        )