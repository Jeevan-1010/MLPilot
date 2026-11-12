
import numpy as np
import pandas as pd


def check_data_quality(
    data: pd.DataFrame,
    required_columns: list,
    numeric_columns: list,
    non_negative_columns: list = None
) -> dict:

    issues = {}

    # 1. Check missing values
    missing_values = data[required_columns].isnull().sum()
    missing_values = missing_values[missing_values > 0]

    if not missing_values.empty:
        issues["missing_values"] = missing_values.to_dict()

    # 2. Check duplicate records
    duplicate_count = int(data.duplicated().sum())

    if duplicate_count > 0:
        issues["duplicate_records"] = duplicate_count

    # 3. Check invalid numeric values
    invalid_values = {}

    for column in numeric_columns:
        if column not in data.columns:
            continue

        values = pd.to_numeric(data[column], errors="coerce")

        invalid_count = int(
            (values.isna() & data[column].notna()).sum()
        )

        infinite_count = int(
            np.isinf(values.dropna()).sum()
        )

        if invalid_count > 0 or infinite_count > 0:
            invalid_values[column] = {
                "invalid_values": invalid_count,
                "infinite_values": infinite_count
            }

    if invalid_values:
        issues["invalid_numeric_values"] = invalid_values

    # 4. Check negative values where not allowed
    negative_values = {}

    for column in (non_negative_columns or []):
        if column not in data.columns:
            continue

        values = pd.to_numeric(data[column], errors="coerce")
        count = int((values < 0).sum())

        if count > 0:
            negative_values[column] = count

    if negative_values:
        issues["negative_values"] = negative_values

    # 5. Overall quality status
    return {
        "total_records": len(data),
        "quality_status": "ISSUES_FOUND" if issues else "HEALTHY",
        "issues": issues
    }
