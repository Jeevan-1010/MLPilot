
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# 1. LOAD AND PREPARE TRANSACTION DATA
# --------------------------------------------------

def load_and_prepare_data(data_path):
    """
    Load the cleaned transaction CSV and apply the
    subject-model-specific preparation from the notebook.
    """

    df = pd.read_csv(data_path)

    required_columns = [
        "invoice",
        "stockcode",
        "description",
        "quantity",
        "invoicedate",
        "price",
        "customer_id",
        "country",
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Remove negative-price records
    df = df[df["price"] >= 0].copy()

    # Convert invoice date to datetime
    df["invoicedate"] = pd.to_datetime(
        df["invoicedate"]
    )

    # Remove records without descriptions
    df = df.dropna(
        subset=["description"]
    ).copy()

    # Replace missing customer IDs
    df["customer_id"] = (
        df["customer_id"].fillna("Unknown")
    )

    # Remove zero-price records
    df = df[df["price"] > 0].copy()

    # Identify cancellations
    df["is_cancelled"] = (
        df["invoice"]
        .astype(str)
        .str.startswith("C")
    )

    # Calculate revenue
    df["revenue"] = (
        df["quantity"] * df["price"]
    )

    return df


# --------------------------------------------------
# 2. CREATE DAILY DEMAND DATASET
# --------------------------------------------------

def create_daily_demand(df):
    """
    Create daily demand using positive quantities only,
    matching the notebook's demand aggregation logic.
    """

    # Keep only genuine sales
    sales_df = df[
        df["quantity"] > 0
    ].copy()

    # Extract date without time
    sales_df["date"] = (
        sales_df["invoicedate"].dt.date
    )

    # Aggregate quantity sold per day
    daily_demand = (
        sales_df.groupby("date")["quantity"]
        .sum()
        .reset_index()
    )

    # Rename target column
    daily_demand.rename(
        columns={"quantity": "daily_demand"},
        inplace=True,
    )

    # Convert date back to datetime
    daily_demand["date"] = pd.to_datetime(
        daily_demand["date"]
    )

    # Create complete daily calendar
    all_dates = pd.date_range(
        start=daily_demand["date"].min(),
        end=daily_demand["date"].max(),
        freq="D",
    )

    # Fill missing dates with zero demand
    daily_demand = (
        daily_demand
        .set_index("date")
        .reindex(all_dates, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )

    return daily_demand


# --------------------------------------------------
# 3. COMPLETE DATA PREPARATION PIPELINE
# --------------------------------------------------

def prepare_data(data_path):
    """
    Run the complete data preparation workflow.
    """

    df = load_and_prepare_data(data_path)

    daily_demand = create_daily_demand(df)

    return df, daily_demand