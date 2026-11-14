
import pandas as pd


# --------------------------------------------------
# 1. LOAD AND PREPARE TRANSACTION DATA
# --------------------------------------------------

def load_and_prepare_data(data_path):
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

    df = pd.read_csv(data_path, usecols=required_columns)

    df = df[df["price"] > 0]
    df["invoicedate"] = pd.to_datetime(df["invoicedate"])
    df = df.dropna(subset=["description"])
    df["customer_id"] = df["customer_id"].fillna("Unknown")
    df["is_cancelled"] = df["invoice"].astype(str).str.startswith("C")
    df["revenue"] = df["quantity"] * df["price"]

    return df


# --------------------------------------------------
# 2. CREATE DAILY DEMAND DATASET
# --------------------------------------------------

def create_daily_demand(df):
    sales_df = df.loc[
        df["quantity"] > 0,
        ["invoicedate", "quantity"],
    ]

    daily_demand = (
        sales_df.groupby(
            sales_df["invoicedate"].dt.normalize()
        )["quantity"]
        .sum()
        .reset_index()
    )

    daily_demand.columns = ["date", "daily_demand"]

    all_dates = pd.date_range(
        start=daily_demand["date"].min(),
        end=daily_demand["date"].max(),
        freq="D",
    )

    daily_demand = (
        daily_demand.set_index("date")
        .reindex(all_dates, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )

    return daily_demand


# --------------------------------------------------
# 3. ORIGINAL PIPELINE — PRESERVES TWO-VALUE RETURN
# --------------------------------------------------

def prepare_data(data_path):
    df = load_and_prepare_data(data_path)
    daily_demand = create_daily_demand(df)

    return df, daily_demand


# --------------------------------------------------
# 4. LIGHTWEIGHT DAILY DEMAND PREPARATION
# --------------------------------------------------

def prepare_daily_demand(data_path):
    """
    Load only the columns required for monitoring.
    Avoids loading the full transaction dataset.
    """

    df = pd.read_csv(
        data_path,
        usecols=["description", "price", "quantity", "invoicedate"],
    )

    df = df.loc[
        (df["price"] > 0)
        & df["description"].notna()
        & (df["quantity"] > 0),
        ["invoicedate", "quantity"],
    ]

    df["invoicedate"] = pd.to_datetime(df["invoicedate"])

    daily_demand = (
        df.groupby(
            df["invoicedate"].dt.normalize()
        )["quantity"]
        .sum()
        .rename("daily_demand")
        .reset_index()
    )

    daily_demand.columns = ["date", "daily_demand"]

    if daily_demand.empty:
        raise ValueError("No valid positive-demand records found.")

    all_dates = pd.date_range(
        start=daily_demand["date"].min(),
        end=daily_demand["date"].max(),
        freq="D",
    )

    daily_demand = (
        daily_demand.set_index("date")
        .reindex(all_dates, fill_value=0)
        .rename_axis("date")
        .reset_index()
    )

    return daily_demand
