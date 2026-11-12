
from pathlib import Path

from data_preparation import prepare_data


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset location
DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "prepwise_cleaned_dataset (3).csv"
)


# Run the pipeline
df, daily_demand = prepare_data(DATA_PATH)


# --------------------------------------------------
# VALIDATION
# --------------------------------------------------

print("=" * 50)
print("MLPilot — Data Preparation Test")
print("=" * 50)

print("\nTransaction dataset shape:")
print(df.shape)

print("\nDaily demand dataset shape:")
print(daily_demand.shape)

print("\nDate range:")
print(daily_demand["date"].min())
print("to", daily_demand["date"].max())

print("\nMissing values:")
print(daily_demand.isnull().sum())

print("\nZero-demand days:")
print(
    (daily_demand["daily_demand"] == 0).sum()
)

print("\nFirst 5 rows:")
print(daily_demand.head())

print("\nLast 5 rows:")
print(daily_demand.tail())


# --------------------------------------------------
# ASSERTIONS
# --------------------------------------------------

assert df["invoicedate"].notna().all()
assert df["description"].notna().all()
assert df["price"].gt(0).all()

assert list(daily_demand.columns) == [
    "date",
    "daily_demand",
]

assert daily_demand["date"].is_monotonic_increasing
assert daily_demand["date"].is_unique
assert daily_demand["daily_demand"].notna().all()

print("\nAll data preparation checks passed!")